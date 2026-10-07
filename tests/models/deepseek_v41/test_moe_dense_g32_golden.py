# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P5-MOE: per-layer L-REF golden gates with the attention projections kept FP8 group-32 (D9 switch).

The modules of L-ATTN's golden harness are built as usual (FP16 fallback weights) and then their switched projections
(``qc.G32_PROJECTIONS``: ``fused_wqa_wkv``, ``wq_b``, ``wo_a`` grouped) are rebuilt from the RAW checkpoint bytes + UE8M0
scales through the quant method's own ``_keep_fp8_g32`` -- exactly what ``process_weights_after_loading`` does when
``VLLM_DS41_MOE_DENSE_G32`` is on. Gates = L-ATTN's for the FP16 fallback (PORT_DESIGN §4.5):

* ``test_g32_golden_per_op``: golden attn.x -> qr, q (<= 2e-3), window-KV records from the FP16 x (<= 3 % flips),
  golden attn.o -> o-proj (<= 2e-3), golden attn.qr -> indexer q after RoPE + FP4 QAT through
  ``v41_linear_fp32_input`` on a g32 copy of indexer.wq_b (bitwise == golden and == fallback); every link also
  reported against the same link on the FP16 fallback weights. Report-only: QAT flips of the FP32-x g32 GEMV (the
  measurement behind keeping indexer.wq_b off the switch: 1e-6..3e-6 of values, so not bitwise).

Requires the call-site seam in ``attention.py`` (DCR in P5-MOE.progress.md); skipped with that reason until merged.
* ``test_g32_golden_layer_end_to_end``: L-ATTN's golden-input end-to-end layer test (chunked prefill + decode1..4
  through ``DeepseekV41Attention`` on the simulated paged cache), run unchanged on the g32 modules.

Needs the goldens ($DS41_GOLDEN_DIR) and verified shards; marked ``weights``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vllm.models.deepseek_v41 import quant_config as qc

from . import test_attn_layers
from .test_attn_golden import (  # noqa: F401  (fixture)
    PROMPTS,
    _require,
    _weights,
    weights_cache,
)
from .test_attn_golden import test_golden_layer_end_to_end as _l_attn_end_to_end
from .test_attn_harness import REF_DIR, rel_rms, topology, verified_shard
from .test_attn_layers import DEV, Stage, dist_env  # noqa: F401  (fixture)

pytestmark = [pytest.mark.sm70, pytest.mark.weights]

SWITCHED = {  # module attribute -> checkpoint names (rows concatenated in this order)
    "fused_wqa_wkv": ("attn.wq_a", "attn.wkv"),
    "wq_b": ("attn.wq_b",),
    "wo_a": ("attn.wo_a",),
    "indexer.wq_b": ("attn.indexer.wq_b",),
}
_RAW: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}


def _seam_present() -> bool:
    from vllm.models.deepseek_v41 import attention
    return hasattr(attention, "v41_linear") and hasattr(attention, "v41_grouped_linear")


pytestmark.append(pytest.mark.skipif(not _seam_present(), reason="attention.py lacks the P5-MOE call-site seam "
                                     "(v41_linear / v41_grouped_linear; DCR in P5-MOE.progress.md)"))


def load_raw_fp8(layer_id: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """``attn.<proj>`` -> (E4M3 weight, UE8M0 scale) exactly as stored, on the GPU, from verified shards."""
    if layer_id in _RAW:
        return _RAW[layer_id]
    from safetensors import safe_open

    idx = json.loads((Path(REF_DIR) / "model.safetensors.index.json").read_text())["weight_map"]
    want = {n for names in SWITCHED.values() for n in names}
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for short in sorted(want):
        key = f"layers.{layer_id}.{short}.weight"
        if key not in idx:
            continue
        path = verified_shard(idx[key])
        with safe_open(str(path), framework="pt", device="cpu") as f:
            w = f.get_tensor(key)
        spath = verified_shard(idx[key[: -len(".weight")] + ".scale"])
        with safe_open(str(spath), framework="pt", device="cpu") as f:
            s = f.get_tensor(key[: -len(".weight")] + ".scale")
        if w.dtype != torch.float8_e4m3fn:
            raise TypeError(f"{key}: expected E4M3, got {w.dtype}")
        out[short] = (w.to(DEV), s.to(DEV))
    _RAW[layer_id] = out
    return out


def attn_to_g32(attn, raw: dict[str, tuple[torch.Tensor, torch.Tensor]], tp_rank: int = 0, tp_size: int = 1) -> None:
    """Rebuild the switched projections of a DeepseekV41Attention as group-32 FP8 (TP rows as load_into_module)."""
    hq = 64 // tp_size * 512
    ga = 8 // tp_size * 1024
    rows = {"wq_b": (tp_rank * hq, hq), "wo_a": (tp_rank * ga, ga)}
    for attr, names in SWITCHED.items():
        if f"attn.{attr}" not in qc.G32_PROJECTIONS:
            continue
        lin = attn
        for part in attr.split("."):
            lin = getattr(lin, part, None)
            if lin is None:
                break
        if lin is None:
            continue
        ws = [raw[n][0] for n in names]
        ss = [raw[n][1] for n in names]
        w, s = torch.cat(ws), torch.cat(ss)
        if attr in rows:
            r0, nr = rows[attr]
            w, s = w[r0:r0 + nr], s[r0 // 32:(r0 + nr) // 32]
        before = lin.weight.float().clone()
        qc.DeepseekV41SM70Fp8LinearMethod._keep_fp8_g32(lin, w.contiguous(), s.contiguous())
        from vllm.models.deepseek_v41.sm70.fp8_g32 import dequant_fp8_g32

        # the g32 operands hold exactly the FP16 fallback's weight the harness loaded
        assert torch.equal(dequant_fp8_g32(lin.ds41_g32).t().float(), before), attr


@pytest.fixture(autouse=True)
def _g32_modules(monkeypatch):
    original = test_attn_layers.load_into_module

    def load_then_switch(attn, w, tp_rank: int = 0, tp_size: int = 1) -> None:
        original(attn, w, tp_rank, tp_size)
        attn_to_g32(attn, load_raw_fp8(attn.layer_id), tp_rank, tp_size)

    monkeypatch.setattr(test_attn_layers, "load_into_module", load_then_switch)


@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("layer", [0, 1, 2, 3, 14, 20, 21, 24])
def test_g32_golden_per_op(dist_env, weights_cache, prompt: str, layer: int) -> None:    # noqa: F811
    from vllm.models.deepseek_v41.common.rope import apply_rope_torch
    from vllm.models.deepseek_v41.compressor import mm_fp32_full
    from vllm.models.deepseek_v41.sm70.indexer_kernels import index_q_rope_qat
    from vllm.models.deepseek_v41.sm70.q_rope_kv_insert import q_rope_kv_insert

    cases = _require(prompt, (layer,))
    from .test_attn_harness import ref_config
    cfg = ref_config()
    topo = topology(cfg, layer)
    w = _weights(weights_cache, layer)
    st = Stage(cfg, (layer,), {layer: w})                 # g32 modules (autouse fixture)
    attn = st.attn[layer]
    assert attn.fused_wqa_wkv.ds41_g32 is not None and attn.wo_a.ds41_g32_groups == 8
    fb = {"qkv": w["attn.wq_a.weight"].half(), "wkv": w["attn.wkv.weight"].half(), "qb": w["attn.wq_b.weight"].half()}
    report: dict = {}
    for case in cases:
        def g(n: str, _case=case) -> torch.Tensor:
            return _case.get(layer, n)
        T, start = case.num_tokens, case.start
        pos = torch.arange(start, start + T, device=DEV)
        subset = case.manifest.get("row_subset") or {}
        rows = torch.arange(T, device=DEV)
        if "attn.q" in subset.get("tensors", ()):
            rows = torch.tensor(subset["rows"], device=DEV)
        x16 = g("attn.x").half()
        qr_kv = qc.v41_linear(attn.fused_wqa_wkv, x16, torch.float32)
        qr32 = attn.q_norm(qr_kv[:, :1280])
        q = qc.v41_linear(attn.wq_b, qr32.half()[rows].contiguous(), torch.float16).view(rows.numel(), 64, 512)
        q = apply_rope_torch(q, pos[rows], attn.rotary_emb.cos_sin_cache)
        report.setdefault("qr", []).append(rel_rms(qr32, g("attn.qr")))
        report.setdefault("q", []).append(rel_rms(q, g("attn.q")))
        # the same links on the FP16 fallback weights (the P2 numbers these replace)
        fb_qr_kv = torch.mm(x16, torch.cat([fb["qkv"], fb["wkv"]]).t(), out_dtype=torch.float32)
        fb_qr32 = attn.q_norm(fb_qr_kv[:, :1280])
        fb_q = torch.mm(fb_qr32.half()[rows], fb["qb"].t()).view(rows.numel(), 64, 512)
        fb_q = apply_rope_torch(fb_q, pos[rows], attn.rotary_emb.cos_sin_cache)
        report.setdefault("qr_fallback", []).append(rel_rms(fb_qr32, g("attn.qr")))
        report.setdefault("q_fallback", []).append(rel_rms(fb_q, g("attn.q")))
        report.setdefault("qr_kv_g32_vs_fallback", []).append(rel_rms(qr_kv, fb_qr_kv))
        kv_rows = torch.empty(T, 512, dtype=torch.float16, device=DEV)
        q_rope_kv_insert(torch.zeros(T, 1, 512, dtype=torch.float16, device=DEV), attn.kv_norm(qr_kv[:, 1280:]), pos,
                         attn.rotary_emb.cos_sin_cache, kv_rows, torch.arange(T, device=DEV), impl="sm70")
        report.setdefault("kv_win_mismatch_x16", []).append(float((kv_rows.float() != g("attn.kv_win")).float().mean()))
        out = attn.output_projection(g("attn.o").half().contiguous(), pos[rows])
        report.setdefault("oproj", []).append(rel_rms(out, g("attn.out")[rows]))
        if topo.owns_indexer and case.has(layer, "idx.q"):
            from vllm.models.deepseek_v41.sm70.fp8_g32 import (
                fp8_g32_gemv,
                prepare_fp8_g32,
            )
            ig = type("G32Indexer", (), {})()
            ig.ds41_g32, ig.ds41_g32_groups = prepare_fp8_g32(*load_raw_fp8(layer)["attn.indexer.wq_b"]), 1
            pq = qc.v41_linear_fp32_input(ig, g("attn.qr")).view(T, 32, 128)
            pq = index_q_rope_qat(pq, pos, attn.rotary_emb.cos_sin_cache)
            fq = mm_fp32_full(g("attn.qr"), w["attn.indexer.wq_b.weight"].half()).view(T, 32, 128)
            fq = index_q_rope_qat(fq, pos, attn.rotary_emb.cos_sin_cache)
            report.setdefault("idx_q_mismatch", []).append(float((pq.float() != g("idx.q")).float().mean()))
            # report-only: the FP32-x group-32 GEMV (decode candidate, 8 rows per call) through the same RoPE + QAT
            qr_g = g("attn.qr")
            gq = torch.cat([fp8_g32_gemv(qr_g[r:r + 8], ig.ds41_g32, out_dtype=torch.float32)
                            for r in range(0, T, 8)]).view(T, 32, 128)
            gq = index_q_rope_qat(gq, pos, attn.rotary_emb.cos_sin_cache)
            report.setdefault("idx_q_gemv_mismatch_count", []).append(int((gq.float() != g("idx.q")).sum()))
            report.setdefault("idx_q_values", []).append(int(gq.numel()))
            report.setdefault("idx_q_g32_vs_fallback_bitwise", []).append(bool(torch.equal(pq, fq)))
    print(prompt, layer, {k: [v if isinstance(v, (list, bool)) else round(v, 7) for v in vals]
                          for k, vals in report.items()})
    assert max(report["q"]) <= 2e-3 and max(report["qr"]) <= 2e-3, report
    assert max(report["oproj"]) <= 2e-3, report
    assert max(report["kv_win_mismatch_x16"]) <= 0.03, report
    if "idx_q_mismatch" in report:
        assert max(report["idx_q_mismatch"]) == 0.0, report
        assert all(report["idx_q_g32_vs_fallback_bitwise"]), report


test_g32_golden_layer_end_to_end = _l_attn_end_to_end    # L-ATTN's test, collected here with the g32 fixture
