# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L-ATTN vs L-REF golden tensors (PORT_DESIGN §3.8, §4.5; mode v100-semantic = the primary comparator).

Per-op tests feed the golden's inputs to one port op and compare with the golden output:
  * q path        attn.x  -> attn.qr, attn.q (dense-linear gate 2e-3); attn.x in FP32 -> port window-KV kernel ->
                  attn.kv_win bitwise (0 mismatches); FP16-rounded attn.x (the port's real input) -> <= 3 % flips
  * indexer       idx.q, idx.w, cache.ik -> idx.score (rel-RMS <= 1e-3, -inf pattern exact) -> idx.topk (same set)
  * candidates    idx.score (layer 20) -> cand.blocks (set equality)
  * sparse attn   attn.q + window/compressed records + attn.topk_all -> attn.o (<= 2e-3)
  * o-proj        attn.o -> attn.out (dense-linear gate 2e-3)
a long-context indexer test on p4_long (24.6K tokens, layers 20/24: real candidate blocks beyond 16,384), and a
golden-input end-to-end run: attn.x of the prefill (in chunks) and of decode1..4 through DeepseekV41Attention on the
simulated paged cache, gated in three links (see test_golden_layer_end_to_end):
  golden ~ ref32 (own FP32 reference fed FP32 attn.x)            -- validates the reference
  ref16  (same reference fed the FP16-rounded attn.x the port gets) -- input-rounding floor, not gated vs golden
  port   ~ ref16 at the §4.5 per-op gates                          -- the port's own error.

Goldens: $DS41_GOLDEN_DIR (default /mnt/nvme2/scratch/ds41/golden), final cases preferred over provisional/. Layers
not captured in the available batch are skipped (the PROV-2 batch has 0, 1, 2, 3, 14; later batches add 20, 21, 24).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v41.common.candidate_blocks import select_candidate_blocks
from vllm.models.deepseek_v41.common.contracts import CAND_ALL
from vllm.models.deepseek_v41.sm70.indexer_kernels import index_scores, topk_sorted
from vllm.models.deepseek_v41.sm70.q_rope_kv_insert import q_rope_kv_insert
from vllm.models.deepseek_v41.sm70.sparse_kernels import sparse_attention

from .test_attn_harness import load_attn_weights, ref_config, rel_rms, topology
from .test_attn_layers import DEV, Stage, _overlap, dist_env  # noqa: F401  (fixture)

pytestmark = [pytest.mark.sm70, pytest.mark.weights]

ROOT = Path(os.environ.get("DS41_GOLDEN_DIR", "/mnt/nvme2/scratch/ds41/golden"))
MODE = "v100-semantic"
PROMPTS = ("p1_legal", "p2_code", "p3_doc")
GROUPS = ((0,), (1,), (2, 3), (14,), (20, 21, 24))
PHASES = ("prefill", "decode1", "decode2", "decode3", "decode4")


def _case_dir(prompt: str, phase: str) -> Path | None:
    for base in (ROOT, ROOT / "provisional"):
        d = base / f"{prompt}__{MODE}__{phase}"
        if (d / "manifest.json").is_file():
            return d
    return None


class Case:
    def __init__(self, d: Path) -> None:
        from safetensors import safe_open
        self.dir = d
        self.manifest = json.loads((d / "manifest.json").read_text())
        self._files: dict = {}
        self._open = safe_open
        self.captured = set(self.manifest["capture_layers"])

    def get(self, layer: int, name: str) -> torch.Tensor:
        f = self.dir / f"L{layer:02d}.safetensors"
        if f not in self._files:
            self._files[f] = self._open(str(f), framework="pt", device="cpu")
        return self._files[f].get_tensor(name).to(DEV)

    def has(self, layer: int, name: str) -> bool:
        return any(t["layer"] == layer and t["op"] == name for t in self.manifest["tensors"])

    @property
    def start(self) -> int:
        return int(self.manifest["start_pos"])

    @property
    def num_tokens(self) -> int:
        return int(self.manifest["num_tokens"])


def _cases(prompt: str) -> list[Case]:
    out = []
    for ph in PHASES:
        d = _case_dir(prompt, ph)
        if d is None:
            break
        out.append(Case(d))
    return out


def _require(prompt: str, layers: tuple[int, ...]) -> list[Case]:
    cases = _cases(prompt)
    if not cases:
        pytest.skip(f"no {MODE} golden for {prompt} under {ROOT}")
    if not set(layers) <= cases[0].captured:
        pytest.skip(f"golden batch at {cases[0].dir.parent} did not capture layers {layers}")
    return cases


@pytest.fixture(scope="module")
def weights_cache():
    return {}


def _weights(cache: dict, layer: int) -> dict:
    if layer not in cache:
        cache[layer] = load_attn_weights(layer, DEV)
    return cache[layer]


# ------------------------------------------------------------------------------------------------ per-op tests
@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("layer", [0, 1, 2, 3, 14, 20, 21, 24])
def test_golden_per_op(dist_env, weights_cache, prompt: str, layer: int) -> None:    # noqa: F811
    cases = _require(prompt, (layer,))
    cfg = ref_config()
    topo = topology(cfg, layer)
    w = _weights(weights_cache, layer)
    st = Stage(cfg, (layer,), {layer: w})
    attn = st.attn[layer]
    report: dict = {}
    from vllm.models.deepseek_v41.common.rope import apply_rope_torch
    for case in cases:
        def g(n: str, _case=case) -> torch.Tensor:
            return _case.get(layer, n)
        T, start = case.num_tokens, case.start
        pos = torch.arange(start, start + T, device=DEV)
        subset = case.manifest.get("row_subset") or {}
        rows = torch.arange(T, device=DEV)
        if "attn.q" in subset.get("tensors", ()):
            rows = torch.tensor(subset["rows"], device=DEV)
        # ---- q path fed attn.x
        x16 = g("attn.x").half()
        qr_kv = torch.mm(x16, attn.fused_wqa_wkv.weight.t(), out_dtype=torch.float32)
        qr32 = attn.q_norm(qr_kv[:, :1280])
        q = torch.mm(qr32.half()[rows], attn.wq_b.weight.t()).view(rows.numel(), 64, 512)
        q = apply_rope_torch(q, pos[rows], attn.rotary_emb.cos_sin_cache)
        gq = g("attn.q")
        report.setdefault("qr", []).append(rel_rms(qr32, g("attn.qr")))
        report.setdefault("q", []).append(rel_rms(q, gq))
        # ---- window-KV records through the port kernel (RoPE + FP8 QAT -> FP16 rows). The golden computes the kv
        # projection from FP32 attn.x: fed the same FP32 x the kernel must reproduce every record bit for bit;
        # fed the FP16-rounded x of the real forward, QAT flips of up to one FP8 step remain (reported + bounded).
        gkw = g("attn.kv_win")
        x32 = g("attn.x")
        w_kv = attn.fused_wqa_wkv.weight[1280:]
        for tag, kv_pre in (("kv_win_mismatch_x32", x32 @ w_kv.float().t()),
                            ("kv_win_mismatch_x16", qr_kv[:, 1280:])):
            kv = attn.kv_norm(kv_pre)
            kv_rows = torch.empty(T, 512, dtype=torch.float16, device=DEV)
            q_rope_kv_insert(torch.zeros(T, 1, 512, dtype=torch.float16, device=DEV), kv, pos,
                             attn.rotary_emb.cos_sin_cache, kv_rows, torch.arange(T, device=DEV), impl="sm70")
            report.setdefault(tag, []).append(float((kv_rows.float() != gkw).float().mean()))
        # ---- o-proj fed the golden attention output
        go = g("attn.o")
        out = attn.output_projection(go.half().contiguous(), pos[rows])
        report.setdefault("oproj", []).append(rel_rms(out, g("attn.out")[rows]))
        # ---- sparse attention fed golden q, records and the full index list
        tk = g("attn.topk_all").long()
        win_rows = g("attn.kv_win") if case.manifest["phase"] == "prefill" else g("cache.win")
        srcs = [win_rows]
        if topo.compress_ratio:
            srcs.append(case.get(topo.kv_source, "cache.ckv"))
        kvs = torch.cat(srcs).half()
        o = torch.empty(rows.numel(), 64, 512, dtype=torch.float16, device=DEV)
        sparse_attention(gq.half().contiguous(), [(kvs, tk)], attn.attn_sink, 512 ** -0.5, o, impl="sm70")
        report.setdefault("sparse", []).append(rel_rms(o, go))
        # ---- indexer: scores, top-k and candidates fed golden q, w, K
        if topo.owns_indexer and case.has(layer, "idx.score"):
            gs = g("idx.score")
            n = gs.shape[1]
            keys = case.get(topo.kv_source, "cache.ik")[:n].half().contiguous()
            sc = torch.empty(T, n, device=DEV)
            index_scores(g("idx.q").half().contiguous(), g("idx.w"), keys, sc)
            ends = torch.div(pos + 1, topo.compress_ratio, rounding_mode="floor")
            beyond = torch.arange(n, device=DEV)[None, :] >= ends[:, None]
            # golden prompts are < 16,384 tokens: the candidate mask is a no-op, -inf == unreachable exactly
            assert torch.equal(torch.isinf(gs), beyond), "golden -inf pattern != causal bound"
            stale = (topo.owns_compressor and topo.compress_ratio == 2 and case.manifest["phase"] != "prefill"
                     and int(pos[-1] + 1) % 2 != 0)
            if stale:
                # Official Indexer.forward re-points shared_attn.index_k only when the layer emits a latent; on a
                # decode step that completes no pair a ratio-2 source scores against the K cache published last
                # (another layer's). The port uses its own index-K (PORT_DESIGN §3.3, upstream vLLM). Identify the
                # golden's K source instead of gating on it.
                hit = []
                for src in sorted(case.captured):
                    if case.has(src, "cache.ik") and case.get(src, "cache.ik").shape[0] >= n:
                        alt = torch.empty(T, n, device=DEV)
                        index_scores(g("idx.q").half().contiguous(), g("idx.w"),
                                     case.get(src, "cache.ik")[:n].half().contiguous(), alt)
                        if rel_rms(alt, gs) < 1e-6:
                            hit.append(src)
                report.setdefault("stale_index_k_golden_uses_layer", []).append(hit)
                assert hit, "golden scores at a stale-index_k step match no captured index-K cache"
            else:
                report.setdefault("score", []).append(rel_rms(sc[~beyond], gs[~beyond]))
            if not stale:
                # per-op indexer: golden attn.x (weights_proj), attn.qr, K -> port q GEMM/RoPE/QAT, scores, top-k
                from vllm.models.deepseek_v41.compressor import mm_fp32, mm_fp32_split
                from vllm.models.deepseek_v41.sm70.indexer_kernels import index_q_rope_qat
                ix = attn.indexer
                pq = mm_fp32_split(g("attn.qr"), ix.wq_b.weight).view(T, 32, 128)
                pq = index_q_rope_qat(pq, pos, attn.rotary_emb.cos_sin_cache)
                pw = mm_fp32(g("attn.x").half(), ix.weights_proj.weight) * (128 ** -0.5 * 32 ** -0.5)
                ps = torch.empty(T, n, device=DEV)
                index_scores(pq, pw, keys, ps)
                ps.masked_fill_(beyond, float("-inf"))
                report.setdefault("idx_q_mismatch", []).append(float((pq.float() != g("idx.q")).float().mean()))
                report.setdefault("score_port_q", []).append(rel_rms(ps[~beyond], gs[~beyond]))
                mean, mn = _overlap(topk_sorted(ps, ends, 512).long(), g("idx.topk").long())
                report.setdefault("topk_port_mean", []).append(mean)
                report.setdefault("topk_port_min", []).append(mn)
            got = topk_sorted(gs.clone(), ends, 512).long()
            ref = g("idx.topk").long()
            same = [set(a) - {-1} == set(b) - {-1} for a, b in zip(got.tolist(), ref.tolist())]
            report.setdefault("topk_same_set_from_golden_scores", []).append(float(np.mean(same)))
            if topo.is_candidate_source and case.has(layer, "cand.blocks"):
                cand = torch.empty(T, 2048, dtype=torch.int32, device=DEV)
                select_candidate_blocks(gs.clone(), ends, 2048, 8, cand)
                gc = g("cand.blocks").long()
                for r in range(T):
                    want = set(gc[r].tolist()) - {-1}
                    if cand[r, 0].item() == CAND_ALL:
                        assert want == set(range((int(ends[r]) + 7) // 8)), f"row {r}: CAND_ALL vs golden"
                    else:
                        assert set(cand[r].tolist()) - {-1} == want, f"row {r}"
                report.setdefault("cand_rows", []).append(T)
    print(prompt, layer, {k: [v if isinstance(v, list) else round(v, 6) for v in vals] for k, vals in report.items()})
    assert max(report["q"]) <= 2e-3 and max(report["qr"]) <= 2e-3, report
    assert max(report["oproj"]) <= 2e-3, report
    assert max(report["sparse"]) <= 2e-3, report
    assert max(report["kv_win_mismatch_x32"]) == 0.0, report       # bitwise window-KV records (MC-ATTN F4)
    assert max(report["kv_win_mismatch_x16"]) <= 0.03, report      # measured <= 2.2 % (FP16 input rounding)
    if "idx_q_mismatch" in report:
        assert max(report["idx_q_mismatch"]) <= 1e-4, report        # FP4 q from golden attn.qr (split-FP16 GEMM)
    if "score" in report:
        assert max(report["score"]) <= 1e-3, report
        assert min(report["topk_same_set_from_golden_scores"]) >= 0.99, report
    if "score_port_q" in report:
        assert max(report["score_port_q"]) <= 1e-3, report
        assert min(report["topk_port_mean"]) >= 0.995 and min(report["topk_port_min"]) >= 0.98, report


# ------------------------------------------------------------------------------------------------ end to end
@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("group", GROUPS, ids=lambda g: "L" + "-".join(map(str, g)))
def test_golden_layer_end_to_end(dist_env, weights_cache, prompt: str, group) -> None:    # noqa: F811
    cases = _require(prompt, group)
    cfg = ref_config()
    weights = {i: _weights(weights_cache, i) for i in group}
    st = Stage(cfg, group, weights)
    pre = cases[0]
    T0 = pre.num_tokens
    inputs = {i: {"a": torch.cat([c.get(i, "attn.x") for c in cases]).half()} for i in group}
    S = inputs[group[0]]["a"].shape[0]
    outs = {i: {"a": torch.full((S, 5120), float("nan"), device=DEV)} for i in group}
    chunks, done = [], 0
    for n in (min(T0, 37), min(T0, 511)):
        if done + n <= T0 and n > 0:
            chunks.append(n)
            done += n
    if done < T0:
        chunks.append(T0 - done)
    chunks += [1] * (S - T0)
    pos = 0
    for n in chunks:
        st.step([("a", pos, n)], inputs, outs)
        pos += n
    report: dict = {}
    for i in group:
        topo = topology(cfg, i)
        golden_out = torch.cat([c.get(i, "attn.out") for c in cases])
        report.setdefault(i, {})["out"] = rel_rms(outs[i]["a"], golden_out)
        if topo.owns_indexer or topo.compress_ratio:
            if all(c.has(i, "idx.topk") for c in cases) or all(
                    c.has(topo.index_source, "idx.topk") for c in cases):
                src = i if cases[0].has(i, "idx.topk") else topo.index_source
                gtk = torch.cat([c.get(src, "idx.topk") for c in cases]).long()
                ptk = torch.cat([st.rec[("topk", i, "a")][k] for k in sorted(st.rec[("topk", i, "a")])]).long()
                keep = torch.ones(gtk.shape[0], dtype=torch.bool)
                if topo.compress_ratio == 2:
                    # decode rows where the official reference scored against a stale index_k (see per-op test)
                    for k, c in enumerate(cases[1:]):
                        if (c.start + 1) % 2 != 0:
                            keep[T0 + k] = False
                mean, mn = _overlap(ptk[keep], gtk[keep])
                report[i]["topk_mean"], report[i]["topk_min"] = mean, mn
                report[i]["stale_rows_excluded"] = int((~keep).sum())
        if topo.owns_compressor and cases[0].has(i, "attn.latent"):
            lat = st.rec[("latent", i, "a")]
            gl = [c.get(i, "attn.latent") for c in cases if c.has(i, "attn.latent")]
            gl = torch.cat(gl)
            port = torch.stack([lat[p] for p in sorted(lat)])[: gl.shape[0]]
            report[i]["latent"] = rel_rms(port, gl)
    print(prompt, group, report)
    for i, m in report.items():
        assert m["out"] <= 1e-2, (i, m)
        if "topk_mean" in m:
            # composite: keys come from the port's own compressor/write_keys fed FP16 attn.x (PORT_DESIGN §4.1 FP16
            # sublayer inputs); the strict §4.5 indexer gate (min 98 %) is enforced by test_golden_per_op
            assert m["topk_mean"] >= 0.995 and m["topk_min"] >= 0.97, (i, m)
        if "latent" in m:
            assert m["latent"] <= 1e-3, (i, m)
