# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepseekV41Engram vs an FP32 transcription of the official Engram (lane L-ENGRAM; PORT_DESIGN §4.1, §4.5, §7.3).

The transcription follows ref:m.py:296-365 literally (embedding dequant -> BF16, wkv dequant, kv with FP64
accumulation rounded to FP32, the gate in FP32, output to BF16) and is fed the rows read straight from the tables.
Covered: torch and sm70 impls, prefill sub-chunks, padding rows, TP4 row-parallel wkv through the real forward with
a simulated all-reduce, the wkv loader (bias exactness, refusals), real weights of layers 1 and 14 once the shards
are verified, and L-REF golden tensors when they exist.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v41.common.contracts import EngramBatchLayout, EngramReqStep, EngramStepPlan

from .test_engram_synth import SyntheticEngram, build_synthetic_engram, random_e4m3_bytes, tokenizer_path

pytestmark = pytest.mark.sm70

GOLDEN_DIR = Path(os.environ.get("DS41_GOLDEN_DIR", "/mnt/nvme2/scratch/ds41/golden"))
ENGRAM_DIR = Path(os.environ.get("DS41_ENGRAM_DIR", "/home/mermiges/ds41-engram"))
E4M3 = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).to(torch.float64)


# ------------------------------------------------------------------------------ transcription of the reference
def ref_engram(stream: torch.Tensor, rows: np.ndarray, wkv_w: np.ndarray, wkv_s: np.ndarray, q: torch.Tensor,
               k: torch.Tensor, round_kv_bf16: bool = False) -> torch.Tensor:
    """stream [T,4,5120] bf16 (cpu); rows [T,24,264] uint8 (ALL sub-tables, order-major); wkv [25600,6144] u8 E4M3 +
    [800,192] u8 UE8M0; q/k [4,5120] bf16. Returns [T,4,5120] bf16 (ref:m.py:309-365, FP32 math, FP64 GEMM sum)."""
    T = rows.shape[0]
    vals = E4M3[torch.from_numpy(rows[..., :256]).long()].float()
    sc = torch.pow(2.0, torch.from_numpy(rows[..., 256:]).float() - 127)
    emb = (vals.view(T, 24, 8, 32) * sc.unsqueeze(-1)).view(T, 24 * 256).to(torch.bfloat16)  # ref casts to bf16
    W = E4M3[torch.from_numpy(wkv_w).long()]
    S = torch.pow(2.0, torch.from_numpy(wkv_s).double() - 127)
    W = W * S.repeat_interleave(32, 0).repeat_interleave(32, 1)
    kv = (emb.double() @ W.T).float()
    if round_kv_bf16:
        kv = kv.to(torch.bfloat16).float()
    key, value = kv.split([4 * 5120, 5120], dim=-1)
    key = key.unflatten(-1, (4, 5120))
    weight = q.float() * k.float()
    h = stream.float()
    rstd = torch.rsqrt(h.square().mean(-1) + 1e-20) * torch.rsqrt(key.square().mean(-1) + 1e-20)
    dot = (h * weight * key).sum(-1) * rstd * 5120 ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(torch.bfloat16)


def compare(out: torch.Tensor, ref: torch.Tensor, stream_in: torch.Tensor, tag: str) -> dict[str, float]:
    o, r = out.float().cpu(), ref.float().cpu()
    assert torch.isfinite(o).all(), f"{tag}: non-finite output"
    rel = ((o - r).norm() / r.norm()).item()
    maxabs = (o - r).abs().max().item()
    ulp = (out.cpu().view(torch.int16).int() - ref.view(torch.int16).int()).abs()
    d_ref = r - stream_in.float()
    d_rel = ((o - r).norm() / d_ref.norm()).item() if d_ref.norm() > 0 else 0.0
    m = dict(rel_rms=rel, max_abs=maxabs, max_ref=r.abs().max().item(), ulp_max=int(ulp.max()),
             frac_ulp_gt0=(ulp > 0).float().mean().item(), delta_rel_rms=d_rel)
    # PORT_DESIGN §4.5 module gate (rel-RMS <= 1e-3, max-abs <= 1e-2 x max) + a tighter look at the Engram delta
    assert rel <= 1e-3 and maxabs <= 1e-2 * m["max_ref"], f"{tag}: {m}"
    assert d_rel <= 1e-2, f"{tag}: Engram delta off: {m}"
    return m


# ------------------------------------------------------------------------------ synthetic setup
@pytest.fixture(scope="module")
def syn(tmp_path_factory: pytest.TempPathFactory) -> SyntheticEngram:
    return build_synthetic_engram(tmp_path_factory.mktemp("engram_mod"), seed=7)


@pytest.fixture(autouse=True)
def _knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_DS41_ENGRAM_REQUIRE_VERIFIED", "0")


def synthetic_wkv(seed: int) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    """FP8 wkv with the measured exponent span -18..-8 (23 % of real blocks are below 2^-14) and bf16 q/k."""
    rng = np.random.default_rng(seed)
    w = random_e4m3_bytes(rng, 25600 * 6144).reshape(25600, 6144)
    s = rng.integers(127 - 18, 127 - 8 + 1, size=(800, 192), dtype=np.uint8)
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(4, 5120, generator=g) * 0.5 + 1).to(torch.bfloat16)
    k = (torch.randn(4, 5120, generator=g) * 0.5 + 1).to(torch.bfloat16)
    return w, s, q, k


def make_module(syn: SyntheticEngram, layer: int, tp_rank: int, tp_size: int, wkv, max_tokens: int = 512):
    from vllm.models.deepseek_v41.common.engram import DeepseekV41Engram
    from vllm.models.deepseek_v41.common.engram_host import EngramHostService

    svc = EngramHostService(syn.hf_config, (layer,), tp_rank, tp_size, str(syn.row_dir), tokenizer_path(),
                            max_tokens, torch.device("cuda"), io_threads=2)
    with torch.device("cuda"):
        mod = DeepseekV41Engram(None, f"model.layers.{layer}.engram", layer, svc)
    w, s, q, k = wkv
    assert mod.load_checkpoint_tensor("wkv.weight", torch.from_numpy(w).view(torch.float8_e4m3fn)) is None
    assert mod.load_checkpoint_tensor("q_weight", q) is None
    assert mod.load_checkpoint_tensor("wkv.scale", torch.from_numpy(s).view(torch.float8_e8m0fnu)) == "wkv_r"
    assert mod.load_checkpoint_tensor("k_weight", k) == "qk"
    assert mod.weights_loaded()
    return mod, svc


def bind(svc, step: int, ids: np.ndarray, t_pad: int) -> None:
    svc.begin_step(EngramStepPlan(step, (EngramReqStep("r", 0, len(ids), ids, ids),), frozenset()))
    svc.bind_batch(EngramBatchLayout(step, ("r",), np.array([0, len(ids)], np.int32), len(ids), t_pad, None))


def table_rows(syn: SyntheticEngram, svc, layer: int, ids: np.ndarray) -> np.ndarray:
    """All 24 sub-table rows [T, 24, 264] for the token ids, straight from the tables."""
    from vllm.models.deepseek_v41.common.engram import EngramHasher

    h = EngramHasher(svc.layout, svc.hasher.token_map, (layer,), tuple(range(24)))
    rid = h.hash_positions(h.compress(ids), np.arange(len(ids)))[:, 0]
    return np.concatenate([syn.weights[layer][rid], syn.scales[layer][rid]], axis=-1)


def random_stream(T: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    mag = torch.exp(torch.randn(T, 1, 1, generator=g))
    return (torch.randn(T, 4, 5120, generator=g) * mag).to(torch.bfloat16)


# ------------------------------------------------------------------------------ tests
@pytest.mark.parametrize("impl", ["torch", "sm70"])
def test_module_vs_reference_tp1(syn: SyntheticEngram, monkeypatch: pytest.MonkeyPatch, impl: str) -> None:
    monkeypatch.setenv("VLLM_DS41_ENGRAM_IMPL", impl)
    monkeypatch.setenv("VLLM_DS41_ENGRAM_PREFILL_CHUNK", "64")    # 200 tokens -> 4 sub-chunks
    wkv = synthetic_wkv(11)
    for layer in (1, 14):
        mod, svc = make_module(syn, layer, 0, 1, wkv)
        try:
            ids = np.random.default_rng(layer).integers(3, 129000, size=200).astype(np.int32)
            bind(svc, 0, ids, 208)
            stream = random_stream(208, layer)
            stream[200:] = 0
            out = mod(stream.cuda(), torch.arange(208, device="cuda")).cpu()
            ref = ref_engram(stream[:200], table_rows(syn, svc, layer, ids), *wkv)
            m = compare(out[:200], ref, stream[:200], f"{impl} L{layer}")
            assert torch.equal(out[200:], stream[200:]), "padding rows must pass through unchanged"
            print(f"{impl} layer {layer}: {m}")
        finally:
            svc.shutdown()


def test_module_tp4_row_parallel_matches_tp1(syn: SyntheticEngram, monkeypatch: pytest.MonkeyPatch) -> None:
    """Four ranks (6 sub-tables each) through the real forward; the all-reduce is simulated in two passes."""
    import vllm.distributed as dist

    monkeypatch.setenv("VLLM_DS41_ENGRAM_IMPL", "sm70")
    monkeypatch.setenv("VLLM_DS41_ENGRAM_PREFILL_CHUNK", "48")
    wkv = synthetic_wkv(12)
    ids = np.random.default_rng(5).integers(3, 129000, size=100).astype(np.int32)
    stream = random_stream(100, 9)
    ranks = [make_module(syn, 14, r, 4, wkv) for r in range(4)]
    try:
        for _, svc in ranks:
            bind(svc, 0, ids, 100)
        partials: list[list[torch.Tensor]] = [[] for _ in range(4)]
        for r, (mod, _) in enumerate(ranks):
            monkeypatch.setattr(dist, "tensor_model_parallel_all_reduce", lambda x, r=r: partials[r].append(x.clone()) or x)
            mod(stream.cuda(), torch.arange(100, device="cuda"))
        sums = [sum(partials[r][c] for r in range(4)) for c in range(len(partials[0]))]
        outs = []
        for mod, _ in ranks:
            it = iter(sums)
            monkeypatch.setattr(dist, "tensor_model_parallel_all_reduce", lambda x, it=it: next(it))
            outs.append(mod(stream.cuda(), torch.arange(100, device="cuda")).cpu())
        for o in outs[1:]:
            assert torch.equal(o, outs[0]), "TP ranks disagree after the all-reduce"
        ref = ref_engram(stream, table_rows(syn, ranks[0][1], 14, ids), *wkv)
        print("tp4:", compare(outs[0], ref, stream, "tp4"))
        assert sorted(s for _, svc in ranks for s in svc.subtables) == list(range(24))
    finally:
        for _, svc in ranks:
            svc.shutdown()


def test_wkv_loader_exactness_and_refusals(syn: SyntheticEngram) -> None:
    from vllm.models.deepseek_v41.common.engram import DeepseekV41Engram
    from vllm.models.deepseek_v41.common.engram_host import EngramHostService

    svc = EngramHostService(syn.hf_config, (1,), 2, 4, str(syn.row_dir), tokenizer_path(), 16,
                            torch.device("cuda"), io_threads=1)
    try:
        w, s, q, k = synthetic_wkv(13)
        with torch.device("cuda"):
            mod = DeepseekV41Engram(None, "model.layers.1.engram", 1, svc)
        with pytest.raises(ValueError, match="skip_checkpoint_weight"):
            mod.load_checkpoint_tensor("embed.weight", torch.zeros(1))
        with pytest.raises(KeyError):
            mod.load_checkpoint_tensor("bogus", torch.zeros(1))
        mod.load_checkpoint_tensor("wkv.weight", torch.from_numpy(w))
        mod.load_checkpoint_tensor("wkv.scale", torch.from_numpy(s))
        with pytest.raises(RuntimeError, match="twice"):
            mod.load_checkpoint_tensor("wkv.weight", torch.from_numpy(w))
        # wkv_r == exact dequant of this rank's columns x 2^10 (sub-tables 2, 6, 10, 14, 18, 22)
        cols = np.concatenate([np.arange(sub * 256, sub * 256 + 256) for sub in svc.subtables])
        Wd = E4M3[torch.from_numpy(w[:, cols]).long()] * torch.pow(
            2.0, torch.from_numpy(s[:, cols // 32]).double() - 127).repeat_interleave(32, 0)
        assert torch.equal(mod.wkv_r.double().cpu(), Wd * 1024)
        assert (mod.wkv_r.abs() < 2.0**-14).any(), "test data must exercise FP16 subnormals"
        # an exponent that the 2^10 bias cannot bring into FP16's exact range is refused, loudly
        with torch.device("cuda"):
            bad = DeepseekV41Engram(None, "model.layers.1.engram", 1, svc)
        s_bad = s.copy()
        s_bad[3, svc.subtables[0] * 8] = 127 - 30
        bad.load_checkpoint_tensor("wkv.weight", torch.from_numpy(w))
        with pytest.raises(ValueError, match="not exact in FP16"):
            bad.load_checkpoint_tensor("wkv.scale", torch.from_numpy(s_bad))
    finally:
        svc.shutdown()


def test_wait_refuses_cuda_graph_capture(syn: SyntheticEngram, monkeypatch: pytest.MonkeyPatch) -> None:
    """A FULL-mode capture of the Engram wait would replay zero rows forever: it must fail loudly instead."""
    monkeypatch.setenv("VLLM_DS41_ENGRAM_IMPL", "sm70")
    mod, svc = make_module(syn, 1, 0, 1, synthetic_wkv(14), max_tokens=8)
    try:
        svc.bind_batch(EngramBatchLayout(-1, (), np.zeros(1, np.int32), 0, 4, None))
        stream = torch.zeros(4, 4, 5120, dtype=torch.bfloat16, device="cuda")
        mod(stream, torch.arange(4, device="cuda"))          # eager dummy run is fine
        g = torch.cuda.CUDAGraph()
        with pytest.raises(RuntimeError, match="captured into a CUDA graph"):
            with torch.cuda.graph(g):
                mod(stream, torch.arange(4, device="cuda"))
    finally:
        svc.shutdown()


# ------------------------------------------------------------------------------ real weights (shards 47/48)
def _verified_engram_shard(name: str) -> Path:
    p = ENGRAM_DIR / name
    if not (p.is_file() and Path(str(p) + ".sha256-ok").is_file()):
        pytest.skip(f"{p} not downloaded + sha256-verified yet")
    import subprocess

    out = subprocess.run(["bash", "/mnt/nvme2/models/_dl-logs/verify_shard.sh", name], capture_output=True,
                         text=True, timeout=900)
    if out.returncode != 0 or not out.stdout.strip().startswith("OK"):
        pytest.fail(f"verify_shard.sh {name}: {out.stdout} {out.stderr}")
    return p


@pytest.mark.weights
@pytest.mark.parametrize("layer,shard", [(1, "model-00047-of-00048.safetensors"),
                                         (14, "model-00048-of-00048.safetensors")])
def test_module_real_weights_vs_reference(monkeypatch: pytest.MonkeyPatch, layer: int, shard: str) -> None:
    """Real layers 1/14: official tables, wkv, q/k; real token ids (official tokenizer, real text); TP1 and TP4."""
    import json
    import types

    from safetensors import safe_open
    from transformers import PreTrainedTokenizerFast

    from vllm.models.deepseek_v41.common.engram import DeepseekV41Engram
    from vllm.models.deepseek_v41.common.engram_host import EngramHostService

    path = _verified_engram_shard(shard)
    ref_dir = Path(tokenizer_path()).parent
    hf = types.SimpleNamespace(**json.load(open(ref_dir / "config.json")))
    tok = PreTrainedTokenizerFast(tokenizer_file=str(ref_dir / "tokenizer.json"))
    text = open("/mnt/hdd/v100-research/docs/models/deepseek-v4.1-flash/ENGRAM.md", encoding="utf-8").read()
    ids = np.array([0] + tok.encode(text, add_special_tokens=False)[:383], np.int32)
    with safe_open(str(path), framework="pt") as f:
        pre = f"layers.{layer}.engram."
        w = f.get_tensor(pre + "wkv.weight")
        s = f.get_tensor(pre + "wkv.scale")
        q = f.get_tensor(pre + "q_weight")
        k = f.get_tensor(pre + "k_weight")
        table_w = f.get_slice(pre + "embed.weight")
        table_s = f.get_slice(pre + "embed.scale")
        svc0 = EngramHostService(hf, (layer,), 0, 1, str(ENGRAM_DIR), str(ref_dir), 384, torch.device("cuda"))
        try:
            from vllm.models.deepseek_v41.common.engram import EngramHasher

            hasher = EngramHasher(svc0.layout, svc0.hasher.token_map, (layer,), tuple(range(24)))
            rid = hasher.hash_positions(hasher.compress(ids), np.arange(len(ids)))[:, 0]
            rows = np.zeros((len(ids), 24, 264), np.uint8)
            for t in range(len(ids)):
                for j in range(24):
                    r = int(rid[t, j])
                    rows[t, j, :256] = table_w[r:r + 1].view(torch.uint8).numpy()[0]
                    rows[t, j, 256:] = table_s[r:r + 1].view(torch.uint8).numpy()[0]
        finally:
            svc0.shutdown()
    w8, s8 = w.view(torch.uint8).numpy(), s.view(torch.uint8).numpy()
    g = torch.Generator().manual_seed(layer)
    stream = (torch.randn(len(ids), 4, 5120, generator=g) * 0.05).to(torch.bfloat16)   # bf16 stream, real-ish scale
    ref = ref_engram(stream, rows, w8, s8, q, k)
    ref_bf16kv = ref_engram(stream, rows, w8, s8, q, k, round_kv_bf16=True)
    print(f"L{layer} reference with bf16-rounded kv vs FP32 kv: "
          f"{compare(ref_bf16kv, ref, stream, 'ref-bf16kv')}")
    for impl in ("torch", "sm70"):
        monkeypatch.setenv("VLLM_DS41_ENGRAM_IMPL", impl)
        svc = EngramHostService(hf, (layer,), 0, 1, str(ENGRAM_DIR), str(ref_dir), 384, torch.device("cuda"))
        try:
            with torch.device("cuda"):
                mod = DeepseekV41Engram(None, f"model.layers.{layer}.engram", layer, svc)
            for name, t in (("wkv.weight", w), ("wkv.scale", s), ("q_weight", q), ("k_weight", k)):
                mod.load_checkpoint_tensor(name, t)
            bind(svc, 0, ids, len(ids))
            got_rows = svc.wait_rows(layer).cpu().numpy()
            np.testing.assert_array_equal(got_rows, rows[:, list(svc.subtables)])
            out = mod(stream.cuda(), torch.arange(len(ids), device="cuda")).cpu()
            print(f"L{layer} {impl}: {compare(out, ref, stream, f'real L{layer} {impl}')}")
        finally:
            svc.shutdown()


# ------------------------------------------------------------------------------ golden (L-REF) when present
def _golden_cases() -> list[Path]:
    """Final (non-provisional) L-REF cases with Engram applied, v100-semantic mode (the primary comparator)."""
    import json

    out = []
    if GOLDEN_DIR.is_dir():
        for m in sorted(GOLDEN_DIR.glob("*/manifest.json")):
            man = json.load(open(m))
            if man.get("engram_applied") and not man.get("provisional") and man.get("mode") == "v100-semantic":
                out.append(m.parent)
    return out


@pytest.mark.weights
@pytest.mark.parametrize("case", _golden_cases() or [None], ids=lambda c: c.name if c else "none")
def test_engram_vs_golden(case: Path | None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Per captured Engram layer: hash ids and dequantised rows bitwise; key/value (rel-RMS <= 1e-5) and gate
    (|diff| <= 1e-5) from the module's own decode + FP16 GEMM path; module output stream vs the golden output.

    The v100-semantic golden keeps the stream in FP32 while the port stores it in BF16 (PORT_DESIGN A4): BF16 rounding
    alone is ~1.6e-3 rel-RMS, so the stream is compared against the golden rounded to BF16 (rel-RMS <= 1e-3) and the
    raw FP32 metric is reported."""
    if case is None:
        pytest.skip(f"no final L-REF golden case with engram_applied under {GOLDEN_DIR} yet")
    import json
    import types

    from safetensors import safe_open
    from safetensors.torch import load_file

    from vllm.models.deepseek_v41.common.engram import (
        DeepseekV41Engram, EngramHasher, EngramLayout, decode_rows_torch, load_compressed_token_map)
    from vllm.models.deepseek_v41.common.engram_host import EngramHostService

    monkeypatch.setenv("VLLM_DS41_ENGRAM_REQUIRE_VERIFIED", "1")
    man = json.load(open(case / "manifest.json"))
    ref_dir = Path(tokenizer_path()).parent
    hf = types.SimpleNamespace(**json.load(open(ref_dir / "config.json")))
    layout = EngramLayout.from_hf_config(hf)
    tmap = load_compressed_token_map(str(ref_dir), layout.compressed_vocab_size)
    hist = np.asarray(list(man.get("history_token_ids") or []) + list(man["token_ids"]), np.int64)
    start, n = int(man["start_pos"]), int(man["num_tokens"])
    assert hist.shape[0] == start + n, (hist.shape, start, n)
    pos = np.arange(start, start + n)
    found = 0
    for layer, shard in ((1, "model-00047-of-00048.safetensors"), (14, "model-00048-of-00048.safetensors")):
        f = case / f"L{layer:02d}.safetensors"
        if not f.is_file():
            continue
        g = load_file(str(f))
        if "engram.hash" not in g:
            continue
        found += 1
        h = EngramHasher(layout, tmap, (layer,), tuple(range(24)))
        np.testing.assert_array_equal(h.hash_positions(h.compress(hist), pos)[:, 0], g["engram.hash"].numpy())
        path = _verified_engram_shard(shard)
        svc = EngramHostService(hf, (layer,), 0, 1, str(ENGRAM_DIR), str(ref_dir), max(n, 8), torch.device("cuda"))
        try:
            with torch.device("cuda"):
                mod = DeepseekV41Engram(None, f"model.layers.{layer}.engram", layer, svc)
            with safe_open(str(path), framework="pt") as fh:
                for name in ("wkv.weight", "wkv.scale", "q_weight", "k_weight"):
                    mod.load_checkpoint_tensor(name, fh.get_tensor(f"layers.{layer}.engram.{name}"))
            ids32 = hist.astype(np.int32)
            svc.begin_step(EngramStepPlan(0, (EngramReqStep("g", start, n, ids32[start:], ids32),), frozenset()))
            svc.bind_batch(EngramBatchLayout(0, ("g",), np.array([0, n], np.int32), n, n, None))
            rows = svc.wait_rows(layer)
            dec = decode_rows_torch(rows, svc.row_bias(layer)).float().cpu().view(n, 24, 256)
            assert torch.equal(dec, g["engram.rows"].float()), "dequantised rows differ from the golden"
            bias = svc.row_bias(layer)
            kv = torch.mm(decode_rows_torch(rows, bias), mod.wkv_r.t(), out_dtype=torch.float32) * 2.0 ** -(10 + bias)
            key, value = kv[:, :20480].view(n, 4, 5120).cpu(), kv[:, 20480:].cpu()
            for name, ours in (("engram.key", key), ("engram.value", value)):
                ref = g[name].float()
                assert ((ours - ref).norm() / ref.norm()).item() <= 1e-5, name
            h32 = g["stream_in"].float()
            qk = mod.qk.cpu()
            rstd = torch.rsqrt(h32.square().mean(-1) + 1e-20) * torch.rsqrt(key.square().mean(-1) + 1e-20)
            dot = (h32 * qk * key).sum(-1) * rstd * 5120 ** -0.5
            gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
            assert (gate - g["engram.gate"].float()).abs().max().item() <= 1e-5
            stream = h32.to(torch.bfloat16).cuda()
            out = mod(stream, torch.from_numpy(pos).cuda()).float().cpu()
            gold = g["engram.out"].float()
            gold16 = gold.to(torch.bfloat16).float()
            rel16 = ((out - gold16).norm() / gold16.norm()).item()
            rel32 = ((out - gold).norm() / gold.norm()).item()
            print(f"{case.name} L{layer}: stream rel-RMS vs bf16(golden) {rel16:.2e}, vs FP32 golden {rel32:.2e}")
            assert rel16 <= 1e-3 and (out - gold).abs().max().item() <= 1e-2 * gold.abs().max().item()
        finally:
            svc.shutdown()
    if not found:
        pytest.skip(f"{case}: no engram.hash tensors in this golden case")
