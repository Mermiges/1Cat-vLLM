# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P5-ATTN: graph-safe decode metadata (sm70/decode_metadata.py + builders) and the static-shape decode kernels
(sm70/decode_kernels.py).

* builders: a decode batch (one token per real request, runner padding to T_pad requests/tokens) yields metadata
  whose real-token fields equal the general path's, whose padding tokens read/write nothing, and whose tensors keep
  their addresses from step to step (what a captured graph needs);
* kernels vs the general-path ops they replace (FP32 summation order is the only difference);
* CUDA-graph capture + replay of the kernels is bitwise equal to eager execution for buckets 1/2/8/32/64.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v41.common.candidate_blocks import apply_candidate_mask, select_candidate_blocks
from vllm.models.deepseek_v41.common.contracts import CAND_ALL, CAND_BLOCK, CAND_TOPK_BLOCKS
from vllm.models.deepseek_v41.compressor import rmsnorm_fp32
from vllm.models.deepseek_v41.sm70 import decode_kernels as dk
from vllm.models.deepseek_v41.sm70.indexer_kernels import index_scores, topk_sorted
from vllm.models.deepseek_v41.sm70.sparse import DECODE_PATH_ENV, DS41SWAMetadataBuilder
from vllm.models.deepseek_v41.sm70.sparse_kernels import logical_to_rows, sparse_attention
from vllm.v1.attention.backend import AttentionCGSupport

from .test_attn_harness import ref_config, synthetic_attn_weights, topology
from .test_attn_layers import DEV, Stage, dist_env  # noqa: F401  (fixture)

pytestmark = pytest.mark.sm70


# ============================================================================ builders
def _prefill(st: Stage, reqs: dict[str, int]) -> None:
    """Chunked prefill of every request to its length (general path) so the caches hold real records."""
    S = max(reqs.values()) + 4
    inputs = {i: {r: torch.randn(S, 5120, device=DEV).half() * 0.5 for r in reqs} for i in st.layer_ids}

    class _Sink:
        def __setitem__(self, k, v) -> None:
            pass
    outs = {i: {r: _Sink() for r in reqs} for i in st.layer_ids}
    for r, n in reqs.items():
        p = 0
        while p < n:
            c = min(1024, n - p)
            st.step([(r, p, c)], inputs, outs)
            p += c


def test_decode_default_and_explicit_overrides(monkeypatch) -> None:
    from vllm.models.deepseek_v41.sm70.sparse import decode_path_enabled

    monkeypatch.delenv(DECODE_PATH_ENV, raising=False)
    monkeypatch.setenv("VLLM_DS41_ATTN_IMPL", "sm70")
    assert decode_path_enabled()
    monkeypatch.setenv(DECODE_PATH_ENV, "0")
    assert not decode_path_enabled()
    monkeypatch.setenv(DECODE_PATH_ENV, "1")
    assert decode_path_enabled()
    monkeypatch.setenv("VLLM_DS41_ATTN_IMPL", "torch")
    assert not decode_path_enabled()


def test_decode_metadata_matches_general(dist_env, monkeypatch) -> None:
    cfg = ref_config()
    ids = (0, 2, 20)
    st = Stage(cfg, ids, {i: synthetic_attn_weights(cfg, topology(cfg, i), DEV, seed=i) for i in ids})
    for i in ids:                                    # drop the latent recorder (it reads general metadata only)
        c = st.attn[i].compressor
        if c is not None:
            c.__dict__.pop("forward", None)
    monkeypatch.setenv(DECODE_PATH_ENV, "0")
    assert DS41SWAMetadataBuilder.get_cudagraph_support(st.vcfg, None) == AttentionCGSupport.NEVER
    _prefill(st, {"a": 300, "b": 1501})
    batch = [("a", 300, 1), ("b", 1501, 1)]
    gen, _ = st.sim.metadata(batch)
    assert not any(m.decode for m in gen.values())
    monkeypatch.setenv(DECODE_PATH_ENV, "1")
    assert DS41SWAMetadataBuilder.get_cudagraph_support(st.vcfg, None) == \
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    dec, _ = st.sim.metadata(batch, pad_tokens=4, pad_reqs=4)
    ptrs = {}
    for name, g in gen.items():
        d = dec[name]
        assert d.decode and d.num_actual_tokens == 4 and d.num_reqs == 4, name
        assert torch.equal(d.positions[:2], g.positions[:2]) and (d.positions[2:] == 0).all(), name
        w = g.block_table.shape[1]
        assert torch.equal(d.block_table[:2, :w], g.block_table[:2]), name
        assert torch.equal(d.slot_mapping[:2], g.slot_mapping[:2]), name
        assert (d.slot_mapping[2:] == -1).all(), name
        if hasattr(g, "window_slots"):
            assert torch.equal(d.window_slots[:2], g.window_slots[:2]) and (d.window_slots[2:] == -1).all(), name
        if hasattr(g, "num_visible"):
            assert torch.equal(d.num_visible[:2], g.num_visible[:2]) and (d.num_visible[2:] == 0).all(), name
            tok = g.latent_token_idx.tolist()
            want = torch.full((4,), -1, dtype=torch.int64, device=DEV)
            want[tok] = g.latent_slots
            assert torch.equal(d.latent_slots, want), name
            assert torch.equal(d.latent_pos[tok], g.latent_pos), name
        if hasattr(g, "prev_slot"):
            assert torch.equal(d.prev_slot[:2], g.prev_slot[:2]) and (d.prev_slot[2:] == -1).all(), name
        ptrs[name] = [t.data_ptr() for t in vars(d).values() if isinstance(t, torch.Tensor)]
    # next step: every tensor the layers read keeps its address (a captured graph replays against them)
    dec2, _ = st.sim.metadata([("a", 301, 1), ("b", 1502, 1)], pad_tokens=4, pad_reqs=4)
    for name, d in dec2.items():
        assert [t.data_ptr() for t in vars(d).values() if isinstance(t, torch.Tensor)] == ptrs[name], name
        assert int(d.positions[0]) == 301 and int(d.positions[1]) == 1502
    # a batch with a 2-token request or R != T is not a decode batch
    mixed, _ = st.sim.metadata([("a", 302, 1), ("b", 1503, 2)])
    assert not any(m.decode for m in mixed.values())


def test_padded_general_metadata(dist_env, monkeypatch) -> None:
    """Breakable PIECEWISE graphs pad mixed batches: the general path takes padding tokens (no request) as
    position 0 / slot -1 instead of refusing the batch."""
    monkeypatch.setenv(DECODE_PATH_ENV, "0")
    cfg = ref_config()
    st = Stage(cfg, (2,), {2: synthetic_attn_weights(cfg, topology(cfg, 2), DEV, seed=2)})
    md, _ = st.sim.metadata([("a", 0, 5), ("b", 0, 1)], pad_tokens=8)
    for name, m in md.items():
        assert m.num_actual_tokens == 8 and not m.decode
        assert (m.slot_mapping[6:] == -1).all() and (m.positions[6:] == 0).all(), name


# ============================================================================ kernels vs general ops
def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp(min=1e-30))


def test_norm_and_ratio2_kernels() -> None:
    torch.manual_seed(0)
    T = 5
    x = torch.randn(T, 1792, device=DEV) * 3
    qw, kw = torch.rand(1280, device=DEV) + 0.5, torch.rand(512, device=DEV) + 0.5
    q32, q16, kv = dk.qkv_norm(x, qw, kw, 1e-20, 1280)
    assert _rel(q32, rmsnorm_fp32(x[:, :1280], qw)) < 1e-6 and _rel(kv, rmsnorm_fp32(x[:, 1280:], kw)) < 1e-6
    assert torch.equal(q16, q32.half())
    # ratio-2 combine: state write + pair softmax + norm
    state = torch.randn(64, 1024, device=DEV)
    ks = torch.randn(T, 1024, device=DEV) * 2
    slots = torch.tensor([3, 9, -1, 12, 40], device=DEV)
    prev = torch.tensor([7, -1, 20, 5, -1], device=DEV)
    old = state.clone()
    nw = torch.rand(512, device=DEV) + 0.5
    lat = dk.ratio2_decode(ks, state, slots, prev, nw, 1e-20)
    keep = slots >= 0
    assert torch.equal(state[slots[keep]], ks[keep])
    done = prev >= 0
    pair = torch.stack([old[prev[done]], ks[done]], dim=1)
    kvp, scp = pair.split(512, dim=-1)
    ref_kv = (kvp * scp.softmax(dim=1)).sum(dim=1)
    assert _rel(lat[done], rmsnorm_fp32(ref_kv, nw)) < 1e-6
    assert torch.isfinite(lat).all()


def _paged_keys(n_rows_cap: int, storage: int, lens: list[int], seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    width = max(1, max((n + storage - 1) // storage for n in lens))
    nblk = max(n_rows_cap // storage, len(lens) * width + 1)
    keys = (torch.randn(nblk * storage, 128, generator=g) * 0.5).half().to(DEV)
    perm = torch.randperm(nblk, generator=g)
    bt = perm[: len(lens) * width].view(len(lens), width).to(torch.int32).to(DEV)
    return keys, bt


def test_index_scores_and_topk_vs_general() -> None:
    torch.manual_seed(1)
    storage, W = 128, 4096
    lens = [300, 4096, 2500, 0]
    T = len(lens)
    keys, bt = _paged_keys(64 * storage, storage, lens, seed=1)
    q = (torch.randn(T, 32, 128, device=DEV)).half()
    w = torch.randn(T, 32, device=DEV)
    nvis = torch.tensor(lens, device=DEV)
    tok2req = torch.arange(T, device=DEV, dtype=torch.int32)
    sc = dk.index_scores_decode(q, w, 0.0625, keys, bt, tok2req, storage, nvis, W)
    out = torch.empty(T, 512, dtype=torch.int32, device=DEV)
    dk.topk_decode(sc, nvis, 512, out)
    for t, n in enumerate(lens):
        assert torch.isinf(sc[t, n:]).all() and (sc[t, n:] < 0).all()
        if n == 0:
            assert (out[t] == -1).all()
            continue
        rows = logical_to_rows(torch.arange(n, device=DEV)[None], tok2req[t:t + 1], bt, storage)[0]
        ref = torch.empty(1, n, device=DEV)
        index_scores(q[t:t + 1], w[t:t + 1] * 0.0625, keys.index_select(0, rows), ref)
        assert _rel(sc[t, :n], ref[0]) < 1e-5
        ref_top = topk_sorted(sc[t:t + 1, :n].clone(), nvis[t:t + 1], 512)
        assert torch.equal(out[t:t + 1], ref_top)      # same scores -> identical picks (static width, -inf pad)
    # candidate-compacted scores (Reindex layers) == masked full-width scores, same top-512
    lens_c = [20000, 9000]
    keys, bt = _paged_keys(256 * storage, storage, lens_c, seed=2)
    Tc, Wf = len(lens_c), 20480
    q = torch.randn(Tc, 32, 128, device=DEV).half()
    w = torch.rand(Tc, 32, device=DEV)
    nvis = torch.tensor(lens_c, device=DEV)
    tok2req = torch.arange(Tc, device=DEV, dtype=torch.int32)
    full = dk.index_scores_decode(q, w, 1.0, keys, bt, tok2req, storage, nvis, Wf)
    cand = torch.empty(Tc, CAND_TOPK_BLOCKS, dtype=torch.int32, device=DEV)
    select_candidate_blocks(full, nvis, CAND_TOPK_BLOCKS, CAND_BLOCK, out=cand)
    assert int(cand[0, 0]) != CAND_ALL and int(cand[1, 0]) == CAND_ALL
    comp = dk.index_scores_decode(q, w, 1.0, keys, bt, tok2req, storage, nvis, CAND_TOPK_BLOCKS * CAND_BLOCK,
                                  cand=cand)
    masked = full.clone()
    apply_candidate_mask(masked, nvis, cand, CAND_BLOCK)
    got = torch.empty(Tc, 512, dtype=torch.int32, device=DEV)
    dk.topk_decode(comp, torch.where(cand[:, 0] == CAND_ALL, nvis, torch.full_like(nvis, comp.shape[1])), 512,
                   got, cand=cand)
    ref = topk_sorted(masked, nvis, 512)
    assert torch.equal(got, ref)


def _attn_inputs(T: int, H: int, seed: int, ctx: list[int] | None = None):
    g = torch.Generator(device="cpu").manual_seed(seed)
    storage, nrows = 128, 64 * 128
    swa = (torch.randn(nrows, 512, generator=g) * 0.3).half().to(DEV)
    ckv = (torch.randn(nrows, 512, generator=g) * 0.3).half().to(DEV)
    q = (torch.randn(T, H, 512, generator=g)).half().to(DEV)
    win = torch.randint(0, nrows, (T, 128), generator=g).to(DEV)
    win[:, :17] = -1
    topk = torch.randint(0, 2000, (T, 512), generator=g).to(torch.int32).to(DEV)
    topk[:, 400:] = -1
    if T > 1:
        win[-1] = -1                                   # a padding row: no valid entry -> exactly zero output
        topk[-1] = -1
    bt = torch.randint(0, 64, (T, 16), generator=g).to(torch.int32).to(DEV)
    sink = torch.randn(H, generator=g).to(DEV)
    return q, swa, ckv, win, topk, bt, sink, storage


@pytest.mark.parametrize("T", [1, 3])
def test_sparse_attention_decode_vs_general(T: int) -> None:
    H = 16
    q, swa, ckv, win, topk, bt, sink, storage = _attn_inputs(T, H, seed=T)
    tok2req = torch.arange(T, device=DEV, dtype=torch.int32)
    out = torch.empty(T, H, 512, dtype=torch.float16, device=DEV)
    dk.sparse_attention_decode(q, swa, win, ckv, topk, bt, tok2req, storage, sink, 512 ** -0.5, out)
    rows = logical_to_rows(topk, tok2req, bt, storage)
    ref = torch.empty_like(out)
    sparse_attention(q, [(swa, win), (ckv, rows)], sink, 512 ** -0.5, ref, impl="sm70")
    ref32 = torch.empty_like(out)
    sparse_attention(q, [(swa, win), (ckv, rows)], sink, 512 ** -0.5, ref32, impl="torch")
    assert _rel(out, ref) < 1e-3 and _rel(out, ref32) < 2e-3
    if T > 1:
        assert (out[-1] == 0).all()
    # SWA-only layer
    out2 = torch.empty_like(out)
    dk.sparse_attention_decode(q, swa, win, None, None, None, None, 0, sink, 512 ** -0.5, out2)
    ref2 = torch.empty_like(out)
    sparse_attention(q, [(swa, win)], sink, 512 ** -0.5, ref2, impl="sm70")
    assert _rel(out2, ref2) < 1e-3


@pytest.mark.parametrize("T", [1, 2, 8, 32, 64])
def test_decode_kernels_graph_replay_bitwise(T: int) -> None:
    """Capture attention + index scores + top-k once (capture-time inputs: everything invalid, as the runner's
    dummy capture batch), then replay on real data written into the same buffers: bitwise == eager."""
    H, storage, W = 16, 128, 8192
    q, swa, ckv, win, topk, bt, sink, _ = _attn_inputs(T, H, seed=100 + T)
    tok2req = torch.arange(T, device=DEV, dtype=torch.int32)
    keys, kbt = _paged_keys(80 * storage, storage, [W] * T, seed=T)
    iq = torch.randn(T, 32, 128, device=DEV).half()
    iw = torch.randn(T, 32, device=DEV)
    nvis = torch.zeros(T, dtype=torch.int64, device=DEV)
    s_win, s_topk, s_q = win.clone().fill_(-1), topk.clone().fill_(-1), q.clone()
    out = torch.empty(T, H, 512, dtype=torch.float16, device=DEV)
    top = torch.empty(T, 512, dtype=torch.int32, device=DEV)

    def step() -> None:
        dk.sparse_attention_decode(s_q, swa, s_win, ckv, s_topk, bt, tok2req, storage, sink, 512 ** -0.5, out)
        sc = dk.index_scores_decode(iq, iw, 0.125, keys, kbt, tok2req, storage, nvis, W)
        dk.topk_decode(sc, nvis, 512, top)

    step()                                            # compile / warm up outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    lens = torch.tensor([(97 * (i + 3)) % W + (600 if i % 2 else 0) for i in range(T)], device=DEV).clamp(max=W)
    for trial in range(2):
        s_win.copy_(win)
        s_topk.copy_(topk)
        s_q.copy_(q * (1 + trial))
        nvis.copy_(lens - 37 * trial)
        graph.replay()
        torch.cuda.synchronize()
        g_out, g_top = out.clone(), top.clone()
        step()
        torch.cuda.synchronize()
        assert torch.equal(g_out, out) and torch.equal(g_top, top), f"T={T} trial {trial}: replay != eager"
        assert torch.isfinite(out.float()).all()
