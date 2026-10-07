# ruff: noqa: F811
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full attention-stage capture/replay, cache lifecycle and deferred mirror errors."""

from __future__ import annotations

import pytest
import torch

from vllm.forward_context import ForwardContext, override_forward_context
from vllm.models.deepseek_v41.common.contracts import CAND_ALL, StagePlan
from vllm.models.deepseek_v41.kv_mirror import kv20_crc
from vllm.models.deepseek_v41.sm70.sparse import DECODE_PATH_ENV

from .test_attn_harness import ref_config, synthetic_attn_weights, topology
from .test_attn_layers import DEV, Stage, dist_env  # noqa: F401

pytestmark = pytest.mark.sm70


def _stage(monkeypatch, first: int, last: int, bucket: int) -> Stage:
    import vllm.model_executor.layers.linear as lm
    import vllm.models.deepseek_v41.attention as am

    # One TP4 rank's entire attention stage; collectives belong to CORE and are
    # tested separately.
    monkeypatch.setattr(am, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(lm, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(am, "tensor_model_parallel_all_reduce", lambda x: x)
    monkeypatch.setenv(DECODE_PATH_ENV, "1")
    monkeypatch.setenv("VLLM_DS41_ATTN_MIRROR_CHECK", "1")
    cfg = ref_config()
    ids = tuple(range(first, last + 1))
    weights = {
        i: synthetic_attn_weights(cfg, topology(cfg, i), DEV, seed=i) for i in ids
    }
    rank = 0 if first == 0 else 1 if first == 14 else 2
    plan = StagePlan(
        rank, 3, first, last, (20,) if rank == 2 else (), (20,) if rank == 1 else (), ()
    )
    st = Stage(
        cfg,
        ids,
        weights,
        stage=plan,
        mirror=rank == 2,
        max_model_len=18000,
        num_blocks=1300,
        tp_size=4,
        max_seqs=bucket,
    )
    for a in st.attn.values():
        if a.compressor is not None:
            a.compressor.__dict__.pop(
                "forward", None
            )  # recording is a host-syncing test helper
    del weights
    # Synthetic cache history tests arbitrary exact bytes. Real-weight numerics
    # are gated by test_attn_golden.
    for c in st.sim.caches.values():
        c.tensor.uniform_(-0.2, 0.2)
    return st


@pytest.mark.parametrize("bucket", [1, 2, 8, 32, 64])
@pytest.mark.parametrize("bounds", [(0, 13), (14, 27), (28, 39)])
def test_full_attention_stage_capture_replay(
    dist_env, monkeypatch, bucket: int, bounds: tuple[int, int]
) -> None:
    torch.manual_seed(42)
    st = _stage(monkeypatch, *bounds, bucket)
    xs = {i: torch.randn(bucket, 5120, device=DEV).half() for i in st.layer_ids}
    pos = torch.zeros(bucket, dtype=torch.int64, device=DEV)
    payload = (
        torch.randn(bucket, 512, device=DEV).half(),
        torch.randn(bucket, 128, device=DEV).half(),
        torch.full((bucket, 2048), -1, dtype=torch.int32, device=DEV),
    )
    payload[2][:, 0] = CAND_ALL
    crc = kv20_crc(payload[0], payload[1])
    md, _ = st.sim.metadata([("dummy", 0, 1)], pad_tokens=bucket, pad_reqs=bucket)
    for m in md.values():
        m.slot_mapping.fill_(-1)
        if hasattr(m, "latent_slots"):
            m.latent_slots.fill_(-1)
        if hasattr(m, "window_slots"):
            m.window_slots.fill_(-1)
        if hasattr(m, "num_visible"):
            m.num_visible.zero_()
    fc = ForwardContext(no_compile_layers=st.ctx, attn_metadata=md, slot_mapping={})

    def step() -> dict[int, torch.Tensor]:
        with override_forward_context(fc):
            if st.mirror is not None:
                st.mirror.ingest(pos, *payload, crc)
            return {i: st.attn[i](pos, xs[i]) for i in st.layer_ids}

    step()  # warm kernels, FP32 weight copies and cuBLAS outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()

    # Prefix cache at a block boundary and subsequent preemption must update the
    # same graph addresses.
    st.sim.ensure("a", 17001)
    st.sim.share_prefix("prefix", "a", 1024)
    st.sim.ensure("preempt", 1025)
    st.sim.release("preempt")
    # Return to a short context AFTER a >16K replay: catches stale
    # metadata/candidate masks.
    for req, context in [
        ("a", 1024),
        ("a", 17000),
        ("prefix", 1024),
        ("preempt", 1024),
    ]:
        if req == "preempt":
            # Recompute restores history in newly allocated blocks before
            # decoding resumes.
            st.sim.ensure(req, context + 1)
            for c in st.sim.caches.values():
                for b in c.blocks[req]:
                    c.tensor[b].uniform_(-0.2, 0.2)
        fc.attn_metadata, real_pos = st.sim.metadata(
            [(req, context, 1)], pad_tokens=bucket, pad_reqs=bucket
        )
        pos.copy_(real_pos)
        for x in xs.values():
            x.normal_()
        if st.mirror is not None and context > 16384:
            payload[2].copy_(
                torch.arange(2048, device=DEV, dtype=torch.int32)[None].expand(
                    bucket, -1
                )
            )
        graph.replay()
        torch.cuda.synchronize()
        got = {i: y.clone() for i, y in captured.items()}
        cache_got = {n: c.tensor.clone() for n, c in st.sim.caches.items()}
        top = st.shared.topk_indices.clone()
        cand = st.shared.candidate_blocks.clone()
        if context > 16384 and bounds[0] >= 14:
            assert cand[0, 0] != CAND_ALL, "long replay must use actual candidates"
        eager = step()
        for i, y in eager.items():
            assert torch.equal(got[i], y), (bounds, bucket, req, context, i)
            assert torch.isfinite(y).all()
            assert (y[1:] == 0).all(), "padding output must be zero"
        assert torch.equal(top, st.shared.topk_indices)
        assert torch.equal(cand, st.shared.candidate_blocks)
        for name, c in st.sim.caches.items():
            assert torch.equal(
                cache_got[name].view(torch.uint8), c.tensor.view(torch.uint8)
            ), name
        if st.mirror is not None:
            mm = fc.attn_metadata[st.mirror.ckv_cache.layer_name]
            im = fc.attn_metadata[st.mirror.ik_cache.layer_name]
            assert torch.equal(
                st.mirror.ckv_cache.rows()[mm.latent_slots[0]], payload[0][0]
            )
            assert torch.equal(
                st.mirror.ik_cache.rows()[im.latent_slots[0]], payload[1][0]
            )
            st.mirror.check_pending_errors()


def test_decode_mirror_crc_fails_at_next_metadata_build(dist_env, monkeypatch) -> None:
    st = _stage(monkeypatch, 28, 28, 2)
    md, pos = st.sim.metadata([("a", 0, 1)], pad_tokens=2, pad_reqs=2)
    ckv = torch.randn(2, 512, device=DEV).half()
    ik = torch.randn(2, 128, device=DEV).half()
    cand = torch.full((2, 2048), -1, dtype=torch.int32, device=DEV)
    crc = kv20_crc(ckv, ik)
    fc = ForwardContext(no_compile_layers=st.ctx, attn_metadata=md, slot_mapping={})
    with override_forward_context(fc):
        st.mirror.ingest(pos, ckv, ik, cand, crc)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            st.mirror.ingest(pos, ckv, ik, cand, crc)
    # Error injected only after capture; a replay must check LIVE payload bytes.
    ckv.view(torch.int16)[0, 0] ^= 1
    graph.replay()
    with pytest.raises(
        RuntimeError, match="1 CRC/slot errors from preceding decode ingest/replay"
    ):
        st.sim.metadata([("a", 1, 1)], pad_tokens=2, pad_reqs=2)
    # Error counter is sticky: catching an exception cannot silently resume a
    # corrupted mirror.
    with pytest.raises(RuntimeError, match="CRC/slot errors"):
        st.sim.metadata([("a", 1, 1)], pad_tokens=2, pad_reqs=2)
