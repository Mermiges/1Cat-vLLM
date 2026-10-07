# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused decode publication, routing, spill tables and mutable graph replay."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import vllm._custom_ops as ops
from vllm.models.deepseek_v41.sm70 import gemv
from vllm.models.deepseek_v41.sm70 import moe_kernels as mk
from vllm.models.deepseek_v41.sm70.moe_decode import DecodeScratch, decode_front
from vllm.models.deepseek_v41.sm70.moe_kernels import RouteTables

from .test_moe_method import SyntheticCheckpoint

pytestmark = pytest.mark.sm70


@pytest.mark.parametrize("experts,top_k", [(384, 6), (128, 3)])
@pytest.mark.parametrize("spill", [False, True])
def test_decode_front_publication_and_graph(
    experts: int, top_k: int, spill: bool
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(experts)
    gate = (
        torch.randn(experts, 5120, device="cuda", generator=generator) * 0.02
    ).half()
    bias = torch.randn(experts, device="cuda", generator=generator) * 0.03 + 10.8
    shared = (torch.randn(1152, 5120, device="cuda", generator=generator) * 0.01).half()
    x = torch.randn(1, 5120, device="cuda", generator=generator).half()
    phys = (
        torch.randperm(experts, device="cuda", generator=generator).int()
        if spill
        else None
    )
    resident = experts - 17 if spill else experts
    scratch = DecodeScratch.allocate(experts, x.device)

    def fused() -> tuple[torch.Tensor, torch.Tensor, RouteTables, torch.Tensor]:
        return decode_front(
            x,
            gate,
            bias,
            shared,
            scratch,
            top_k=top_k,
            alpha=1.0,
            scale=1.5,
            limit=10.0,
            phys_map=phys,
            n_resident=resident,
            spill=spill,
        )

    def check(
        result: tuple[torch.Tensor, torch.Tensor, RouteTables, torch.Tensor],
    ) -> None:
        weights, ids, (perm, tables), act = result
        ref_logits = gemv.gemv(x, gate)
        torch.testing.assert_close(scratch.logits, ref_logits[0], rtol=0, atol=0)
        ref_w = torch.empty_like(weights)
        ref_ids = torch.empty_like(ids)
        ops.topk_hash_softplus_sqrt(
            ref_w,
            ref_ids,
            torch.empty_like(ids),
            ref_logits,
            True,
            1.5,
            bias,
            None,
            None,
        )
        assert torch.equal(ids, ref_ids)
        torch.testing.assert_close(weights, ref_w, rtol=0, atol=0)
        ref_perm, ref_tables = mk.route_prep(ids, phys, resident, spill)
        assert torch.equal(perm, ref_perm)
        for (gids, goff), (rgids, rgoff) in zip(tables, ref_tables):
            assert torch.equal(gids, rgids)
            assert torch.equal(goff, rgoff)
        assert torch.equal(act, gemv.gate_up_swiglu(x, shared, 10.0))
        assert scratch.counter.item() == 0

    for _ in range(4):
        check(fused())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = fused()
    for _ in range(100):
        x.copy_(torch.randn(x.shape, device="cuda", generator=generator).half())
        graph.replay()
        check(result)
    # Exact ties exercise the low-id tie break; padding stays finite.
    x.zero_()
    bias.fill_(10.8)
    graph.replay()
    check(result)
    assert torch.equal(
        result[1], torch.arange(top_k, device="cuda", dtype=torch.int32)[None]
    )


def test_decode_front_matches_block(  # noqa: F811
    moe_env: Callable[..., None],  # noqa: F811
    ckpt: SyntheticCheckpoint,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from .test_moe_layer import _x, build_block

    moe_env()
    monkeypatch.delenv("VLLM_DS41_MOE_DECODE_FRONT", raising=False)
    block = build_block(384, 6, ckpt)
    assert block.decode_front
    for seed in range(4):
        x = _x(1, seed)
        block.decode_front = False
        reference = block(x)
        baseline_logits = block.gate_logits(x)
        baseline_w, baseline_ids = block.route(baseline_logits)
        w13, _ = block.shared_experts.fp16_weights()
        fw, fi, tables, fa = decode_front(
            x,
            block.gate.weight,
            block.gate.e_score_correction_bias,
            w13,
            block._decode_scratch,
            top_k=6,
            alpha=2.0**-block.gate.weight_exp,
            scale=1.5,
            limit=10.0,
            phys_map=None,
            n_resident=384,
            spill=False,
        )
        torch.testing.assert_close(fw, baseline_w, rtol=0, atol=0)
        assert torch.equal(baseline_logits[0], block._decode_scratch.logits)
        assert torch.equal(fi, baseline_ids)
        assert torch.equal(fa, gemv.gate_up_swiglu(x, w13, 10.0))
        for left, right in zip(tables[1], mk.route_prep(fi, None, 384, False)[1]):
            assert all(torch.equal(a, b) for a, b in zip(left, right))
        block.decode_front = True
        torch.testing.assert_close(block(x), reference, rtol=1e-5, atol=1e-5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = block(x)
    for seed in range(4, 8):
        x.copy_(_x(1, seed))
        graph.replay()
        torch.testing.assert_close(result, block(x), rtol=0, atol=0)


from .test_moe_layer import ckpt, moe_env  # noqa: E402,F401
