# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-token router, top-k, routing tables and shared SwiGLU in one launch.

Router CTAs publish disjoint FP32 logits then increment a module-local counter.
The last router CTA acquires all publications and selects/routes the experts.
Shared CTAs have no dependency on routing. No spinning or host synchronization.
The counter self-resets; each module's forwards must use one ordered stream,
as with its other decode workspaces. Scratch is allocated before graph capture.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton

from .gemv import _dot_rows
from .moe_kernels import RouteTables, _route_prep_small


@triton.jit
def _decode_front_kernel(
    x,
    gate_w,
    bias,
    shared_w,
    logits,
    counter,
    weights,
    ids,
    shared_act,
    phys,
    perm,
    gids0,
    goff0,
    gids1,
    goff1,
    E: tl.constexpr,
    H: tl.constexpr,
    INTER: tl.constexpr,
    TOP_K: tl.constexpr,
    ALPHA: tl.constexpr,
    SCALE: tl.constexpr,
    LIMIT: tl.constexpr,
    N_RES: tl.constexpr,
    HAS_PHYS: tl.constexpr,
    SPILL: tl.constexpr,
    ROUTER_CTAS: tl.constexpr,
    E_PAD: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.arange(0, 2)
    if pid < ROUTER_CTAS:
        erows = pid * 2 + rows
        acc = _dot_rows(x, H, gate_w, erows, erows < E, H, 1, 1, 2, 1024)
        values = tl.reshape(acc, (2,)) * ALPHA
        tl.store(logits + erows, values, erows < E)
        # Every writer fences its store before the CTA's single publication.
        tl.inline_asm_elementwise(
            "membar.gl; mov.b32 $0, $1;",
            constraints="=f,f",
            args=[values],
            dtype=tl.float32,
            is_pure=False,
            pack=1,
        )
        tl.debug_barrier()
        ticket = tl.atomic_add(counter, 1, sem="acq_rel", scope="gpu")
        if ticket == ROUTER_CTAS - 1:
            es = tl.arange(0, E_PAD)
            val = tl.load(logits + es, es < E, other=0.0, cache_modifier=".cv")
            # CUDA uses __logf/__expf, not Triton's log2-based tl.log.
            # Their last-bit difference survives bias removal and can amplify
            # through large expert outputs. Match those intrinsics explicitly.
            exp_val = tl.extra.cuda.libdevice.fast_expf(val)
            softplus = tl.where(
                val > 20.0, val, tl.extra.cuda.libdevice.fast_logf(1.0 + exp_val)
            )
            score = tl.extra.cuda.libdevice.sqrt_rn(softplus)
            b = tl.load(bias + es, es < E, other=0.0)
            ranked = tl.where(es < E, score + b, -float("inf"))
            slots = tl.arange(0, 16)
            picked = tl.full((16,), 0, tl.int32)
            selected = tl.full((16,), 0.0, tl.float32)
            total = tl.full((), 0.0, tl.float32)
            for j in tl.static_range(TOP_K):
                best = tl.max(ranked, 0)
                eid = tl.min(tl.where(ranked == best, es, 2147483647), 0)
                # CUDA removes the selection bias after choosing each winner.
                s = best - tl.sum(tl.where(es == eid, b, 0.0), 0)
                picked = tl.where(slots == j, eid, picked)
                selected = tl.where(slots == j, s, selected)
                total += s
                ranked = tl.where(es == eid, -float("inf"), ranked)
            scale = tl.div_rn(SCALE, tl.where(total > 0.0, total, 1.0))
            tl.store(weights + slots, selected * scale, slots < TOP_K)
            tl.store(ids + slots, picked, slots < TOP_K)
            _route_prep_small(
                picked,
                phys,
                perm,
                gids0,
                goff0,
                gids1,
                goff1,
                TOP_K,
                N_RES,
                HAS_PHYS,
                SPILL,
                16,
            )
            tl.atomic_xchg(counter, 0, sem="release", scope="gpu")
    else:
        srows = (pid - ROUTER_CTAS) * 2 + rows
        mask = srows < INTER
        g = _dot_rows(x, H, shared_w, srows, mask, H, 1, 1, 2, 1024)
        u = _dot_rows(x, H, shared_w, srows + INTER, mask, H, 1, 1, 2, 1024)
        g = tl.minimum(tl.reshape(g, (2,)).to(tl.float16).to(tl.float32), LIMIT)
        u = tl.minimum(
            tl.maximum(tl.reshape(u, (2,)).to(tl.float16).to(tl.float32), -LIMIT), LIMIT
        )
        act = tl.div_rn(g, 1.0 + tl.extra.cuda.libdevice.exp(-g)) * u
        tl.store(shared_act + srows, act.to(tl.float16), mask)


@dataclass
class DecodeScratch:
    logits: torch.Tensor
    counter: torch.Tensor

    @classmethod
    def allocate(cls, n_experts: int, device: torch.device) -> DecodeScratch:
        return cls(
            torch.empty(n_experts, dtype=torch.float32, device=device),
            torch.zeros((), dtype=torch.int32, device=device),
        )


def decode_front(
    x: torch.Tensor,
    gate: torch.Tensor,
    bias: torch.Tensor,
    shared_w13: torch.Tensor,
    scratch: DecodeScratch,
    *,
    top_k: int,
    alpha: float,
    scale: float,
    limit: float,
    phys_map: torch.Tensor | None,
    n_resident: int,
    spill: bool,
) -> tuple[torch.Tensor, torch.Tensor, RouteTables, torch.Tensor]:
    """Return weights, ids, precomputed skinny tables and shared activation (T=1)."""
    if x.dtype != torch.float16 or x.shape != (1, 5120) or not x.is_contiguous():
        raise TypeError("decode_front requires contiguous FP16 [1, 5120]")
    e, h = gate.shape
    inter = shared_w13.shape[0] // 2
    if (e, top_k) not in ((384, 6), (128, 3)):
        raise ValueError(f"unsupported decode experts/top-k {e}/{top_k}")
    if (
        gate.dtype != torch.float16
        or not gate.is_contiguous()
        or h != 5120
        or shared_w13.dtype != torch.float16
        or not shared_w13.is_contiguous()
        or shared_w13.shape != (2 * inter, h)
        or bias.shape != (e,)
        or bias.dtype != torch.float32
        or not bias.is_contiguous()
    ):
        raise TypeError("decode_front requires contiguous FP16 weights and FP32 bias")
    if (
        scratch.logits.shape != (e,)
        or scratch.logits.dtype != torch.float32
        or scratch.counter.shape != ()
        or scratch.counter.dtype != torch.int32
    ):
        raise ValueError("decode_front scratch does not match router")
    tensors = (gate, bias, shared_w13, scratch.logits, scratch.counter)
    if not x.is_cuda or any(t.device != x.device for t in tensors):
        raise ValueError("decode_front operands must share the input CUDA device")
    if phys_map is not None and (
        phys_map.shape != (e,)
        or phys_map.dtype != torch.int32
        or not phys_map.is_contiguous()
        or phys_map.device != x.device
    ):
        raise TypeError("decode_front requires a contiguous device int32 physical map")
    if not 0 <= n_resident <= e or spill != (n_resident < e):
        raise ValueError("decode_front spill flag and resident count disagree")
    if spill and phys_map is None:
        raise ValueError("decode_front spill requires a physical expert map")
    ids = torch.empty((1, top_k), dtype=torch.int32, device=x.device)
    weights = torch.empty((1, top_k), dtype=torch.float32, device=x.device)
    act = torch.empty((1, inter), dtype=torch.float16, device=x.device)
    perm = torch.empty(top_k, dtype=torch.int32, device=x.device)
    gids0 = torch.empty_like(perm)
    goff0 = torch.empty(top_k + 1, dtype=torch.int32, device=x.device)
    gids1 = torch.empty_like(perm) if spill else gids0
    goff1 = torch.empty_like(goff0) if spill else goff0
    p = triton.cdiv(e, 2)
    _decode_front_kernel[(p + triton.cdiv(inter, 2),)](
        x,
        gate,
        bias,
        shared_w13,
        scratch.logits,
        scratch.counter,
        weights,
        ids,
        act,
        phys_map if phys_map is not None else ids,
        perm,
        gids0,
        goff0,
        gids1,
        goff1,
        e,
        h,
        inter,
        top_k,
        float(alpha),
        float(scale),
        float(limit),
        n_resident,
        phys_map is not None,
        spill,
        p,
        triton.next_power_of_2(e),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return (
        weights,
        ids,
        (perm, [(gids0, goff0)] + ([(gids1, goff1)] if spill else [])),
        act,
    )
