# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused hyper-connection kernels for DeepSeek-V4.1 on SM70 (PORT_DESIGN §2.2 hc_kernels.py, P5; D18 item 1).

One HC "step" = everything between two sublayers, in TWO launches instead of ~100 eager aten ops:

    stream' = hc_post(sub_out, stream, post, comb)             (optional; BF16 stream, FP32 math)
    pre'/post'/comb' = hc_mixes(stream', fn, scale, base)      (optional; rsqrt-RMS * fn GEMV + 20-iter Sinkhorn)
    act = rmsnorm_to_act(hc_pre(stream', collapse_pre), w)     (optional; FP16 activation of the next sublayer)

This is the semantics of upstream ``mhc_shifted_post_pre`` (the single-pass shift: a sublayer's post is fused with
the NEXT sublayer's mixes and its collapse, which uses the pre computed one sublayer earlier).

Phase A (grid: token tiles x 80 hidden tiles of 64): post -> BF16 stream' (stored), then on the ROUNDED stream'
(the port's stream dtype, as the reference reads it) the partial FP32 sums of the 24 mix dot products, of stream'^2,
and the collapse sum_c pre[c]*stream'[c] -> BF16 (stored) with its partial sum of squares.
Phase B (grid: tokens): reduce the 80 partials, mixes * rsqrt(mean(x^2) + 1e-20), Sinkhorn in the op order of
kernel.py hc_split_sinkhorn, and the RMSNorm of the collapse -> FP16 activation.

Numerics vs common/hc.py: identical formulas and op order, FP32 values everywhere, with two accuracy choices:
the 24 x 20480 mix dot products accumulate in FP64 (tl.dot on SM70 = FMA; V100 FP64 runs at half the FP32 rate and
the GEMV is tiny), and Sinkhorn / sigmoid / rsqrt use correctly rounded division, sqrt and expf (Triton's defaults
are the approximate PTX forms). Measured vs the FP64 truth: pre/post ~6e-8, comb ~1.4e-7 (the FP32 Sinkhorn floor is
~9e-8; the eager FP32 GEMM path is ~2.8e-7; AM-3 gate 1e-6). The BF16 stream and FP16 activation are the correct
rounding of the same FP32 values up to rare 1-ulp ties (hc_out rule: >= 99.9 % equal, all <= 1 ulp).
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, tldevice, triton

from ..common.contracts import HC, HC_EPS, HC_SINKHORN_ITERS, HIDDEN, NORM_EPS, STREAM_DTYPE
from ..common.hc import MIX_HC

_BT = 16            # tokens per phase-A tile when T > _SMALL_T (tl.dot needs M >= 16)
_SMALL_T = 4        # up to here (decode, DSpark verify): 1-token tiles, broadcast-multiply FP64 sums
_BT_SMALL = 1      # (measured on V100: BT=1 beats 2 for decode)
_BD = 64            # hidden columns per phase-A tile
_NB = HIDDEN // _BD  # 80 partials per token
_NBP = 128          # _NB padded to a power of two
_MIXP = 32          # 24 mixes padded; column 24 = sum stream'^2, column 25 = sum collapse^2
_COL_SSQ, _COL_CSSQ = MIX_HC, MIX_HC + 1
_NORM_BLOCK = 1024
assert HIDDEN % _BD == 0 and _NB <= _NBP and MIX_HC + 2 <= _MIXP and HIDDEN % _NORM_BLOCK == 0


@triton.jit
def _hc_phase_a(res_ptr, sub_ptr, post_ptr, comb_ptr, cpre_ptr, fn_ptr, out_ptr, coll_ptr, part_ptr, T,
                HAS_POST: tl.constexpr, DO_MIX: tl.constexpr, DO_COLLAPSE: tl.constexpr, USE_DOT: tl.constexpr,
                BT: tl.constexpr, BD: tl.constexpr, NB: tl.constexpr, H: tl.constexpr, NC: tl.constexpr,
                MIX: tl.constexpr, MIXP: tl.constexpr, COL_SSQ: tl.constexpr, COL_CSSQ: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    rows = pid_t * BT + tl.arange(0, BT)
    rmask = rows < T
    rows64 = rows.to(tl.int64)
    d = pid_d * BD + tl.arange(0, BD)
    m2 = rmask[:, None]
    srow = rows64[:, None] * (NC * H) + d[None, :]          # [BT, BD] offset of copy 0
    r0 = tl.load(res_ptr + srow, mask=m2, other=0.0).to(tl.float32)
    r1 = tl.load(res_ptr + srow + H, mask=m2, other=0.0).to(tl.float32)
    r2 = tl.load(res_ptr + srow + 2 * H, mask=m2, other=0.0).to(tl.float32)
    r3 = tl.load(res_ptr + srow + 3 * H, mask=m2, other=0.0).to(tl.float32)
    if HAS_POST:
        xs = tl.load(sub_ptr + rows64[:, None] * H + d[None, :], mask=m2, other=0.0).to(tl.float32)
    kk = tl.arange(0, MIXP)
    kmask = kk < MIX
    acc = tl.zeros((BT, MIXP), dtype=tl.float64)
    ssq = tl.zeros((BT,), dtype=tl.float32)
    coll = tl.zeros((BT, BD), dtype=tl.float32)
    for c in tl.static_range(NC):
        if HAS_POST:
            cb = comb_ptr + rows64 * (NC * NC) + c                   # comb[t, j, c] at j*NC + c
            v = tl.load(cb, mask=rmask, other=0.0)[:, None] * r0
            v += tl.load(cb + NC, mask=rmask, other=0.0)[:, None] * r1
            v += tl.load(cb + 2 * NC, mask=rmask, other=0.0)[:, None] * r2
            v += tl.load(cb + 3 * NC, mask=rmask, other=0.0)[:, None] * r3
            v += tl.load(post_ptr + rows64 * NC + c, mask=rmask, other=0.0)[:, None] * xs
            vb = v.to(tl.bfloat16)
            tl.store(out_ptr + srow + c * H, vb, mask=m2)
            v = vb.to(tl.float32)
        else:
            if c == 0:
                v = r0
            elif c == 1:
                v = r1
            elif c == 2:
                v = r2
            else:
                v = r3
        if DO_MIX:
            ssq += tl.sum(v * v, axis=1)
            if USE_DOT:     # token tiles of 16 (prefill): FMA-based FP64 dot
                fnt = tl.load(fn_ptr + kk[None, :].to(tl.int64) * (NC * H) + c * H + d[:, None],
                              mask=kmask[None, :], other=0.0)            # [BD, MIXP] = fn[k, c*H + d]^T
                acc += tl.dot(v.to(tl.float64), fnt.to(tl.float64))
            else:           # decode-sized tiles: broadcast products, FP64 row sums (no 16-row padding)
                fnk = tl.load(fn_ptr + kk[:, None].to(tl.int64) * (NC * H) + c * H + d[None, :],
                              mask=kmask[:, None], other=0.0)            # [MIXP, BD]
                acc += tl.sum(v.to(tl.float64)[:, None, :] * fnk.to(tl.float64)[None, :, :], axis=2)
        if DO_COLLAPSE:
            coll += tl.load(cpre_ptr + rows64 * NC + c, mask=rmask, other=0.0)[:, None] * v
    pbase = part_ptr + rows64 * (NB * MIXP) + pid_d * MIXP
    if DO_MIX:
        tl.store(pbase[:, None] + kk[None, :], acc.to(tl.float32), mask=m2 & kmask[None, :])
        tl.store(pbase + COL_SSQ, ssq, mask=rmask)
    if DO_COLLAPSE:
        cb16 = coll.to(tl.bfloat16)
        tl.store(coll_ptr + rows64[:, None] * H + d[None, :], cb16, mask=m2)
        cf = cb16.to(tl.float32)
        tl.store(pbase + COL_CSSQ, tl.sum(cf * cf, axis=1), mask=rmask)


@triton.jit
def _rsqrt_rn(x):
    # correctly rounded 1/sqrt (Triton's default sqrt / division are the approximate PTX forms)
    return tl.math.div_rn(1.0, tl.math.sqrt_rn(x))


@triton.jit
def _sigmoid_rn(x):
    return tl.math.div_rn(1.0, 1.0 + tldevice.exp(-x))


@triton.jit
def _hc_phase_b(part_ptr, scale_ptr, base_ptr, coll_ptr, w_ptr, pre_ptr, post_ptr, comb_ptr, act_ptr,
                DO_MIX: tl.constexpr, DO_COLLAPSE: tl.constexpr, NB: tl.constexpr, NBP: tl.constexpr,
                H: tl.constexpr, NC: tl.constexpr, MIXP: tl.constexpr, COL_SSQ: tl.constexpr,
                COL_CSSQ: tl.constexpr, ITERS: tl.constexpr, HC_EPS_C: tl.constexpr, NORM_EPS_C: tl.constexpr,
                NORM_BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)          # the RMSNorm is split over H // NORM_BLOCK programs; program 0 also mixes
    nb = tl.arange(0, NBP)
    nbm = nb < NB
    prow = part_ptr + t * (NB * MIXP) + nb * MIXP              # [NBP]
    if DO_MIX and split == 0:
        k4 = tl.arange(0, NC)
        k16 = tl.arange(0, NC * NC)
        ssq = tl.sum(tl.load(prow + COL_SSQ, mask=nbm, other=0.0), axis=0)
        rs = _rsqrt_rn(ssq / (NC * H) + NORM_EPS_C)
        m_pre = tl.sum(tl.load(prow[:, None] + k4[None, :], mask=nbm[:, None], other=0.0), axis=0) * rs
        m_post = tl.sum(tl.load(prow[:, None] + NC + k4[None, :], mask=nbm[:, None], other=0.0), axis=0) * rs
        m_comb = tl.sum(tl.load(prow[:, None] + 2 * NC + k16[None, :], mask=nbm[:, None], other=0.0), axis=0) * rs
        s0 = tl.load(scale_ptr)
        s1 = tl.load(scale_ptr + 1)
        s2 = tl.load(scale_ptr + 2)
        pre = _sigmoid_rn(m_pre * s0 + tl.load(base_ptr + k4)) + HC_EPS_C
        post = 2.0 * _sigmoid_rn(m_post * s1 + tl.load(base_ptr + NC + k4))
        cmb = tl.reshape(m_comb * s2 + tl.load(base_ptr + 2 * NC + k16), (NC, NC))
        cmb = tldevice.exp(cmb - tl.max(cmb, axis=1)[:, None])
        cmb = tl.math.div_rn(cmb, tl.sum(cmb, axis=1)[:, None]) + HC_EPS_C
        cmb = tl.math.div_rn(cmb, tl.sum(cmb, axis=0)[None, :] + HC_EPS_C)
        for _ in range(ITERS - 1):
            cmb = tl.math.div_rn(cmb, tl.sum(cmb, axis=1)[:, None] + HC_EPS_C)
            cmb = tl.math.div_rn(cmb, tl.sum(cmb, axis=0)[None, :] + HC_EPS_C)
        tl.store(pre_ptr + t * NC + k4, pre)
        tl.store(post_ptr + t * NC + k4, post)
        tl.store(comb_ptr + t * (NC * NC) + k16, tl.reshape(cmb, (NC * NC,)))
    if DO_COLLAPSE:
        cssq = tl.sum(tl.load(prow + COL_CSSQ, mask=nbm, other=0.0), axis=0)
        crs = _rsqrt_rn(cssq / H + NORM_EPS_C)
        offs = split * NORM_BLOCK + tl.arange(0, NORM_BLOCK)
        x = tl.load(coll_ptr + t * H + offs).to(tl.float32)
        w = tl.load(w_ptr + offs)
        tl.store(act_ptr + t * H + offs, (w * (x * crs)).to(tl.float16))


def _check(stream: torch.Tensor) -> None:
    if stream.dim() != 3 or stream.shape[1] != HC or stream.shape[2] != HIDDEN or stream.dtype != STREAM_DTYPE:
        raise ValueError(f"HC stream must be [T, {HC}, {HIDDEN}] {STREAM_DTYPE}, got {stream.dtype} "
                         f"{tuple(stream.shape)}")
    if not stream.is_cuda or not stream.is_contiguous():
        raise ValueError("fused HC kernels need a contiguous CUDA stream")


def hc_fused_step(stream: torch.Tensor, *,
                  sub_out: torch.Tensor | None = None, post: torch.Tensor | None = None,
                  comb: torch.Tensor | None = None,
                  mix: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
                  collapse_pre: torch.Tensor | None = None, norm_weight: torch.Tensor | None = None,
                  ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
                             torch.Tensor | None, torch.Tensor | None]:
    """One fused HC step (module docstring). Returns (stream', (pre, post, comb) of ``mix`` or None,
    FP16 activation rmsnorm(collapsed) * norm_weight or None, collapsed = hc_pre(stream', collapse_pre) BF16 or None).
    ``sub_out`` [T,5120] (any float dtype) with ``post`` [T,4] f32 / ``comb`` [T,4,4] f32 applies hc_post first
    (stream' is a new tensor); without it stream' is ``stream`` itself."""
    _check(stream)
    has_post = sub_out is not None
    if has_post != (post is not None) or has_post != (comb is not None):
        raise ValueError("sub_out, post and comb go together")
    do_collapse = collapse_pre is not None
    if do_collapse != (norm_weight is not None):
        raise ValueError("collapse_pre and norm_weight go together")
    do_mix = mix is not None
    num_tokens = stream.shape[0]
    dev = stream.device
    out = torch.empty_like(stream) if has_post else stream
    mixes = (torch.empty((num_tokens, HC), dtype=torch.float32, device=dev),
             torch.empty((num_tokens, HC), dtype=torch.float32, device=dev),
             torch.empty((num_tokens, HC, HC), dtype=torch.float32, device=dev)) if do_mix else None
    act = torch.empty((num_tokens, HIDDEN), dtype=torch.float16, device=dev) if do_collapse else None
    coll = torch.empty((num_tokens, HIDDEN), dtype=STREAM_DTYPE, device=dev) if do_collapse else None
    if num_tokens == 0 or not (has_post or do_mix or do_collapse):
        return out, mixes, act, coll
    if has_post:
        if sub_out.shape != (num_tokens, HIDDEN) or not sub_out.is_contiguous():
            raise ValueError(f"sub_out must be contiguous [{num_tokens}, {HIDDEN}], got {tuple(sub_out.shape)}")
        if post.shape != (num_tokens, HC) or comb.shape != (num_tokens, HC, HC):
            raise ValueError(f"post/comb must be [T,4]/[T,4,4], got {tuple(post.shape)}/{tuple(comb.shape)}")
        post, comb = post.float().contiguous(), comb.float().contiguous()
    if do_mix:
        fn, scale, base = mix
        if fn.dtype != torch.float32 or fn.shape != (MIX_HC, HC * HIDDEN) or not fn.is_contiguous():
            raise ValueError(f"hc fn must be contiguous float32 [{MIX_HC}, {HC * HIDDEN}]")
        scale, base = scale.float().contiguous(), base.float().contiguous()
    if do_collapse:
        if collapse_pre.shape != (num_tokens, HC):
            raise ValueError(f"collapse_pre must be [{num_tokens}, {HC}], got {tuple(collapse_pre.shape)}")
        collapse_pre = collapse_pre.float().contiguous()
        norm_weight = norm_weight.float().contiguous()
    part = torch.empty((num_tokens, _NB, _MIXP), dtype=torch.float32, device=dev) if (do_mix or do_collapse) \
        else stream
    dummy = stream
    use_dot = num_tokens > _SMALL_T
    bt = _BT if use_dot else _BT_SMALL
    _hc_phase_a[(triton.cdiv(num_tokens, bt), _NB)](
        stream, sub_out if has_post else dummy, post if has_post else dummy, comb if has_post else dummy,
        collapse_pre if do_collapse else dummy, mix[0] if do_mix else dummy, out, coll if do_collapse else dummy,
        part, num_tokens,
        HAS_POST=has_post, DO_MIX=do_mix, DO_COLLAPSE=do_collapse, USE_DOT=use_dot, BT=bt, BD=_BD, NB=_NB, H=HIDDEN, NC=HC,
        MIX=MIX_HC, MIXP=_MIXP, COL_SSQ=_COL_SSQ, COL_CSSQ=_COL_CSSQ, num_warps=4)
    if do_mix or do_collapse:
        _hc_phase_b[(num_tokens, HIDDEN // _NORM_BLOCK if do_collapse else 1)](
            part, scale if do_mix else dummy, base if do_mix else dummy, coll if do_collapse else dummy,
            norm_weight if do_collapse else dummy,
            mixes[0] if do_mix else dummy, mixes[1] if do_mix else dummy, mixes[2] if do_mix else dummy,
            act if do_collapse else dummy,
            DO_MIX=do_mix, DO_COLLAPSE=do_collapse, NB=_NB, NBP=_NBP, H=HIDDEN, NC=HC, MIXP=_MIXP,
            COL_SSQ=_COL_SSQ, COL_CSSQ=_COL_CSSQ, ITERS=HC_SINKHORN_ITERS, HC_EPS_C=HC_EPS, NORM_EPS_C=NORM_EPS,
            NORM_BLOCK=_NORM_BLOCK, num_warps=1)
    return out, mixes, act, coll
