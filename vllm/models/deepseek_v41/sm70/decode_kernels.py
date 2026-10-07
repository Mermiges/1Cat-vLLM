# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static-shape decode kernels for V4.1 attention (P5-ATTN; owner L-ATTN).

Used for decode batches (``DS41BatchMetadata.decode``) in eager mode and inside FULL CUDA graphs alike, so a
replay is bitwise equal to the eager step. Every launch has a grid fixed by the padded token count ``T`` and the
static widths (window 128, top-k 512, ``max_model_len // ratio``); context lengths, slots and block ids are read on
the device. No host reads, no data-dependent shapes.

Numerics (PORT_DESIGN §4.1 contract unchanged): RMSNorm variance / scores / softmax statistics in FP32 (eps 1e-20),
exp via libdevice ``expf`` (the accurate CUDA ``expf`` torch uses), FP32 latents up to the QATs, FP16 records, FP16
probabilities into PV with FP32 accumulation, global-max two-pass softmax with the sink in the denominator.
Differences to the general (prefill) path are FP32 summation order only (Triton FMA ``tl.dot`` -- SM70 has no
Triton MMA -- instead of cuBLAS HMMA, ``tl.sum`` trees instead of torch reductions); the ratio-2 pair combine is
written to match torch op for op (``enable_fp_fusion=False``).
"""

from __future__ import annotations

import torch
from triton.language.extra import libdevice

from vllm.models.deepseek_v41.common import qat
from vllm.models.deepseek_v41.sm70.indexer_kernels import _rope_fp4_tile
from vllm.triton_utils import tl, triton

RUNNING_MAX_FLOOR = -1e30
ATTN_BLOCK_N = 32          # key rows per score program
ATTN_OUT_BLOCK_N = 64      # key rows per PV step
ATTN_HEAD_BLOCK = 16       # heads per program (tl.dot needs M >= 16)
ATTN_D_SLICE = 64          # output dims per PV program
SCORE_BLOCK_N = 64         # index keys per score step
SCORE_TARGET_PROGRAMS = 320


# ============================================================================ norms
@triton.jit
def _rms(x, w, eps, D: tl.constexpr):
    var = tl.sum(x * x, axis=0) / D
    return x * libdevice.rsqrt(var + eps) * w


@triton.jit
def _qkv_norm_kernel(x_ptr, x_st, qw_ptr, kw_ptr, q32_ptr, q16_ptr, kv_ptr, eps,
                     DQ: tl.constexpr, BQ: tl.constexpr, DK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, BQ)
    m = i < DQ
    xq = tl.load(x_ptr + t * x_st + i, mask=m, other=0.0)
    yq = _rms(xq, tl.load(qw_ptr + i, mask=m, other=0.0), eps, DQ)
    tl.store(q32_ptr + t * DQ + i, yq, mask=m)
    tl.store(q16_ptr + t * DQ + i, yq.to(tl.float16), mask=m)
    j = tl.arange(0, DK)
    xk = tl.load(x_ptr + t * x_st + DQ + j)
    tl.store(kv_ptr + t * DK + j, _rms(xk, tl.load(kw_ptr + j), eps, DK))


def qkv_norm(qr_kv: torch.Tensor, q_w: torch.Tensor, kv_w: torch.Tensor, eps: float, dq: int
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """qr_kv [T, dq + 512] fp32 -> (q_norm(qr) fp32 [T, dq], same in fp16, kv_norm(kv) fp32 [T, 512])."""
    T, width = qr_kv.shape
    dk = width - dq
    q32 = torch.empty((T, dq), dtype=torch.float32, device=qr_kv.device)
    q16 = torch.empty((T, dq), dtype=torch.float16, device=qr_kv.device)
    kv = torch.empty((T, dk), dtype=torch.float32, device=qr_kv.device)
    if T:
        _qkv_norm_kernel[(T,)](qr_kv, qr_kv.stride(0), q_w, kv_w, q32, q16, kv, eps,
                               DQ=dq, BQ=triton.next_power_of_2(dq), DK=dk, num_warps=4)
    return q32, q16, kv


@triton.jit
def _rms_rows_kernel(x_ptr, x_st, w_ptr, out_ptr, eps, D: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, D)
    tl.store(out_ptr + t * D + j, _rms(tl.load(x_ptr + t * x_st + j), tl.load(w_ptr + j), eps, D))


def rms_rows(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """FP32 RMSNorm of the rows of x [T, D] (FP32) -> new FP32 tensor."""
    T, D = x.shape
    out = torch.empty((T, D), dtype=torch.float32, device=x.device)
    if T:
        _rms_rows_kernel[(T,)](x, x.stride(0), w, out, eps, D=D, num_warps=4)
    return out


# ============================================================================ compressor (ratio 2)
@triton.jit
def _ratio2_decode_kernel(ks_ptr, ks_st, state_ptr, slot_ptr, prev_ptr, w_ptr, lat_ptr, eps, D: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, D)
    kv1 = tl.load(ks_ptr + t * ks_st + j)
    sc1 = tl.load(ks_ptr + t * ks_st + D + j)
    slot = tl.load(slot_ptr + t)
    if slot >= 0:                                    # this token's FP32 [kv | score] row of the state cache
        tl.store(state_ptr + slot * (2 * D) + j, kv1)
        tl.store(state_ptr + slot * (2 * D) + D + j, sc1)
    prev = tl.load(prev_ptr + t)
    ok = prev >= 0                                   # completing token: partner p-1 from an earlier step
    pr = tl.where(ok, prev, 0)
    kv0 = tl.load(state_ptr + pr * (2 * D) + j, mask=ok, other=0.0)
    sc0 = tl.load(state_ptr + pr * (2 * D) + D + j, mask=ok, other=0.0)
    # softmax over the pair, op for op as torch's (spatial) softmax and the following sum over the pair
    mx = tl.maximum(sc0, sc1)
    e0 = libdevice.exp(sc0 - mx)
    e1 = libdevice.exp(sc1 - mx)
    s = e0 + e1
    kv = kv0 * (e0 / s) + kv1 * (e1 / s)
    tl.store(lat_ptr + t * D + j, _rms(kv, tl.load(w_ptr + j), eps, D))


def ratio2_decode(kv_score: torch.Tensor, state_rows: torch.Tensor, slots: torch.Tensor, prev_slots: torch.Tensor,
                  norm_w: torch.Tensor, eps: float) -> torch.Tensor:
    """Writes every valid token's [kv | score] row to the state cache, combines completing tokens with the row of
    p-1 (softmax pooling over the pair) and returns the normed FP32 latent [T, 512] (rows of non-completing tokens
    are finite and never stored)."""
    T = kv_score.shape[0]
    D = kv_score.shape[1] // 2
    out = torch.empty((T, D), dtype=torch.float32, device=kv_score.device)
    if T:
        _ratio2_decode_kernel[(T,)](kv_score, kv_score.stride(0), state_rows, slots, prev_slots, norm_w, out, eps,
                                    D=D, num_warps=4, enable_fp_fusion=False)
    return out


# ============================================================================ index keys
@triton.jit
def _index_k_norm_store_kernel(k_ptr, k_st, w_ptr, eps, pos_ptr, cs_ptr, rows_ptr, slot_ptr, exp_ptr,
                               D: tl.constexpr, RD: tl.constexpr, EXPORT: tl.constexpr):
    n = tl.program_id(0).to(tl.int64)
    pos = tl.load(pos_ptr + n).to(tl.int64)
    j = tl.arange(0, D)
    k = _rms(tl.load(k_ptr + n * k_st + j), tl.load(w_ptr + j), eps, D)
    e, o = tl.split(tl.reshape(k, (D // 2, 2)))
    y = tl.reshape(_rope_fp4_tile(e[None, :], o[None, :], cs_ptr, pos, RD, D), (D,)).to(rows_ptr.dtype.element_ty)
    slot = tl.load(slot_ptr + n)
    if slot >= 0:
        tl.store(rows_ptr + slot * D + j, y)
    if EXPORT:
        tl.store(exp_ptr + n * D + j, y)


def index_k_norm_store(k: torch.Tensor, norm_w: torch.Tensor, eps: float, latent_pos: torch.Tensor,
                       cos_sin_cache: torch.Tensor, rows: torch.Tensor, slots: torch.Tensor,
                       export: torch.Tensor | None) -> None:
    """k [N, 128] FP32 (= wk(latent)) -> k_norm -> RoPE at latent_pos -> FP4 x UE8M0/32 QAT -> FP16 rows at slots
    (-1: skip); ``export`` [N, 128] receives the same records."""
    N, D = k.shape
    if N:
        _index_k_norm_store_kernel[(N,)](k, k.stride(0), norm_w, eps, latent_pos, cos_sin_cache, rows, slots,
                                         export if export is not None else rows, D=D, RD=cos_sin_cache.shape[-1],
                                         EXPORT=export is not None, num_warps=2)


# ============================================================================ index scores
@triton.jit
def _index_scores_kernel(q_ptr, w_ptr, wscale, keys_ptr, bt_ptr, bt_st, tok2req_ptr, storage, nvis_ptr,
                         cand_ptr, cand_st, out_ptr, out_st, W,
                         H: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr, NSPLIT: tl.constexpr,
                         CAND: tl.constexpr, CAND_ALL_ID: tl.constexpr):
    """score[t, c] = sum_h relu(q[t,h] . k[pos(c)]) * w[t,h] * wscale for c < n_t, -inf for n_t <= c < W.
    Full scan: pos(c) = c, n_t = visible count. Candidate-compacted (CAND): column c holds position
    cand[t, c // 8] * 8 + c % 8 (identity when cand[t, 0] == CAND_ALL), n_t = visible count (CAND_ALL) or W,
    positions >= visible or in a -1 block are -inf. Static grid (T, NSPLIT); trip counts follow n_t."""
    t = tl.program_id(0).to(tl.int64)
    sp = tl.program_id(1)
    e = tl.load(nvis_ptr + t).to(tl.int32)
    if CAND:
        all_rows = tl.load(cand_ptr + t * cand_st) == CAND_ALL_ID
        n = tl.where(all_rows, e, W)
    else:
        n = e
    n = tl.minimum(n, W)
    h = tl.arange(0, H)
    d = tl.arange(0, D)
    q = tl.load(q_ptr + t * H * D + h[:, None] * D + d[None, :])
    w = tl.load(w_ptr + t * H + h) * wscale
    req = tl.load(tok2req_ptr + t).to(tl.int64)
    chunks = tl.cdiv(W, BLOCK_N)
    per = tl.cdiv(chunks, NSPLIT)
    lo = sp * per * BLOCK_N
    hi = tl.minimum(lo + per * BLOCK_N, W)
    for start in range(lo, hi, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        inb = cols < hi
        if start < n:
            if CAND:
                blk = tl.load(cand_ptr + t * cand_st + cols // 8, mask=inb & (cols < W), other=-1)
                pos = tl.where(all_rows, cols, blk * 8 + cols % 8)
                valid = (cols < n) & (pos < e) & (all_rows | (blk >= 0))
            else:
                pos = cols
                valid = cols < n
            pp = tl.where(valid, pos, 0).to(tl.int64)
            b = tl.load(bt_ptr + req * bt_st + pp // storage, mask=valid, other=0).to(tl.int64)
            row = b * storage + pp % storage
            k = tl.load(keys_ptr + row[:, None] * D + d[None, :], mask=valid[:, None], other=0.0)
            s = tl.dot(q, tl.trans(k))
            s = tl.maximum(s, 0.0) * w[:, None]
            sc = tl.sum(s, axis=0)
            tl.store(out_ptr + t * out_st + cols, tl.where(valid, sc, float("-inf")), mask=inb)
        else:
            tl.store(out_ptr + t * out_st + cols, tl.full((BLOCK_N,), float("-inf"), tl.float32), mask=inb)


def index_scores_decode(q: torch.Tensor, w: torch.Tensor, wscale: float, keys: torch.Tensor,
                        block_table: torch.Tensor, tok2req: torch.Tensor, storage: int, num_visible: torch.Tensor,
                        width: int, cand: torch.Tensor | None = None, cand_all_id: int = -2) -> torch.Tensor:
    """q [T, 32, 128] fp16 (QAT'd), w [T, 32] fp32 (weights_proj output), keys [rows, 128] fp16 (index-K cache of
    the kv source), block_table [T, *] / tok2req [T] / storage of that cache, num_visible [T] -> scores [T, width]
    FP32 (-inf beyond each row's length)."""
    T, H, D = q.shape
    out = torch.empty((T, width), dtype=torch.float32, device=q.device)
    if T == 0:
        return out
    if q.stride(-1) != 1 or q.stride(1) != D or q.stride(0) != H * D or w.stride(0) != H or w.stride(1) != 1:
        raise ValueError("index_scores_decode: q [T, H, D] and w [T, H] must be contiguous")
    chunks = triton.cdiv(width, SCORE_BLOCK_N)
    nsplit = max(1, min(chunks, SCORE_TARGET_PROGRAMS // T))
    _index_scores_kernel[(T, nsplit)](
        q, w, wscale, keys, block_table, block_table.stride(0), tok2req, storage, num_visible,
        cand if cand is not None else num_visible, cand.stride(0) if cand is not None else 0, out, out.stride(0),
        width, H=H, D=D, BLOCK_N=SCORE_BLOCK_N, NSPLIT=nsplit, CAND=cand is not None, CAND_ALL_ID=cand_all_id,
        num_warps=4)
    return out


def topk_decode(scores: torch.Tensor, lengths: torch.Tensor, k: int, out: torch.Tensor,
                cand: torch.Tensor | None = None, cand_all_id: int = -2) -> None:
    """out [T, k] int32 = the top-k columns of scores [T, W >= k] (W static; -inf beyond each row's length),
    ascending, -1 for picks >= length; with ``cand`` (candidate-compacted columns) the columns are mapped to
    positions cand[t, c // 8] * 8 + c % 8 (identity on CAND_ALL rows). Same op sequence as the general path
    (torch.topk unsorted -> sort), on a static width."""
    idx = scores.topk(k, dim=-1, sorted=False).indices.sort(dim=-1).values
    valid = idx < lengths[:, None].to(idx.dtype)
    if cand is not None:
        all_rows = (cand[:, :1] == cand_all_id)
        mapped = cand.gather(1, torch.div(idx, 8, rounding_mode="floor")).to(idx.dtype) * 8 + idx % 8
        idx = torch.where(all_rows, idx, mapped)
    out.copy_(torch.where(valid, idx, torch.full_like(idx, -1)))


# ============================================================================ sparse attention
@triton.jit
def _attn_scores_kernel(q_ptr, q_st, swa_ptr, win_ptr, ckv_ptr, topk_ptr, topk_st, bt_ptr, bt_st, tok2req_ptr,
                        storage, s_ptr, rows_ptr, scale,
                        H: tl.constexpr, HB: tl.constexpr, D: tl.constexpr, KW: tl.constexpr, KC: tl.constexpr,
                        BLOCK_N: tl.constexpr):
    """S[t, h, c] = scale * q[t, h] . K[c] (FP32) over the window rows (c < KW) then the compressed rows (top-k
    logical positions -> rows through the kv source's block table); -inf for invalid columns. Also records the
    row of every column (head block 0) for the PV kernel."""
    t = tl.program_id(0).to(tl.int64)
    nb = tl.program_id(1)
    hb = tl.program_id(2)
    KT: tl.constexpr = KW + KC
    cols = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    is_w = cols < KW
    row = tl.load(win_ptr + t * KW + cols, mask=is_w, other=-1)
    if KC > 0:
        is_c = (cols >= KW) & (cols < KT)
        j = tl.load(topk_ptr + t * topk_st + (cols - KW), mask=is_c, other=-1).to(tl.int64)
        req = tl.load(tok2req_ptr + t).to(tl.int64)
        jj = tl.where(j >= 0, j, 0)
        b = tl.load(bt_ptr + req * bt_st + jj // storage, mask=is_c & (j >= 0), other=0).to(tl.int64)
        row = tl.where(is_w, row, tl.where(j >= 0, b * storage + jj % storage, -1))
    valid = row >= 0
    rr = tl.where(valid, row, 0)
    d = tl.arange(0, D)
    kw = tl.load(swa_ptr + rr[:, None] * D + d[None, :], mask=(valid & is_w)[:, None], other=0.0)
    if KC > 0:
        kc = tl.load(ckv_ptr + rr[:, None] * D + d[None, :], mask=(valid & ~is_w)[:, None], other=0.0)
        k = tl.where(is_w[:, None], kw, kc)
    else:
        k = kw
    h = hb * HB + tl.arange(0, HB)
    q = tl.load(q_ptr + t * q_st + h[:, None] * D + d[None, :])
    s = tl.dot(q, tl.trans(k)) * scale
    s = tl.where(valid[None, :], s, float("-inf"))
    tl.store(s_ptr + t * H * KT + h[:, None] * KT + cols[None, :], s, mask=(cols < KT)[None, :])
    if hb == 0:
        tl.store(rows_ptr + t * KT + cols, row, mask=cols < KT)


@triton.jit
def _attn_out_kernel(s_ptr, rows_ptr, swa_ptr, ckv_ptr, sink_ptr, out_ptr, out_st, floor,
                     H: tl.constexpr, HB: tl.constexpr, D: tl.constexpr, KW: tl.constexpr, KT: tl.constexpr,
                     DS: tl.constexpr, BLOCK_N: tl.constexpr, HAS_C: tl.constexpr):
    """o[t, h, slice] = sum_c fp16(exp(S - m)) V[c, slice] / (sum_c exp(S - m) + exp(sink - m)), m = max(max_c S,
    -1e30): the global-max two-pass softmax of the general path, PV over one 64-dim slice."""
    t = tl.program_id(0).to(tl.int64)
    hb = tl.program_id(1)
    ds = tl.program_id(2)
    h = hb * HB + tl.arange(0, HB)
    srow = s_ptr + t * H * KT + h[:, None] * KT
    m = tl.full((HB,), float("-inf"), tl.float32)
    for k0 in tl.static_range(0, KT, BLOCK_N):
        c = k0 + tl.arange(0, BLOCK_N)
        s = tl.load(srow + c[None, :], mask=(c < KT)[None, :], other=float("-inf"))
        m = tl.maximum(m, tl.max(s, axis=1))
    m = tl.maximum(m, floor)
    dsl = ds * DS + tl.arange(0, DS)
    acc = tl.zeros((HB, DS), dtype=tl.float32)
    l = tl.zeros((HB,), dtype=tl.float32)
    for k0 in tl.static_range(0, KT, BLOCK_N):
        c = k0 + tl.arange(0, BLOCK_N)
        s = tl.load(srow + c[None, :], mask=(c < KT)[None, :], other=float("-inf"))
        p = libdevice.exp(s - m[:, None])
        l += tl.sum(p, axis=1)
        row = tl.load(rows_ptr + t * KT + c, mask=c < KT, other=-1)
        valid = row >= 0
        rr = tl.where(valid, row, 0)
        is_w = c < KW
        v = tl.load(swa_ptr + rr[:, None] * D + dsl[None, :], mask=(valid & is_w)[:, None], other=0.0)
        if HAS_C:
            vc = tl.load(ckv_ptr + rr[:, None] * D + dsl[None, :], mask=(valid & ~is_w)[:, None], other=0.0)
            v = tl.where(is_w[:, None], v, vc)
        acc += tl.dot(p.to(tl.float16), v)
    denom = l + libdevice.exp(tl.load(sink_ptr + h) - m)
    o = acc / denom[:, None]
    tl.store(out_ptr + t * out_st + h[:, None] * D + dsl[None, :], o.to(out_ptr.dtype.element_ty))


def sparse_attention_decode(q: torch.Tensor, swa_rows: torch.Tensor, window_slots: torch.Tensor,
                            ckv_rows: torch.Tensor | None, topk: torch.Tensor | None, block_table: torch.Tensor | None,
                            tok2req: torch.Tensor | None, storage: int, sink: torch.Tensor, scale: float,
                            out: torch.Tensor) -> None:
    """q [T, H, 512] fp16 (rotated), window_slots [T, 128] int64 rows of swa_rows (-1 pad); compressed part:
    topk [T, 512] logical positions (-1 pad) of the kv source -> rows of ckv_rows through block_table (one row per
    token's request, tok2req) with ``storage`` rows per block. Writes out [T, H, 512] fp16 (rows with no valid
    entry -> exactly 0). Two launches, static grids."""
    T, H, D = q.shape
    if T == 0:
        return
    if H % ATTN_HEAD_BLOCK or q.stride(2) != 1 or q.stride(1) != D or out.stride(1) != D or out.stride(2) != 1:
        raise ValueError(f"sparse_attention_decode: q/out must be [T, H, {D}] row-major with H % {ATTN_HEAD_BLOCK} == 0")
    kw = window_slots.shape[1]
    kc = topk.shape[1] if topk is not None else 0
    kt = kw + kc
    s = torch.empty((T, H, kt), dtype=torch.float32, device=q.device)
    rows = torch.empty((T, kt), dtype=torch.int64, device=q.device)
    has_c = topk is not None
    _attn_scores_kernel[(T, triton.cdiv(kt, ATTN_BLOCK_N), H // ATTN_HEAD_BLOCK)](
        q, q.stride(0), swa_rows, window_slots, ckv_rows if has_c else swa_rows, topk if has_c else window_slots,
        topk.stride(0) if has_c else 0, block_table if has_c else window_slots,
        block_table.stride(0) if has_c else 0, tok2req if has_c else window_slots, storage, s, rows, scale,
        H=H, HB=ATTN_HEAD_BLOCK, D=D, KW=kw, KC=kc, BLOCK_N=ATTN_BLOCK_N, num_warps=4)
    _attn_out_kernel[(T, H // ATTN_HEAD_BLOCK, D // ATTN_D_SLICE)](
        s, rows, swa_rows, ckv_rows if has_c else swa_rows, sink, out, out.stride(0), RUNNING_MAX_FLOOR,
        H=H, HB=ATTN_HEAD_BLOCK, D=D, KW=kw, KT=kt, DS=ATTN_D_SLICE, BLOCK_N=ATTN_OUT_BLOCK_N, HAS_C=has_c,
        num_warps=4)


def qat_tile_import_guard() -> None:
    """Keeps the QAT device functions imported here (the fused stores call them through indexer_kernels)."""
    assert qat.fp4_e8m0_qdq_tile is not None
