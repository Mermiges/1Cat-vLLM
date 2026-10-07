# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-graph-safe decode metadata for the V4.1 cache groups (P5-ATTN; owner L-ATTN).

A batch is a *decode batch* when every real request schedules exactly one token and the runner's padded request
count equals the padded token count (FULL-decode CUDA graphs: ``num_reqs = num_actual_tokens = T_pad``; padded
requests have zero query tokens, ``seq_len`` 0, null-block table rows and slot -1). For such a batch every tensor a
layer reads lives in a buffer owned by the builder (allocated once, capacity ``max_num_seqs``), so the addresses a
captured graph bakes in stay valid on every replay; one Triton kernel per builder and step rewrites them on the
device from the runner's ``seq_lens`` / ``slot_mapping`` / ``block_table`` (no host round trip, no per-field ops).

Token ``t`` of a decode batch belongs to request ``t``. It is *valid* iff ``t < T_real`` (host count of real
tokens), its runner slot is ``>= 0`` and its ``seq_len > 0``; an invalid token gets position 0, slot -1, no window
rows, no visible compressed entries and no latent -- so it reads nothing, writes nothing and attends to nothing
(its attention output row is exactly 0, as the general path's padding rows).
"""

from __future__ import annotations

import torch

from vllm.models.deepseek_v41.common.contracts import WINDOW
from vllm.triton_utils import tl, triton

KIND_SWA = 0
KIND_COMPRESSED = 1
KIND_STATE = 2


@triton.jit
def _decode_md_kernel(seq_ptr, rslot_ptr, bt_in_ptr, bt_in_stride, bt_in_width,
                      bt_ptr, bt_stride, pos_ptr, slot_ptr, aux_ptr, aux2_ptr, aux3_ptr, win_ptr,
                      t_real, block_size, storage,
                      KIND: tl.constexpr, RATIO: tl.constexpr, W: tl.constexpr, COPY_BLOCK: tl.constexpr):
    t = tl.program_id(0)
    t64 = t.to(tl.int64)
    # copy this request's block-table row into the builder-owned table (the graph reads the copy)
    for c0 in range(0, bt_in_width, COPY_BLOCK):
        c = c0 + tl.arange(0, COPY_BLOCK)
        m = c < bt_in_width
        tl.store(bt_ptr + t64 * bt_stride + c, tl.load(bt_in_ptr + t64 * bt_in_stride + c, mask=m, other=0), mask=m)
    seq = tl.load(seq_ptr + t).to(tl.int64)
    rslot = tl.load(rslot_ptr + t).to(tl.int64)
    valid = (t < t_real) & (rslot >= 0) & (seq > 0)
    p = tl.where(valid, seq - 1, 0)
    tl.store(pos_ptr + t, p)
    row_base = bt_in_ptr + t64 * bt_in_stride
    if KIND == 0:
        # SWA: slot = runner slot; window rows of positions p-127..p (ascending), -1 before 0 / for invalid tokens
        tl.store(slot_ptr + t, tl.where(valid, rslot, -1))
        k = tl.arange(0, W)
        q = p - (W - 1) + k
        ok = valid & (q >= 0)
        qq = tl.where(ok, q, 0)
        blk = tl.load(row_base + qq // block_size, mask=ok, other=0).to(tl.int64)
        tl.store(win_ptr + t64 * W + k, tl.where(ok, blk * block_size + qq % block_size, -1))
    elif KIND == 1:
        # compressed: entry j = p // r completes at (p + 1) % r == 0; slot / latent slot -1 otherwise
        completes = valid & ((p + 1) % RATIO == 0)
        j = p // RATIO
        blk = tl.load(row_base + j // storage, mask=completes, other=0).to(tl.int64)
        cslot = tl.where(completes, blk * storage + j % storage, -1)
        tl.store(slot_ptr + t, cslot)                                   # slot_mapping == latent_slots
        tl.store(aux_ptr + t, tl.where(completes, cslot, -1))           # latent_slots
        tl.store(aux2_ptr + t, tl.maximum(p - (RATIO - 1), 0))          # latent_pos (first token of the group)
        tl.store(aux3_ptr + t, tl.where(valid, (p + 1) // RATIO, 0))    # num_visible
    else:
        # compressor state (ratio-2 pairing): own row at the runner slot; row of p-1 when p completes a pair
        tl.store(slot_ptr + t, tl.where(valid, rslot, -1))
        completes = valid & ((p + 1) % 2 == 0)
        pm1 = tl.where(completes, p - 1, 0)
        blk = tl.load(row_base + pm1 // block_size, mask=completes, other=0).to(tl.int64)
        tl.store(aux_ptr + t, tl.where(completes, blk * block_size + pm1 % block_size, -1))


class DecodeBuffers:
    """Builder-owned decode metadata (capacity ``cap`` tokens = requests)."""

    def __init__(self, kind: int, cap: int, max_blocks: int, device: torch.device) -> None:
        self.kind = kind
        self.cap = cap
        self.max_blocks = max_blocks
        i64 = dict(dtype=torch.int64, device=device)
        self.positions = torch.zeros(cap, **i64)
        self.slot_mapping = torch.full((cap,), -1, **i64)
        self.block_table = torch.zeros((cap, max_blocks), dtype=torch.int32, device=device)
        self.tok2req = torch.arange(cap, dtype=torch.int32, device=device)
        self.token_idx = torch.arange(cap, **i64)
        self.query_start_loc = torch.arange(cap + 1, dtype=torch.int32, device=device)
        self.window_slots = torch.full((cap, WINDOW), -1, **i64) if kind == KIND_SWA else None
        self.aux = torch.full((cap,), -1, **i64)           # latent_slots (compressed) / prev_slot (state)
        self.aux2 = torch.zeros(cap, **i64)                # latent_pos (compressed)
        self.aux3 = torch.zeros(cap, **i64)                # num_visible (compressed)

    def fill(self, T: int, t_real: int, seq_lens: torch.Tensor, runner_slots: torch.Tensor,
             block_table: torch.Tensor, block_size: int, storage: int, ratio: int) -> None:
        if T > self.cap:
            raise RuntimeError(f"decode batch of {T} tokens exceeds the decode-metadata capacity {self.cap} "
                               "(max_num_seqs)")
        if block_table.shape[0] < T or seq_lens.shape[0] < T or runner_slots.shape[0] < T:
            raise RuntimeError(f"decode metadata: runner tensors cover {block_table.shape[0]} requests / "
                               f"{seq_lens.shape[0]} seq_lens / {runner_slots.shape[0]} slots for {T} tokens")
        width = block_table.shape[1]
        if width > self.max_blocks:
            raise RuntimeError(f"decode metadata: block table width {width} > capacity {self.max_blocks}")
        if block_table.stride(1) != 1:
            raise RuntimeError("decode metadata: block table must be row-major")
        win = self.window_slots if self.window_slots is not None else self.aux
        _decode_md_kernel[(T,)](
            seq_lens, runner_slots, block_table, block_table.stride(0), width,
            self.block_table, self.block_table.stride(0), self.positions, self.slot_mapping,
            self.aux, self.aux2, self.aux3, win, t_real, block_size, storage,
            KIND=self.kind, RATIO=max(1, ratio), W=WINDOW, COPY_BLOCK=256, num_warps=1)
