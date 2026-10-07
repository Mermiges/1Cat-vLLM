# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 SM70 sparse MLA: KV-cache specs, cache layers, backends + metadata builders, and the
sparse attention impl over FP16 records (PORT_DESIGN §3.3, A3; owner L-ATTN).

Cache layers per attention layer ``P = model.layers.{i}.attn`` (§3.3); ``B`` = ``cache_config.block_size``
(the SWA backend prefers 256):

=====================================  ==============================================  ===========
name                                   spec                                            row
=====================================  ==============================================  ===========
``P.swa_cache`` (every layer)          SlidingWindowMLASpec(B/2 tok/block, window 128)  512 x fp16
``P`` (kv sources + mirror)            MLAAttentionSpec(B tok/block, ratio r)          512 x fp16
``P.indexer.k_cache`` (idem)           MLAAttentionSpec(B tok/block, ratio r)          128 x fp16
``P.compressor.state_cache`` (r = 2)   SlidingWindowMLASpec(B/8 tok/block, window 2)   1024 x fp32
=====================================  ==============================================  ===========

Block sizes: every block id of vLLM's single pool reserves one page in every tensor of the stage
(``kv_cache_utils._deepseek_v41_tensor_plan``), so the windowed groups use the largest blocks whose
page equals the ratio-2 compressed page (B/2 x 1 KiB = B/8 x 4 KiB = 128 KiB at B = 256): fewer
block ids per prefill chunk and no extra tensor on stages holding ratio-2 sources. Sweep
(``scratch/ds41/attn/kv_sweep.py``, 4096-token chunks, 256K context): PP3 pool 509 / 764 / 626 MiB per
stage vs 928 / 1161 / 571 MiB with 64-token SWA and 8-token state blocks.
"""

from __future__ import annotations

import torch

from vllm.models.deepseek_v41.common.contracts import (
    CKV_CACHE_DTYPE_STR,
    CKV_RECORD_DIM,
    IK_CACHE_DTYPE_STR,
    IK_RECORD_DIM,
    KV_MODEL_VERSION,
    KV_RECORD_DTYPE,
    SWA_RECORD_DIM,
    WINDOW,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec

STATE_ROW_DIM = 2 * CKV_RECORD_DIM   # FP32 [kv 512 | score 512] of one token (ratio-2 pairing)
STATE_WINDOW = 2
PREFERRED_BLOCK_SIZE = 256     # MLA (compressed / index-K) tokens per block


def swa_block_size(block_size: int) -> int:
    _check_block(block_size)
    return block_size // 2


def state_block_size(block_size: int) -> int:
    _check_block(block_size)
    return block_size // 8


def swa_cache_spec(block_size: int) -> SlidingWindowMLASpec:
    return SlidingWindowMLASpec(
        block_size=swa_block_size(block_size), num_kv_heads=1, head_size=SWA_RECORD_DIM, dtype=KV_RECORD_DTYPE,
        sliding_window=WINDOW, compress_ratio=1, model_version=KV_MODEL_VERSION)


def compressed_cache_spec(block_size: int, compress_ratio: int) -> MLAAttentionSpec:
    _check_mla_block(block_size, compress_ratio)
    return MLAAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=CKV_RECORD_DIM, dtype=KV_RECORD_DTYPE,
        compress_ratio=compress_ratio, cache_dtype_str=CKV_CACHE_DTYPE_STR, model_version=KV_MODEL_VERSION)


def index_k_cache_spec(block_size: int, compress_ratio: int) -> MLAAttentionSpec:
    _check_mla_block(block_size, compress_ratio)
    return MLAAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=IK_RECORD_DIM, dtype=KV_RECORD_DTYPE,
        compress_ratio=compress_ratio, cache_dtype_str=IK_CACHE_DTYPE_STR, model_version=KV_MODEL_VERSION)


def state_cache_spec(block_size: int) -> SlidingWindowMLASpec:
    return SlidingWindowMLASpec(
        block_size=state_block_size(block_size), num_kv_heads=1, head_size=STATE_ROW_DIM, dtype=torch.float32,
        sliding_window=STATE_WINDOW, compress_ratio=1, model_version=KV_MODEL_VERSION)


def _check_block(block_size: int) -> None:
    if block_size < 64 or block_size % 8:
        raise ValueError(f"DeepSeek-V4.1 needs cache block_size >= 64 and a multiple of 8, got {block_size}")


def _check_mla_block(block_size: int, compress_ratio: int) -> None:
    _check_block(block_size)
    if compress_ratio not in (1, 2):
        raise ValueError(f"DeepSeek-V4.1 compressed caches have ratio 1 or 2, got {compress_ratio}")
    if block_size <= 0 or block_size % compress_ratio:
        raise ValueError(f"block_size {block_size} must be a positive multiple of compress_ratio {compress_ratio}")
