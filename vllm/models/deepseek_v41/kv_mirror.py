# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-local replica of kv source 20 for PP3 (PORT_DESIGN A7, §3.3, §3.5; owner L-ATTN, consumer half).

Stage 3 ([28-39]) reads source 20's compressed KV and index-K records (Reuse/Reindex layers) and its
candidate blocks (Reindex 28/32/36), but layer 20 runs on stage 2. The mirror registers two cache layers
under the SOURCE's names -- ``model.layers.20.attn`` (compressed KV) and ``model.layers.20.attn.indexer.k_cache``
(index-K) -- with specs equal to the source's, so ``get_kv_cache_configs`` merges them into the source's KV
cache group: the single scheduler-side KV manager hands both stages the same block ids and slot mappings,
and prefix-cache hits / preemption act on both copies together.

Exactness: layer 20 has compress ratio 1, so it emits one latent and one index key per scheduled token, in
token order; ``shared.export_ckv[:T]`` / ``shared.export_ik[:T]`` hold the exact FP16 records the source wrote
(copied by the same kernels). ``ingest`` scatters them to the slots of this stage's metadata for the same
names and copies the candidate rows into ``shared.candidate_blocks[:T]``. Nothing is recomputed.
Debug knob ``VLLM_DS41_ATTN_MIRROR_CHECK=1`` re-reads the written rows and fails loudly on any difference.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.models.deepseek_v41 import knobs
from vllm.models.deepseek_v41.common.contracts import (
    CAND_SOURCE,
    CAND_TOPK_BLOCKS,
    CKV_RECORD_DIM,
    IK_RECORD_DIM,
    KV_RECORD_DTYPE,
    SharedAttnBuffers,
)
from vllm.models.deepseek_v41.sm70.sparse import (
    DS41CacheLayer,
    DS41CompressedBackend,
    DS41CompressedMetadata,
    compressed_cache_spec,
    index_k_cache_spec,
)

MIRROR_CHECK_ENV = "VLLM_DS41_ATTN_MIRROR_CHECK"


def source_names(source_layer: int) -> tuple[str, str]:
    p = f"model.layers.{source_layer}.attn"
    return p, f"{p}.indexer.k_cache"


class DeepseekV41KVSourceMirror(nn.Module):
    def __init__(self, vllm_config: VllmConfig, source_layer: int, shared: SharedAttnBuffers) -> None:
        """Registers in the static forward context, under the SOURCE's names, two AttentionLayerBase children:
        f"model.layers.{source_layer}.attn" (compressed-KV spec) and
        f"model.layers.{source_layer}.attn.indexer.k_cache" (index-K spec), whose get_kv_cache_spec() returns
        specs equal (dataclass ==) to the source's on the earlier stage."""
        super().__init__()
        if source_layer != CAND_SOURCE:
            raise ValueError(f"v1 mirrors only kv source {CAND_SOURCE}, got {source_layer}")
        ratio = int(vllm_config.model_config.hf_config.compress_ratios[source_layer])
        if ratio != 1:
            raise ValueError(f"the mirror needs a ratio-1 source (one record per token), source {source_layer} "
                             f"has ratio {ratio}")
        block_size = vllm_config.cache_config.block_size
        self.source_layer = source_layer
        self.shared = shared
        ckv_name, ik_name = source_names(source_layer)
        self.ckv_cache = DS41CacheLayer(ckv_name, compressed_cache_spec(block_size, ratio), DS41CompressedBackend,
                                        vllm_config)
        self.ik_cache = DS41CacheLayer(ik_name, index_k_cache_spec(block_size, ratio), DS41CompressedBackend,
                                       vllm_config)
        self.check = knobs.env_bool(MIRROR_CHECK_ENV, False)
        self.layer_name = f"{ckv_name}.ds41_mirror"
        ctx = vllm_config.compilation_config.static_forward_context
        if self.layer_name in ctx:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        ctx[self.layer_name] = self

    def ingest(self, positions: torch.Tensor, ckv: torch.Tensor, ik: torch.Tensor, cand: torch.Tensor) -> None:
        """ckv [T,512] fp16, ik [T,128] fp16 (exact records as written by the source), cand [T,2048] int32.
        Scatter ckv/ik to the slots of attn_metadata[source prefix].slot_mapping; copy cand to
        shared.candidate_blocks[:T]. Eager-break op."""
        if ckv.dtype != KV_RECORD_DTYPE or ckv.shape[-1] != CKV_RECORD_DIM:
            raise ValueError(f"mirror ckv must be fp16 [T, {CKV_RECORD_DIM}], got {ckv.dtype} {tuple(ckv.shape)}")
        if ik.dtype != KV_RECORD_DTYPE or ik.shape[-1] != IK_RECORD_DIM:
            raise ValueError(f"mirror ik must be fp16 [T, {IK_RECORD_DIM}], got {ik.dtype} {tuple(ik.shape)}")
        if cand.dtype != torch.int32 or cand.shape[-1] != CAND_TOPK_BLOCKS:
            raise ValueError(f"mirror cand must be int32 [T, {CAND_TOPK_BLOCKS}], got {cand.dtype} {tuple(cand.shape)}")
        ds41_mirror_ingest(positions, ckv, ik, cand, self.layer_name)

    def ingest_impl(self, positions: torch.Tensor, ckv: torch.Tensor, ik: torch.Tensor, cand: torch.Tensor) -> None:
        md_all = get_forward_context().attn_metadata
        if not isinstance(md_all, dict):
            return                                    # profile / dummy run: nothing to replicate
        m_ckv = cast(DS41CompressedMetadata, md_all[self.ckv_cache.layer_name])
        m_ik = cast(DS41CompressedMetadata, md_all[self.ik_cache.layer_name])
        n = m_ckv.num_latents
        if m_ik.num_latents != n or m_ckv.compress_ratio != 1:
            raise RuntimeError(f"mirror metadata mismatch: ckv {n} latents (ratio {m_ckv.compress_ratio}), "
                               f"ik {m_ik.num_latents}")
        if ckv.shape[0] < n or ik.shape[0] < n or cand.shape[0] < n:
            raise RuntimeError(f"mirror payload has {ckv.shape[0]}/{ik.shape[0]}/{cand.shape[0]} rows for {n} tokens")
        ckv_rows, ik_rows = self.ckv_cache.rows(), self.ik_cache.rows()
        for rows, src, slots in ((ckv_rows, ckv, m_ckv.latent_slots), (ik_rows, ik, m_ik.latent_slots)):
            keep = slots >= 0
            rows.index_copy_(0, slots[keep], src[:n][keep])
        self.shared.candidate_blocks[:n].copy_(cand[:n])
        if self.check:
            for rows, src, slots, what in ((ckv_rows, ckv, m_ckv.latent_slots, "ckv"),
                                           (ik_rows, ik, m_ik.latent_slots, "ik")):
                keep = slots >= 0
                got = rows.index_select(0, slots[keep])
                if not torch.equal(got.view(torch.int16), src[:n][keep].view(torch.int16)):
                    raise RuntimeError(f"KV mirror {what} rows differ from the payload after ingest")


@eager_break_during_capture
def ds41_mirror_ingest(positions: torch.Tensor, ckv: torch.Tensor, ik: torch.Tensor, cand: torch.Tensor,
                       layer_name: str) -> None:
    forward_context: ForwardContext = get_forward_context()
    forward_context.no_compile_layers[layer_name].ingest_impl(positions, ckv, ik, cand)
