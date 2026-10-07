# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 DSpark: final-stage-local, three shifted-HC blocks, noncausal N=5.

Uses the V1 DSpark proposer and the existing V4.1 projection/MoE factories.
Draft cache contains committed main_x KVs and temporary query KVs. Only the
latest accepted context is inserted on the next round; rejected query rows
are outside that round's readable window. Synchronous/eager bring-up only.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import ClassVar, Protocol

import numpy as np
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.models.deepseek_v41 import knobs
from vllm.models.deepseek_v41.attention import DeepseekV41Attention
from vllm.models.deepseek_v41.common import contracts as C
from vllm.models.deepseek_v41.common.hc import (
    hc_expand,
    hc_mixes,
    hc_post,
    hc_pre,
    rmsnorm_to_act,
)
from vllm.models.deepseek_v41.common.topology import layer_topology
from vllm.models.deepseek_v41.quant_config import v41_linear
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec

from .model import (
    _STACKED,
    _STACKED_SHARDS,
    DeepseekV41ForCausalLM,
    DeepseekV41Norm,
    map_checkpoint_name,
)
from .sparse import (
    DeepseekV41SM70SparseImpl,
    DS41SWABackend,
    DS41SWAMetadata,
    DS41SWAMetadataBuilder,
    rows_of_positions,
)

TARGETS = (37, 38, 39)
BLOCK = 5
EXPERTS = 128
TOP_K = 3
_MAIN_SCALE = 2.0**-6
_MTP = re.compile(r"^mtp\.(\d+)\.(.+)$")


def capture_target_input(
    stream: torch.Tensor,
    pending: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize the shifted FFN post before averaging a block INPUT.

    Mean rounds to BF16 like official h.mean(dim=2), then widens to FP32:
    the BOS stream exceeds FP16 range, so the aux transport must not be half.
    """
    if pending is not None:
        from .hc_kernels import hc_fused_step

        stream = hc_fused_step(
            stream, sub_out=pending[0], post=pending[1], comb=pending[2]
        )[0]
    return stream, stream.float().mean(dim=1).to(torch.bfloat16).float()


class TargetCapture(Protocol):
    pp_is_last: bool
    dspark_aux_layers: tuple[int, ...]

    def owns_layer(self, layer_id: int) -> bool: ...


def configure_target_capture(model: TargetCapture, layers: tuple[int, ...]) -> None:
    # V1 calls the Eagle interface with layer-id + 1; V4.1 DSpark captures inputs.
    if tuple(layers) != tuple(i + 1 for i in TARGETS):
        raise ValueError(f"V4.1 DSpark aux layers must be (38, 39, 40), got {layers}")
    if model.pp_is_last and not all(model.owns_layer(i) for i in TARGETS):
        raise ValueError("V4.1 DSpark targets 37-39 must all be on the final stage")
    model.dspark_aux_layers = TARGETS if model.pp_is_last else ()


@dataclass
class DSparkMetadata(DS41SWAMetadata):
    causal: bool = False


class DSparkMetadataBuilder(DS41SWAMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    @classmethod
    def get_cudagraph_support(
        cls, vllm_config: VllmConfig, kv_cache_spec: AttentionSpec
    ) -> AttentionCGSupport:
        return AttentionCGSupport.NEVER

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DSparkMetadata:
        cm = common_attn_metadata
        if cm.causal:
            raise ValueError("V4.1 DSpark requires noncausal draft metadata")
        # DFlash's CPU upper bound includes rejected tokens. Synchronous draft
        # lengths must come from the corrected GPU seq_lens, never that bound.
        exact = cm.seq_lens.detach().cpu()
        c = self._common(
            replace(cm, seq_lens_cpu_upper_bound=exact, _seq_lens_cpu=exact)
        )
        qsl = c["query_start_loc_cpu"]
        lens = np.diff(qsl)
        if any(n != BLOCK for n in lens):
            raise ValueError(f"V4.1 DSpark needs five queries/request, got {lens}")
        starts = c["seq_lens_cpu"] - lens
        anchors = torch.as_tensor(starts, device=self.device, dtype=torch.int64)
        first = anchors[c["token_to_req_indices"].long()]
        # All five queries see the SAME 128 committed rows plus all five query
        # rows. A causal SWA window here silently destroys DSpark acceptance.
        offsets = torch.arange(-C.WINDOW, BLOCK, device=self.device)
        pos = first[:, None] + offsets[None, :]
        real = c["slot_mapping"] >= 0
        pos = torch.where(real[:, None], pos, torch.full_like(pos, -1))
        slots = rows_of_positions(
            c["block_table"], c["token_to_req_indices"], pos, c["block_size"]
        )
        return DSparkMetadata(**c, window_slots=slots)


class DSparkBackend(DS41SWABackend):
    @staticmethod
    def get_name() -> str:
        return "DS41_DSPARK_SWA"

    @staticmethod
    def get_builder_cls() -> type[DSparkMetadataBuilder]:
        return DSparkMetadataBuilder


class DSparkAttention(DeepseekV41Attention):
    """V4.1 SWA projections; DSpark owns the noncausal cache read/write path."""

    def __init__(
        self,
        vc: VllmConfig,
        prefix: str,
        stage: C.StagePlan,
        shared: C.SharedAttnBuffers,
        layer_id: int,
    ) -> None:
        super().__init__(
            vc,
            prefix,
            layer_topology(vc.model_config.hf_config, layer_id),
            stage,
            shared,
        )
        self.swa_cache._backend = DSparkBackend
        # Cache eviction sees the end of the five-query block. Retain the
        # anchor's 128-row context too (a 128-token spec loses rows at page edges).
        self.swa_cache._spec = replace(
            self.swa_cache._spec, sliding_window=C.WINDOW + BLOCK
        )
        # The synchronous drafter executes directly; no target opaque-op entry.
        vc.compilation_config.static_forward_context.pop(self.layer_name)

    def _qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        qrkv = v41_linear(self.fused_wqa_wkv, x, torch.float32)
        qr = self.q_norm(qrkv[:, : C.Q_LORA]).half()
        q = v41_linear(self.wq_b, qr, torch.float16).view(
            -1, self.n_local_heads, C.HEAD_DIM
        )
        return q, self.kv_norm(qrkv[:, C.Q_LORA :])

    def store_context(
        self, main_x: torch.Tensor, positions: torch.Tensor, slots: torch.Tensor | None
    ) -> None:
        from .q_rope_kv_insert import q_rope_kv_insert

        # main_x is shared by every draft block (not its attention norm input).
        qrkv = v41_linear(self.fused_wqa_wkv, main_x, torch.float32)
        kv = self.kv_norm(qrkv[:, C.Q_LORA :]).contiguous()
        if slots is None:  # memory profiler: compute projections, no real cache
            return
        dummy = torch.zeros(
            (main_x.shape[0], self.n_local_heads, C.HEAD_DIM),
            device=main_x.device,
            dtype=torch.float16,
        )
        q_rope_kv_insert(
            dummy,
            kv,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.swa_cache.rows(),
            slots,
        )

    def forward(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        from .q_rope_kv_insert import q_rope_kv_insert

        ctx = get_forward_context()
        if ctx.attn_metadata is None and not ctx.is_dummy_run:
            raise RuntimeError("DSpark attention metadata missing outside a dummy run")
        q, kv = self._qkv(x)
        o = torch.empty_like(q)
        if ctx.attn_metadata is None:
            DeepseekV41SM70SparseImpl.reserve_workspace(o, False)
            o.zero_()
        else:
            md = ctx.attn_metadata[self.swa_cache.layer_name]
            if not isinstance(md, DSparkMetadata) or md.causal:
                raise TypeError("V4.1 DSpark needs its noncausal metadata backend")
            t = md.num_actual_tokens
            q_rope_kv_insert(
                q[:t],
                kv[:t].contiguous(),
                positions[:t],
                self.rotary_emb.cos_sin_cache,
                self.swa_cache.rows(),
                md.slot_mapping,
            )
            DeepseekV41SM70SparseImpl.forward(
                q[:t],
                md,
                self.swa_cache.rows(),
                None,
                None,
                self.attn_sink,
                self.scale,
                o[:t],
            )
            o[t:].zero_()
        return self.output_projection(o, positions)


class DSparkBlock(nn.Module):
    def __init__(
        self,
        vc: VllmConfig,
        prefix: str,
        layer_id: int,
        stage: C.StagePlan,
        shared: C.SharedAttnBuffers,
    ) -> None:
        super().__init__()
        from .moe import DeepseekV41MoE

        self.hc_fused = knobs.env_bool("VLLM_DS41_HC_FUSED", True)
        self.attn = DSparkAttention(vc, f"{prefix}.attn", stage, shared, layer_id)
        self.ffn = DeepseekV41MoE(
            vc,
            f"{prefix}.ffn",
            layer_id,
            n_routed_experts=EXPERTS,
            top_k=TOP_K,
            spill=None,
        )
        self.attn_norm = DeepseekV41Norm(C.HIDDEN)
        self.ffn_norm = DeepseekV41Norm(C.HIDDEN)
        for sub in ("attn", "ffn"):
            for suffix, shape in (
                ("fn", (24, C.HC * C.HIDDEN)),
                ("base", (24,)),
                ("scale", (3,)),
            ):
                self.register_parameter(
                    f"hc_{sub}_{suffix}",
                    nn.Parameter(
                        torch.empty(shape, dtype=torch.float32), requires_grad=False
                    ),
                )

    def forward(
        self, stream: torch.Tensor, pre: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.hc_fused:
            return self.forward_fused(stream, pre, positions)
        ap, post, comb = hc_mixes(
            stream, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = rmsnorm_to_act(hc_pre(stream, pre), self.attn_norm.weight)
        stream = hc_post(self.attn(positions, x), stream, post, comb)
        fp, post, comb = hc_mixes(
            stream, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = rmsnorm_to_act(hc_pre(stream, ap), self.ffn_norm.weight)
        return hc_post(self.ffn(x), stream, post, comb), fp

    def forward_fused(
        self, stream: torch.Tensor, pre: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .hc_kernels import hc_fused_step

        stream, mixes, x, _ = hc_fused_step(
            stream,
            mix=(self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base),
            collapse_pre=pre,
            norm_weight=self.attn_norm.weight,
        )
        assert mixes is not None and x is not None
        ap, post, comb = mixes
        stream, mixes, x, _ = hc_fused_step(
            stream,
            sub_out=self.attn(positions, x).contiguous(),
            post=post,
            comb=comb,
            mix=(self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base),
            collapse_pre=ap,
            norm_weight=self.ffn_norm.weight,
        )
        assert mixes is not None and x is not None
        fp, post, comb = mixes
        return hc_fused_step(
            stream, sub_out=self.ffn(x).contiguous(), post=post, comb=comb
        )[0], fp


class DSparkModel(nn.Module):
    def __init__(self, vc: VllmConfig, prefix: str) -> None:
        super().__init__()
        hf = vc.model_config.hf_config
        if (
            tuple(hf.dspark_target_layer_ids),
            hf.dspark_block_size,
            hf.dspark_noise_token_id,
            hf.dspark_markov_rank,
            hf.dspark_n_routed_experts,
            hf.dspark_num_experts_per_tok,
            hf.num_nextn_predict_layers,
        ) != (TARGETS, BLOCK, 128799, 256, EXPERTS, TOP_K, 3):
            raise ValueError("V4.1 DSpark checkpoint does not match the draft contract")
        if vc.scheduler_config.async_scheduling:
            raise NotImplementedError("V4.1 DSpark bring-up requires sync scheduling")
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        stage = C.StagePlan(0, 1, 40, 42, (), (), ())
        shared = C.allocate_shared_attn_buffers(
            vc.scheduler_config.max_num_batched_tokens,
            stage,
            torch.get_default_device(),
        )
        self.embed_tokens = VocabParallelEmbedding(
            hf.vocab_size, C.HIDDEN, prefix=f"{prefix}.embed_tokens"
        )
        self.main_proj = ReplicatedLinear(
            3 * C.HIDDEN,
            C.HIDDEN,
            bias=False,
            return_bias=False,
            quant_config=vc.quant_config,
            prefix=f"{prefix}.main_proj",
        )
        self.main_norm = DeepseekV41Norm(C.HIDDEN)
        self.layers = nn.ModuleList(
            [
                DSparkBlock(vc, f"{prefix}.layers.{i}", 40 + i, stage, shared)
                for i in range(3)
            ]
        )
        self.norm = DeepseekV41Norm(C.HIDDEN)
        # Replicate these small heads to avoid five TP collectives per round.
        self.markov_embed = nn.Embedding(hf.vocab_size, 256, dtype=torch.float16)
        self.markov_weight = nn.Parameter(
            torch.empty(hf.vocab_size, 256, dtype=torch.float16), requires_grad=False
        )
        self.confidence_weight = nn.Parameter(
            torch.empty(1, C.HIDDEN + 256, dtype=torch.float32), requires_grad=False
        )

    def combine_hidden_states(self, aux: torch.Tensor) -> torch.Tensor:
        if aux.dtype == torch.float16:
            raise TypeError(
                "V4.1 DSpark aux input must preserve BF16 range (FP32 transport)"
            )
        # Bias by a power of two BEFORE half conversion; undo in FP32 before
        # normalization so the epsilon and zero-row semantics are unchanged.
        x = (aux.float() * _MAIN_SCALE).half()
        projected = v41_linear(self.main_proj, x, torch.float32) / _MAIN_SCALE
        return rmsnorm_to_act(projected, self.main_norm.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embed = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        stream, pre = hc_expand(embed)
        for layer in self.layers:
            stream, pre = layer(stream, pre, positions)
        return hc_pre(stream, pre)  # confidence consumes this BEFORE final norm


class DSparkDeepseekV41ForCausalLM(nn.Module):
    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = DSparkModel(vllm_config, f"{prefix}.model".lstrip("."))
        self.lm_head = ParallelLMHead(
            self.config.vocab_size, C.HIDDEN, prefix=f"{prefix}.lm_head".lstrip(".")
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(aux_hidden_states)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.attn.swa_cache.layer_name for layer in self.model.layers]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        for layer in self.model.layers:
            name = layer.attn.swa_cache.layer_name
            if context_slot_mappings is not None and name not in context_slot_mappings:
                raise KeyError(f"DSpark missing context slot mapping for {name}")
            layer.attn.store_context(
                context_states,
                context_positions,
                None if context_slot_mappings is None else context_slot_mappings[name],
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    @staticmethod
    def _linear_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return torch.mm(x.half(), weight.half().t(), out_dtype=torch.float32)
        return x.float() @ weight.float().t()

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = rmsnorm_to_act(hidden_states, self.model.norm.weight)
        logits = self._linear_fp32(x, self.lm_head.weight)
        if get_tensor_model_parallel_world_size() > 1:
            logits = tensor_model_parallel_all_gather(logits, dim=-1)
        return logits[:, : self.config.vocab_size]

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self._linear_fp32(markov_embed, self.model.markov_weight)

    def confidence_logits(
        self, hidden_states: torch.Tensor, markov_embeds: torch.Tensor
    ) -> torch.Tensor:
        features = torch.cat((hidden_states.float(), markov_embeds.float()), dim=-1)
        shape = features.shape[:-1]
        return (
            features.reshape(-1, features.shape[-1]) @ self.model.confidence_weight.t()
        ).view(shape)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    @staticmethod
    def _remap_dspark_name(name: str) -> str | None:
        if name in ("embed.weight", "head.weight"):
            return (
                "model.embed_tokens.weight"
                if name == "embed.weight"
                else "lm_head.weight"
            )
        match = _MTP.fullmatch(name)
        if match is None:
            return None
        i, rest = int(match[1]), match[2]
        if not 0 <= i < 3:
            raise KeyError(f"unexpected DSpark stage in {name}")
        if rest.endswith(".bias_vl"):
            return None  # documented text-only gate bias exclusion
        special = {
            "markov_head.embed.weight": "markov_embed.weight",
            "markov_head.head.weight": "markov_weight",
            "confidence_head.proj.weight": "confidence_weight",
            "norm.weight": "norm.weight",
        }
        if rest in special:
            if i != 2:
                raise KeyError(f"final draft head on wrong stage: {name}")
            return "model." + special[rest]
        if rest.startswith(("main_proj.", "main_norm.")):
            if i != 0:
                raise KeyError(f"main projection on wrong draft stage: {name}")
            if rest.endswith(".scale"):
                rest = rest.removesuffix(".scale") + ".weight_scale_inv"
            return "model." + rest
        # Reuse the backbone's exact dense/expert/bias name conversion.
        return map_checkpoint_name(f"layers.{i}.{rest}")

    def skip_checkpoint_weight(self, name: str) -> bool:
        return self._remap_dspark_name(name) is None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from .moe import DeepseekV41MoE

        params = dict(self.named_parameters())
        mapping: dict[str, tuple[str, int, str]] = {}
        keys: dict[str, set[object]] = {}
        for param, ckpt, e, shard in DeepseekV41MoE.expert_params_mapping(EXPERTS):
            mapping[ckpt] = (param, e, shard)
            keys.setdefault(param, set()).add((e, shard))
        loaded: set[str] = set()
        parts: dict[str, set[object]] = {}
        expected: dict[str, set[object]] = {}
        seen: set[str] = set()
        for ckpt, tensor in weights:
            name = self._remap_dspark_name(ckpt)
            if name is None:
                continue
            if ckpt in seen:
                raise ValueError(f"duplicate DSpark checkpoint tensor {ckpt}")
            seen.add(ckpt)
            if ".ffn.experts." in name:
                mapped, frag, key = DeepseekV41ForCausalLM._load_expert(
                    params, name, tensor, mapping
                )
                loaded.add(mapped)
                parts.setdefault(mapped, set()).add(key)
                expected[mapped] = keys[frag]
                continue
            for param_frag, ckpt_frag, shard in _STACKED:
                if ckpt_frag in name:
                    mapped = name.replace(ckpt_frag, param_frag)
                    if mapped not in params:
                        raise KeyError(f"DSpark tensor {ckpt} maps to absent {mapped}")
                    params[mapped].weight_loader(params[mapped], tensor, shard)
                    loaded.add(mapped)
                    parts.setdefault(mapped, set()).add(shard)
                    expected[mapped] = set(_STACKED_SHARDS[param_frag])
                    break
            else:
                if name not in params:
                    raise KeyError(f"DSpark tensor {ckpt} maps to absent {name}")
                p = params[name]
                getattr(p, "weight_loader", default_weight_loader)(p, tensor)
                loaded.add(name)
        # Same strict coverage and per-expert/fused shard checks as the target.
        DeepseekV41ForCausalLM._check_complete(
            self, loaded, {name: expected[name] - got for name, got in parts.items()}
        )
        return loaded

    _describe_missing_parts = staticmethod(
        DeepseekV41ForCausalLM._describe_missing_parts
    )
