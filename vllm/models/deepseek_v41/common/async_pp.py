# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in async PP token handoff. Each D2H owns its pinned buffer until consumed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.models.deepseek_v41.knobs import env_bool

ASYNC_PP_KNOB = "VLLM_DS41_CORE_ASYNC_PP"

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def require_async_pp(config: VllmConfig) -> None:
    if not env_bool(ASYNC_PP_KNOB, False):
        raise NotImplementedError(
            "DeepSeek-V4.1 Engram needs synchronous scheduling unless "
            f"{ASYNC_PP_KNOB}=1 (PORT_DESIGN A9)"
        )
    if config.parallel_config.pipeline_parallel_size <= 1:
        raise NotImplementedError("DeepSeek-V4.1 async Engram requires PP > 1")
    if config.speculative_config is not None:
        raise NotImplementedError(
            "DeepSeek-V4.1 async PP Engram refuses speculative decoding"
        )


def pinned_sampled_fill(ids: torch.Tensor) -> tuple[torch.Tensor, torch.cuda.Event]:
    if ids.dtype != torch.int32 or ids.device.type != "cuda":
        raise ValueError("Engram sampled ids must be CUDA int32")
    fill = torch.empty(ids.numel(), dtype=torch.int32, device="cpu", pin_memory=True)
    fill.copy_(ids.reshape(-1), non_blocking=True)
    ready = torch.cuda.Event()
    ready.record()
    return fill, ready


@dataclass(frozen=True)
class SampledPPIds:
    ids: torch.Tensor
    req_order: tuple[str, ...]
    fill: tuple[torch.Tensor, torch.cuda.Event]

    @classmethod
    def receive(cls, ids: torch.Tensor, req_order: tuple[str, ...]) -> SampledPPIds:
        if ids.shape != (len(req_order), 1):
            raise ValueError("Engram async PP expects one sampled id per request")
        return cls(ids, req_order, pinned_sampled_fill(ids))

    def for_batch(
        self, req_order: tuple[str, ...], required: frozenset[str]
    ) -> tuple[torch.Tensor, torch.cuda.Event]:
        index = {rid: i for i, rid in enumerate(self.req_order)}
        missing = required - index.keys()
        if missing:
            raise RuntimeError(
                f"Engram async PP has no sampled id for {sorted(missing)}"
            )
        if req_order == self.req_order:
            return self.fill
        # Reorder on device before D2H: reading the first pinned copy here
        # would race its event. Zero entries for new prompt requests are unused.
        gather = torch.tensor(
            [index.get(rid, -1) for rid in req_order],
            dtype=torch.long,
            device=self.ids.device,
        )
        if not self.req_order:
            return pinned_sampled_fill(
                torch.zeros(len(req_order), dtype=torch.int32, device=self.ids.device)
            )
        ids = self.ids.reshape(-1)[gather.clamp_min(0)]
        ids.masked_fill_(gather < 0, 0)
        return pinned_sampled_fill(ids)
