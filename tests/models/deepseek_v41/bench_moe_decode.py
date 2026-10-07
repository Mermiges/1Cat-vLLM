# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Board B (180 W cap) fused decode A/B, graph timings with L2-busting weights."""

from __future__ import annotations

import argparse
import json
import time
from itertools import cycle
from pathlib import Path

import torch

import vllm._custom_ops as ops
from vllm.models.deepseek_v41.sm70 import gemv
from vllm.models.deepseek_v41.sm70 import moe_kernels as mk
from vllm.models.deepseek_v41.sm70.moe_decode import DecodeScratch, decode_front
from vllm.models.deepseek_v41.sm70.moe_kernels import RouteTables

from .test_moe_bench import graph_us
from .test_moe_method import SyntheticCheckpoint, make_layer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    layer, method = make_layer(
        384, 6, 4, 0, "skinny", None, SyntheticCheckpoint(seed=5)
    )
    x = torch.randn(1, 5120, device="cuda").half()
    gates = [torch.randn(384, 5120, device="cuda").half() * 0.02 for _ in range(4)]
    shared = [torch.randn(1152, 5120, device="cuda").half() * 0.01 for _ in range(4)]
    down = [torch.randn(5120, 576, device="cuda").half() * 0.01 for _ in range(4)]
    bias = torch.full((384,), 10.8, device="cuda")
    scratch = DecodeScratch.allocate(384, x.device)
    rotation = cycle(range(4))

    def front(
        fused: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, RouteTables, torch.Tensor]:
        idx = next(rotation)
        if fused:
            return decode_front(
                x,
                gates[idx],
                bias,
                shared[idx],
                scratch,
                top_k=6,
                alpha=1.0,
                scale=1.5,
                limit=10.0,
                phys_map=None,
                n_resident=384,
                spill=False,
            )
        logits = gemv.gemv(x, gates[idx])
        weights = torch.empty(1, 6, device="cuda")
        ids = torch.empty(1, 6, device="cuda", dtype=torch.int32)
        ops.topk_hash_softplus_sqrt(
            weights, ids, torch.empty_like(ids), logits, True, 1.5, bias, None, None
        )
        tables = mk.route_prep(ids, None, 384, False)
        return weights, ids, tables, gemv.gate_up_swiglu(x, shared[idx], 10.0)

    def block(fused: bool) -> torch.Tensor:
        weights, ids, tables, act = front(fused)
        y = method.expert_slots(layer, x, ids, tables)
        return gemv.down_combine(act, down[next(rotation)], y, weights)

    def eager_us(fused: bool) -> float:
        for _ in range(5):
            block(fused)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            block(fused)
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1e4

    report = {
        "board": "B",
        "power_cap_W": 180,
        "device": torch.cuda.get_device_name(),
        "T": 1,
        "TP_local_shape": 4,
        "baseline_kernels": 8,
        "fused_kernels": 5,
        "front_baseline_us": graph_us(lambda: front(False)),
        "front_fused_us": graph_us(lambda: front(True)),
        "block_baseline_us": graph_us(lambda: block(False)),
        "block_fused_us": graph_us(lambda: block(True)),
        "block_eager_baseline_us": eager_us(False),
        "block_eager_fused_us": eager_us(True),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
