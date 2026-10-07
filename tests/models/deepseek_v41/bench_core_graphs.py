# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded four-layer core graph benchmark; never an end-to-end throughput claim."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from pytest import MonkeyPatch

from tests.models.deepseek_v41.test_core_graph_replay import subset
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import set_forward_context


def measure(fn, steps: int = 50) -> dict[str, float]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        fn()
    issue = time.perf_counter() - t0
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
    events = prof.events()
    kernels = [e for e in events if e.device_type == torch.autograd.DeviceType.CUDA]
    launches = [e for e in events if e.name.startswith("cudaLaunch")]
    return {
        "issue_us_per_step": issue * 1e6 / steps,
        "wall_us_per_step": wall * 1e6 / steps,
        "device_events_per_step": len(kernels) / 5,
        "host_launch_calls_per_step": len(launches) / 5,
    }


def main() -> None:
    with MonkeyPatch.context() as patch, torch.inference_mode():
        run, current, _ = subset(patch)
        vc = VllmConfig()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            with set_forward_context(
                None,
                vc,
                num_tokens=1,
                is_dummy_run=True,
                cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            ):
                for _ in range(5):
                    run()
                stream.synchronize()
                cap = BreakableCUDAGraphCapture()
                with cap:
                    run()
            current[0] = 7
            with set_forward_context(
                None, vc, num_tokens=1, cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE
            ):
                for _ in range(5):
                    cap.replay()
                eager = measure(run)
                replay = measure(cap.replay)
        result = {
            "scope": "4 layers, real fused HC; stand-in attention/MoE/row store",
            "eager_breaks": cap.num_eager_breaks,
            "graph_segments": cap.num_graphs,
            "eager": eager,
            "replay": replay,
        }
        dest = Path("/mnt/nvme2/scratch/ds41/sol/core/graphs_bench.json")
        dest.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
