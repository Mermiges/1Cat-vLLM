# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated DS41 communication measurements; no model/server is launched.

Run from /tmp with PYTHONPATH pointing to the comm worktree. CUDA_VISIBLE_DEVICES
must contain the explicitly allocated board-B UUIDs. Refuses an E2E lock or busy
GPU before spawning. JSON records both host enqueue and CUDA-event latency.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

BOARD_B = (
    "GPU-0b2779f9-b525-ece6-857f-502d36299cfe",
    "GPU-6f7b0a42-1b75-3cc9-2371-e8178c7d1d22",
    "GPU-049de101-5db3-c937-2ebd-76ac1e5117d8",
    "GPU-54038fcd-3b24-141f-6799-ea800dbab677",
)


def gpu_gate(world: int) -> None:
    if Path("/mnt/nvme2/scratch/ds41/E2E-LOCK").exists():
        raise RuntimeError("E2E-LOCK present: GPU job must wait")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).splitlines()
    board = [x for x in inventory if x in BOARD_B]
    allowed = board if world == 4 else list(BOARD_B[2:])
    if len(visible) != world or set(visible) != set(allowed):
        raise RuntimeError(f"expected exactly allowed board-B UUIDs: {allowed}")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv"],
        text=True,
    )
    if any(x in active for x in visible):
        raise RuntimeError(f"allocated GPU busy; wait before retrying:\n{active}")


def _pp(
    rank: int, rows: int, mirrored: bool, grouped: str, pynccl: object
) -> dict[str, float]:
    schema = [((rows, 4, 5120), torch.bfloat16), ((rows, 4), torch.float32)]
    if mirrored:
        schema += [
            ((rows, 512), torch.float16),
            ((rows, 128), torch.float16),
            ((rows, 2048), torch.int32),
            ((rows,), torch.int32),
        ]
    tensors = [
        torch.full(s, 7 if rank == 0 else 0, device="cuda", dtype=d) for s, d in schema
    ]

    from types import SimpleNamespace

    from vllm.distributed.parallel_state import GroupCoordinator

    coordinator = GroupCoordinator.__new__(GroupCoordinator)
    coordinator.world_size = 2
    coordinator.rank_in_group = rank
    coordinator.ranks = [0, 1]
    coordinator.use_cpu_custom_send_recv = False
    coordinator.device_communicator = SimpleNamespace(pynccl_comm=pynccl)
    payload = {str(i): t for i, t in enumerate(tensors)}

    def step() -> None:
        op = dist.isend if rank == 0 else dist.irecv
        if grouped == "pynccl":
            work = (
                coordinator.isend_tensor_dict_static(payload)
                if rank == 0
                else coordinator.irecv_tensor_dict_static(payload)
            )
            for w in work:
                w.wait()
            return
        if grouped == "torch_batch":
            work = dist.batch_isend_irecv(
                [dist.P2POp(op, t, 1 - rank) for t in tensors]
            )
        else:
            work = [op(t, 1 - rank) for t in tensors]
        for w in work:
            w.wait()

    for _ in range(10):
        step()
    torch.cuda.synchronize()
    dist.barrier()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    t0 = time.perf_counter()
    for _ in range(100):
        step()
    host = (time.perf_counter() - t0) * 1e4
    end.record()
    end.synchronize()
    assert all(bool(torch.all(t == 7)) for t in tensors)
    return {"host_us": host, "gpu_us": start.elapsed_time(end) * 10}


def _ar(rank: int, world: int) -> list[dict[str, object]]:
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

    gloo = dist.new_group(backend="gloo")
    comm = CustomAllreduce(gloo, rank, max_size=2 * 1024 * 1024)
    if comm.disabled:
        raise RuntimeError("custom AR disabled on allocated board")
    results = []
    try:
        for rows in (1, 2, 4, 8):
            for width, dtype in (
                (5120, torch.float16),
                (5120, torch.float32),
                (25600, torch.float32),
            ):
                x = torch.full(
                    (rows, width), float(rank + 1), device="cuda", dtype=dtype
                )
                y = torch.empty_like(x)
                assert comm.should_custom_ar(x)
                for limit in (None, 1, 2, 4, 8, 16):
                    if limit is None:
                        os.environ.pop("VLLM_CUSTOM_ALLREDUCE_BLOCK_LIMIT", None)
                    else:
                        os.environ["VLLM_CUSTOM_ALLREDUCE_BLOCK_LIMIT"] = str(limit)
                    graph = torch.cuda.CUDAGraph()
                    torch.cuda.synchronize()
                    dist.barrier()
                    with comm.capture(), torch.cuda.graph(graph):
                        for _ in range(32):
                            comm.all_reduce(x, out=y, registered=True)
                    for _ in range(3):
                        graph.replay()
                    torch.cuda.synchronize()
                    dist.barrier()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    t0 = time.perf_counter()
                    for _ in range(20):
                        graph.replay()
                    host = (time.perf_counter() - t0) * 1e6 / 640
                    end.record()
                    end.synchronize()
                    assert bool(torch.all(y == world * (world + 1) / 2))
                    results.append(
                        {
                            "rows": rows,
                            "width": width,
                            "dtype": str(dtype),
                            "block_limit": limit,
                            "host_us": host,
                            "gpu_us": start.elapsed_time(end) * 1000 / 640,
                        }
                    )
                    del graph
    finally:
        comm.close()
    return results


def worker(rank: int, world: int, mode: str, rendezvous: str, output: str) -> None:
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=120),
    )
    try:
        if mode == "ar":
            results = _ar(rank, world)
        else:
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

            gloo = dist.new_group(backend="gloo")
            pynccl = PyNcclCommunicator(gloo, rank)
            assert not pynccl.disabled
            results = []
            for rows in (1, 4, 8):
                for mirrored in (False, True):
                    for grouped in ("separate", "torch_batch", "pynccl"):
                        result = _pp(rank, rows, mirrored, grouped, pynccl)
                        results.append(
                            {
                                "rows": rows,
                                "mirrored": mirrored,
                                "grouped": grouped,
                                **result,
                            }
                        )
        Path(f"{output}.rank{rank}.json").write_text(json.dumps(results, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("pp", "ar"))
    parser.add_argument("--world", type=int, choices=(2, 4), default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.mode == "pp" and args.world != 2:
        parser.error("PP benchmark uses exactly 2 GPUs")
    gpu_gate(args.world)
    rendezvous = Path(args.output + ".rendezvous")
    if rendezvous.exists():
        raise RuntimeError(f"use a fresh output path: {rendezvous}")
    mp.spawn(
        worker,
        args=(args.world, args.mode, rendezvous.as_uri(), args.output),
        nprocs=args.world,
        join=True,
    )
