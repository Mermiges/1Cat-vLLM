# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicitly allocated board-B gates. No implicit skips or GPU selection."""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file

from .bench_ds41_communication import gpu_gate

GOLDEN = Path("/mnt/nvme2/scratch/ds41/golden")


def _bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def _reduce_gate(comm, rank: int, world: int) -> None:
    nccl_mismatches = 0
    for rows in range(1, 9):
        for width, dtype in (
            (5120, torch.float16),
            (5120, torch.float32),
            (25600, torch.float32),
        ):
            x = torch.empty((rows, width), device="cuda", dtype=dtype)
            storage = torch.full((x.numel() + 16,), 123, device="cuda", dtype=dtype)
            y = storage[8:-8].view_as(x)
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            dist.barrier()
            with comm.capture(), torch.cuda.graph(graph):
                for _ in range(8):
                    comm.all_reduce(x, out=y, registered=True)
            for cycle in range(4):
                inputs = []
                for peer in range(world):
                    gen = torch.Generator(device="cuda").manual_seed(
                        1000 + peer + cycle * 10
                    )
                    val = torch.randn(
                        x.shape, generator=gen, device="cuda", dtype=dtype
                    )
                    if cycle == 1:
                        val.fill_((65504.0, 0.0001, -65504.0, 0.03125)[peer])
                    elif cycle == 2:
                        val.zero_()
                        val[::2].fill_(-0.0)
                    inputs.append(val)
                expected = inputs[0].float()
                for val in inputs[1:]:
                    expected = expected + val.float()
                expected = expected.to(dtype)
                x.copy_(inputs[rank])
                dist.barrier()
                if rank == cycle % world:
                    torch.cuda._sleep(10000)
                graph.replay()
                torch.cuda.synchronize()
                _bits_equal(y, expected)
                _bits_equal(comm.all_reduce(x), expected)
                assert bool(torch.all(storage[:8] == 123))
                assert bool(torch.all(storage[-8:] == 123))
                nccl = x.clone()
                dist.all_reduce(nccl)
                nccl_mismatches += int(
                    torch.count_nonzero(nccl.view(torch.uint8) != y.view(torch.uint8))
                )
            del graph
    print(
        f"rank={rank} rank-ordered FP32 sum; NCCL byte mismatches={nccl_mismatches}",
        flush=True,
    )


def _layer_goldens(comm, rank: int, world: int) -> None:
    files = sorted(GOLDEN.glob("*__v100-semantic+ownk__decode*/L*.safetensors"))
    if not files:
        raise RuntimeError("mandatory per-layer decode goldens missing")
    count, layers = 0, set()
    for path in files:
        values = load_file(str(path))
        names = [name for name in ("attn.out", "moe.out") if name in values]
        if "engram.key" in values:
            values["engram.partial"] = torch.cat(
                (values["engram.key"].flatten(1), values["engram.value"]), dim=1
            )
            names.append("engram.partial")
        for name in names:
            golden = values[name].cuda().contiguous().float()
            if not 1 <= golden.shape[0] <= 8:
                continue
            # Exact binary partition of the per-layer golden output tests
            # collective transport/rounding independently of GEMM tactics.
            x = golden / 2 if rank < 2 else torch.zeros_like(golden)
            expected = golden / 2 + golden / 2
            for _ in range(world - 2):
                expected = expected + torch.zeros_like(expected)
            y = torch.empty_like(x)
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            dist.barrier()
            with comm.capture(), torch.cuda.graph(graph):
                comm.all_reduce(x, out=y, registered=True)
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            _bits_equal(y, expected)
            torch.testing.assert_close(y, golden, rtol=0, atol=0)
            _bits_equal(comm.all_reduce(x), expected)
            count += 1
            layers.add(path.stem)
            del graph
    assert (
        count > 0
        and {"L00", "L01", "L02", "L03", "L14", "L20", "L21", "L24", "L28"} <= layers
    )
    print(
        f"rank={rank} per-layer golden reduction gates={count}, "
        f"layers={sorted(layers)}",
        flush=True,
    )


def _pp_gate(rank: int, world: int, gloo) -> None:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.static_cuda_transfer import enqueue_static_cuda_transfer

    comm = PyNcclCommunicator(gloo, rank)
    assert not comm.disabled
    for rows in range(1, 9):
        for mirrored in (False, True):
            schema = [((rows, 4, 5120), torch.bfloat16), ((rows, 4), torch.float32)]
            if mirrored:
                schema += [
                    ((rows, 512), torch.float16),
                    ((rows, 128), torch.float16),
                    ((rows, 2048), torch.int32),
                    ((rows,), torch.int32),
                ]
            payload = {
                str(i): torch.full(s, 7 if rank == 0 else 0, device="cuda", dtype=d)
                for i, (s, d) in enumerate(schema)
            }
            consumer = torch.cuda.Stream()
            # First-use chain: interior ranks receive then forward, without
            # a Torch NCCL warmup or all-rank batched-P2P requirement.
            handles = []
            if rank > 0:
                handles = enqueue_static_cuda_transfer(
                    comm, payload, rank - 1, send=False
                )
                with torch.cuda.stream(consumer):
                    for handle in handles:
                        handle.wait()
                    checks = [t.clone() for t in payload.values()]
                torch.cuda.current_stream().wait_stream(consumer)
            else:
                checks = list(payload.values())
            if rank < world - 1:
                handles += enqueue_static_cuda_transfer(
                    comm, payload, rank + 1, send=True
                )
            for handle in handles:
                handle.wait()
            torch.cuda.synchronize()
            assert all(handle.is_completed() for handle in handles)
            assert all(bool(torch.all(t == 7)) for t in checks)
            dist.barrier()
    comm.destroy()


def _worker(rank: int, world: int, rendezvous: str, mode: str) -> None:
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=120),
    )
    gloo = dist.new_group(backend="gloo")
    try:
        if mode == "pp":
            _pp_gate(rank, world, gloo)
        else:
            from vllm.distributed.device_communicators.custom_all_reduce import (
                CustomAllreduce,
            )

            comm = CustomAllreduce(gloo, rank, max_size=2 * 1024 * 1024)
            assert not comm.disabled
            try:
                _reduce_gate(comm, rank, world)
                _layer_goldens(comm, rank, world)
            finally:
                comm.close()
    finally:
        dist.destroy_process_group()


def test_ds41_all_reduce_graph_and_layer_goldens(tmp_path: Path) -> None:
    world = int(os.environ.get("DS41_COMM_WORLD", "2"))
    gpu_gate(world)
    mp.spawn(
        _worker, args=(world, (tmp_path / "ar").as_uri(), "ar"), nprocs=world, join=True
    )


def test_ds41_static_pp_stream_order_and_first_use_chain(tmp_path: Path) -> None:
    world = int(os.environ.get("DS41_COMM_WORLD", "2"))
    gpu_gate(world)
    mp.spawn(
        _worker, args=(world, (tmp_path / "pp").as_uri(), "pp"), nprocs=world, join=True
    )
