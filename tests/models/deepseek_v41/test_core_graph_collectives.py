# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP4 custom all-reduce in graph segments and between them (board A only)."""

import socket
from datetime import timedelta

import pytest
import torch
import torch.multiprocessing as mp

pytestmark = pytest.mark.sm70


def _rank(rank: int, port: int) -> None:
    import os

    from vllm.compilation.breakable_cudagraph import (
        BreakableCUDAGraphCapture,
        eager_break_during_capture,
    )
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
        tensor_model_parallel_all_reduce,
    )
    from vllm.distributed.parallel_state import get_tp_group, graph_capture

    os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    vc = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=4))
    with set_current_vllm_config(vc), torch.inference_mode():
        init_distributed_environment(
            4, rank, f"tcp://127.0.0.1:{port}", rank, timeout=timedelta(seconds=90)
        )
        initialize_model_parallel(tensor_model_parallel_size=4)
        comm = get_tp_group().device_communicator.ca_comm
        assert comm is not None and not comm.disabled, "custom AR must be active"
        x = torch.full((1, 5120), float(rank), device=device)
        out = torch.empty_like(x)
        calls = [0]

        @eager_break_during_capture
        def eager_reduce(y, out):
            assert not torch.cuda.is_current_stream_capturing()
            calls[0] += 1
            out.copy_(tensor_model_parallel_all_reduce(y))

        def run():
            y = tensor_model_parallel_all_reduce(x)
            eager_reduce(y, out)
            return out + 1

        run()
        torch.cuda.synchronize()
        with graph_capture(device):
            cap = BreakableCUDAGraphCapture()
            with cap:
                captured = run()
        assert cap.num_eager_breaks == 1 and cap.num_graphs == 2
        for step in range(4):
            x.fill_(rank + step)
            before = calls[0]
            cap.replay()
            torch.cuda.synchronize()
            assert calls[0] == before + 1
            assert torch.equal(
                captured, torch.full_like(captured, (6 + 4 * step) * 4 + 1)
            )
            assert torch.equal(captured, run())
        torch.cuda.synchronize()


def test_tp4_custom_allreduce_breakable_capture():
    if torch.cuda.device_count() != 4:
        pytest.skip("requires the four assigned board-A GPUs")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_rank, args=(port,), nprocs=4, join=True)
