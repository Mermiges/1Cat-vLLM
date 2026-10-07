# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pin actual V4.1 payload admission independently of graph push dispatch."""

import pytest
import torch

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce


@pytest.mark.parametrize("world", (2, 4))
@pytest.mark.parametrize("rows", range(1, 9))
@pytest.mark.parametrize(
    "width,dtype",
    ((5120, torch.float16), (5120, torch.float32), (25600, torch.float32)),
)
def test_v41_payload_admission(
    world: int, rows: int, width: int, dtype: torch.dtype
) -> None:
    comm = CustomAllreduce.__new__(CustomAllreduce)
    comm.disabled = False
    comm.world_size = world
    comm.fully_connected = True
    comm.tp8_hierarchical = False
    comm.dispatch_max_size = 1024 * 1024
    assert comm.should_custom_ar(torch.empty(rows, width, dtype=dtype))
    comm.disabled = True
    assert not comm.should_custom_ar(torch.empty(rows, width, dtype=dtype))
