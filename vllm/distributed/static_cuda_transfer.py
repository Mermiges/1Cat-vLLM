# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped, metadata-free CUDA P2P on an initialized NCCL communicator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator


class StaticCudaTransferHandle:
    """Expose completion of grouped P2P without synchronizing the host.

    Like ProcessGroupNCCL Work.wait(), wait orders the consumer's current stream;
    it does not block the host waiting for a peer to finish its forward.
    """

    def __init__(self, event: torch.cuda.Event, device: torch.device) -> None:
        self.event = event
        self.device = device

    def is_completed(self) -> bool:
        return self.event.query()

    def wait(self) -> None:
        torch.cuda.current_stream(self.device).wait_event(self.event)


def enqueue_static_cuda_transfer(
    comm: PyNcclCommunicator,
    tensors: dict[str, torch.Tensor],
    peer: int,
    *,
    send: bool,
) -> list[StaticCudaTransferHandle]:
    """Submit ordered tensors as one NCCL group, with no copies or metadata.

    The communicator must already have been collectively initialized. Reject
    invalid buffers before entering a group; never catch NCCL failures and retry
    on a different communicator (the peer may already have consumed a message).
    """
    if comm.disabled:
        raise RuntimeError("static CUDA transfer requires enabled PyNccl")
    payload = list(tensors.values())
    for tensor in payload:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("static tensor transfer only accepts tensors")
        if tensor.device != comm.device or not tensor.is_contiguous():
            raise ValueError(
                "static CUDA transfer needs contiguous communicator-device tensors"
            )
    payload = [t for t in payload if t.numel()]
    if not payload:
        return []
    stream = torch.cuda.current_stream(comm.device)
    operation = comm.send if send else comm.recv
    comm.group_start()
    try:
        for tensor in payload:
            operation(tensor, peer, stream=stream)
    finally:
        comm.group_end()
    # Protect both send sources and receive destinations if the caller switches
    # streams or drops a temporary view before the transfer has completed.
    for tensor in payload:
        tensor.record_stream(stream)
    event = torch.cuda.Event()
    event.record(stream)
    return [StaticCudaTransferHandle(event, comm.device)]
