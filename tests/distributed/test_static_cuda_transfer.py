# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.distributed.static_cuda_transfer import enqueue_static_cuda_transfer


@pytest.fixture
def cuda_mocks(monkeypatch: pytest.MonkeyPatch):
    stream = Mock()
    event = Mock()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    records = []
    monkeypatch.setattr(
        torch.Tensor, "record_stream", lambda t, s: records.append((t, s))
    )
    calls = Mock()
    comm = SimpleNamespace(
        disabled=False,
        device=torch.device("cpu"),
        group_start=calls.start,
        group_end=calls.end,
        send=calls.send,
        recv=calls.recv,
    )
    return comm, calls, stream, event, records


@pytest.mark.parametrize("send", (False, True))
@pytest.mark.parametrize("count", (2, 6))
def test_grouped_static_transfer_preserves_views_and_orders_wait(
    cuda_mocks, send, count
):
    comm, calls, stream, event, records = cuda_mocks
    tensors = {str(i): torch.arange(i + 1) for i in range(count)}
    handles = enqueue_static_cuda_transfer(comm, tensors, 1, send=send)
    expected = ["start"] + (["send"] if send else ["recv"]) * count + ["end"]
    assert [call[0] for call in calls.mock_calls] == expected
    operation = calls.send if send else calls.recv
    for call, tensor in zip(operation.call_args_list, tensors.values()):
        assert call.args[0] is tensor
        assert call.args[1] == 1 and call.kwargs == {"stream": stream}
    assert len(records) == count
    assert all(
        t is value and s is stream for (t, s), value in zip(records, tensors.values())
    )
    event.record.assert_called_once_with(stream)
    assert len(handles) == 1
    handles[0].wait()
    stream.wait_event.assert_called_once_with(event)
    event.synchronize.assert_not_called()


def test_group_end_closes_failed_submission_without_retry(cuda_mocks):
    comm, calls, _, event, _ = cuda_mocks
    calls.send.side_effect = RuntimeError("NCCL failure")
    with pytest.raises(RuntimeError, match="NCCL failure"):
        enqueue_static_cuda_transfer(comm, {"h": torch.ones(2)}, 1, send=True)
    assert [call[0] for call in calls.mock_calls] == ["start", "send", "end"]
    event.record.assert_not_called()


@pytest.mark.parametrize("invalid", ("disabled", "noncontiguous", "object", "device"))
def test_invalid_payload_rejected_before_group_start(cuda_mocks, invalid):
    comm, calls, _, _, _ = cuda_mocks
    tensor = torch.ones(2)
    error = (RuntimeError, TypeError, ValueError)
    if invalid == "disabled":
        comm.disabled = True
    elif invalid == "noncontiguous":
        tensor = torch.ones(2, 2).t()
    elif invalid == "object":
        tensor = object()
    else:
        comm.device = torch.device("cuda:0")
    with pytest.raises(error):
        enqueue_static_cuda_transfer(comm, {"h": tensor}, 1, send=True)
    calls.start.assert_not_called()


def test_zero_sized_payload_does_not_submit(cuda_mocks):
    comm, calls, _, _, _ = cuda_mocks
    assert enqueue_static_cuda_transfer(comm, {"h": torch.empty(0)}, 1, send=True) == []
    calls.start.assert_not_called()


@pytest.mark.parametrize("send", (False, True))
def test_group_coordinator_routes_multitensor_cuda_payload(
    cuda_mocks, monkeypatch, send
):
    from .test_static_tensor_transfer import _group

    comm, calls, stream, event, _ = cuda_mocks
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    group = _group(0 if send else 1)
    group.device_communicator = SimpleNamespace(pynccl_comm=comm)
    payload = {"hidden_states": torch.ones(1, 4, 5120), "pre_mix": torch.ones(1, 4)}
    handles = (
        group.isend_tensor_dict_static(payload)
        if send
        else group.irecv_tensor_dict_static(payload)
    )
    assert len(handles) == 1
    assert [call[0] for call in calls.mock_calls] == (
        ["start", "send", "send", "end"] if send else ["start", "recv", "recv", "end"]
    )
    event.record.assert_called_once_with(stream)


def test_event_handle_exposes_nonblocking_completion(cuda_mocks):
    comm, _, _, event, _ = cuda_mocks
    handle = enqueue_static_cuda_transfer(comm, {"h": torch.ones(2)}, 1, send=True)[0]
    event.query.return_value = False
    assert not handle.is_completed()
    event.query.return_value = True
    assert handle.is_completed()
    event.synchronize.assert_not_called()
