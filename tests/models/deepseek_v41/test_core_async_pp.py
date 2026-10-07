# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async PP opt-in, sampled-id handoff ordering, and four decode-step row parity."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v41.common.async_pp import SampledPPIds, require_async_pp
from vllm.models.deepseek_v41.common.contracts import EngramBatchLayout
from vllm.v1.worker.gpu_worker import _EngramStepPlanner

from .test_core_worker_hooks import _new_req, _sched, _vllm_config, _worker


def test_async_pp_opt_in_and_refusals(monkeypatch):
    monkeypatch.delenv("VLLM_DS41_CORE_ASYNC_PP", raising=False)
    cfg = _vllm_config(async_scheduling=True)
    with pytest.raises(NotImplementedError, match="synchronous"):
        require_async_pp(cfg)
    monkeypatch.setenv("VLLM_DS41_CORE_ASYNC_PP", "bad")
    with pytest.raises(ValueError, match="not a boolean"):
        require_async_pp(cfg)
    monkeypatch.setenv("VLLM_DS41_CORE_ASYNC_PP", "1")
    require_async_pp(cfg)
    worker = _worker(SimpleNamespace(engram_service=object()), cfg)
    assert worker._engram_service() is not None
    with pytest.raises(NotImplementedError, match="PP > 1"):
        require_async_pp(_vllm_config(pp=1, async_scheduling=True))
    cfg.speculative_config = object()
    with pytest.raises(NotImplementedError, match="speculative"):
        require_async_pp(cfg)


def test_async_planner_keeps_unknown_decode_ids_unknown():
    planner = _EngramStepPlanner()
    p = [11, 12, 13]
    planner.plan(_sched(new=[_new_req("a", p)], num_scheduled={"a": 3}))
    cached = SimpleNamespace(
        req_ids=["a"],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        num_computed_tokens=[3],
    )
    for pos in range(3, 7):
        cached.num_computed_tokens = [pos]
        step = planner.plan(_sched(cached=cached, num_scheduled={"a": 1}))
        assert step.reqs[0].token_ids is None
        assert step.reqs[0].start_pos == pos
    cached.resumed_req_ids = {"a"}
    cached.all_token_ids = {"a": [11, 12, 13, 20, 21, 22, 23]}
    step = planner.plan(_sched(cached=cached, num_scheduled={"a": 1}))
    assert step.reqs[0].token_ids.tolist() == [23]


@pytest.mark.sm70
def test_received_fill_owns_storage_and_reorders(monkeypatch):
    ids = torch.tensor([[101], [202]], device="cuda", dtype=torch.int32)
    received = SampledPPIds.receive(ids, ("a", "b"))
    first, event = received.for_batch(("a", "b"), frozenset({"a", "b"}))
    assert first.is_pinned() and first.dtype == torch.int32
    event.synchronize()
    assert first.tolist() == [101, 202]
    second, event = received.for_batch(("b", "new", "a"), frozenset({"a", "b"}))
    event.synchronize()
    assert second.tolist() == [202, 0, 101]
    assert first.data_ptr() != second.data_ptr()
    with pytest.raises(RuntimeError, match="no sampled id"):
        received.for_batch(("unknown",), frozenset({"unknown"}))


@pytest.mark.sm70
def test_async_four_decode_steps_equal_sync_rows(tmp_path, monkeypatch):
    from .test_engram_service import make_service
    from .test_engram_synth import build_synthetic_engram

    monkeypatch.setenv("VLLM_DS41_ENGRAM_REQUIRE_VERIFIED", "0")
    syn = build_synthetic_engram(tmp_path / "async")
    sync, async_ = make_service(syn, max_tokens=16), make_service(syn, max_tokens=16)
    planners = [_EngramStepPlanner(), _EngramStepPlanner()]
    prompts = {"a": [11, 12, 13], "b": [21, 22, 23]}
    try:
        sched = _sched(
            new=[_new_req(r, p) for r, p in prompts.items()],
            num_scheduled={"a": 3, "b": 3},
        )
        for svc, planner in zip((sync, async_), planners):
            plan = planner.plan(sched)
            svc.begin_step(plan)
            svc.bind_batch(
                EngramBatchLayout(
                    plan.step_id, ("a", "b"), np.array([0, 3, 6], np.int32), 6, 8, None
                )
            )
            for lid in svc.layers:
                svc.wait_rows(lid)
            svc.end_step(plan.step_id)
        for k in range(4):
            ids = [101 + k, 201 + k]
            orders = ("b", "a") if k % 2 else ("a", "b")
            received = SampledPPIds.receive(
                torch.tensor(ids, dtype=torch.int32, device="cuda").view(2, 1),
                ("a", "b"),
            )
            results = []
            for idx, (svc, planner) in enumerate(zip((sync, async_), planners)):
                cached = SimpleNamespace(
                    req_ids=["a", "b"],
                    resumed_req_ids=set(),
                    new_token_ids=[[x] for x in ids] if idx == 0 else [],
                    all_token_ids={},
                    num_computed_tokens=[3 + k, 3 + k],
                )
                plan = planner.plan(
                    _sched(cached=cached, num_scheduled={"a": 1, "b": 1})
                )
                svc.begin_step(plan)
                fill = (
                    received.for_batch(orders, frozenset(orders)) if idx == 1 else None
                )
                svc.bind_batch(
                    EngramBatchLayout(
                        plan.step_id, orders, np.array([0, 1, 2], np.int32), 2, 4, fill
                    )
                )
                results.append({lid: svc.wait_rows(lid).clone() for lid in svc.layers})
                svc.end_step(plan.step_id)
            torch.cuda.synchronize()
            for lid in sync.layers:
                assert torch.equal(results[0][lid], results[1][lid])
                assert not results[1][lid][2:].any()
    finally:
        sync.shutdown()
        async_.shutdown()


@pytest.mark.sm70
def test_runner_receive_and_layout_carry_pinned_fill(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gm

    from .test_core_worker_hooks import _runner

    runner = _runner(3, monkeypatch)
    pp = SimpleNamespace(
        world_size=3, is_last_rank=False, last_rank=8, device_group=object()
    )
    monkeypatch.setattr(gm, "get_pp_group", lambda: pp)
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda recv, **kw: recv.copy_(
            torch.tensor([[101], [202]], dtype=torch.int32, device="cuda")
        ),
    )
    runner.device = torch.device("cuda")
    runner.num_spec_tokens = 0
    runner.use_async_scheduling = True
    runner._engram_bind_service_cache = object()
    runner._is_all_reqs_chunked_prefill = lambda: False
    runner.requests = {}
    runner.input_batch.num_reqs = 2
    runner.input_batch.req_ids = ["r0", "r1"]
    runner.input_batch.num_tokens_no_spec = np.array([5, 3])
    runner.input_batch.is_token_ids = np.zeros((2, 16), dtype=bool)
    runner.discard_request_mask = SimpleNamespace(np=np.zeros(2, dtype=bool))
    runner._pp_receive_prev_sampled_token_ids_to_input_batch()
    runner.input_batch.req_ids = ["r1", "r0"]
    runner._engram_unknown_req_ids = frozenset({"r0", "r1"})
    runner._engram_step_id = 4
    layout = runner._engram_batch_layout(2, 2, 4)
    fill, ready = layout.sampled_fill
    assert fill.is_pinned() and fill.dtype == torch.int32
    ready.synchronize()
    assert fill.tolist() == [202, 101]
    assert layout.req_order == ("r1", "r0")
    assert runner._engram_prev_sampled is None
    runner._engram_step_id = 5
    with pytest.raises(RuntimeError, match="no sampled-id handoff"):
        runner._engram_batch_layout(2, 2, 4)


def test_async_layout_requires_worker_plan_and_decode_handoff(monkeypatch):
    from .test_core_worker_hooks import _runner

    runner = _runner(3, monkeypatch)
    runner.use_async_scheduling = True
    runner._engram_step_id = 1
    with pytest.raises(RuntimeError, match="worker's token plan"):
        runner._engram_batch_layout(2, 2, 4)
    runner._engram_unknown_req_ids = frozenset({"r0"})
    runner._engram_step_id = 2
    with pytest.raises(RuntimeError, match="no sampled-id handoff"):
        runner._engram_batch_layout(2, 2, 4)
    runner._engram_unknown_req_ids = frozenset()
    runner._engram_step_id = 3
    assert runner._engram_batch_layout(2, 2, 4).sampled_fill is None
