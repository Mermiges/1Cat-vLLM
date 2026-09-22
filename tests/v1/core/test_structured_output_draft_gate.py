# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured-output requests must not carry draft tokens before their
reasoning has ended (1CatAI/1Cat-vLLM#442): the grammar bitmask is only
applied once the reasoning marker has been seen, so a draft accepted in the
same step as the marker would run past it unconstrained and be rejected by
the FSM on the next advance."""

from unittest.mock import Mock

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.request import Request, RequestStatus

EOS_TOKEN_ID = 50256


def _scheduler_with_request(should_advance: bool) -> tuple[Scheduler, Request]:
    scheduler = Scheduler.__new__(Scheduler)
    sampling_params = SamplingParams(ignore_eos=True, max_tokens=4)
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    request = Request(
        request_id="0",
        prompt_token_ids=[0, 1],
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
    )
    request.structured_output_request = Mock()
    request.structured_output_request.grammar = Mock()
    request.structured_output_request.grammar.validate_tokens.side_effect = (
        lambda ids: ids
    )
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = request.num_tokens
    scheduler.structured_output_manager = Mock()
    scheduler.structured_output_manager.should_advance.return_value = should_advance
    scheduler.requests = {request.request_id: request}
    scheduler.ddtree_payloads_by_req_id = {}
    return scheduler, request


def _drafts() -> DraftTokenIds:
    return DraftTokenIds(req_ids=["0"], draft_token_ids=[[7, 8, 9]])


def test_structured_request_gets_no_drafts_before_reasoning_ends():
    scheduler, request = _scheduler_with_request(should_advance=False)
    request.spec_token_ids = [1, 2, 3]  # stale drafts from an earlier step
    scheduler.update_draft_token_ids(_drafts())
    assert request.spec_token_ids == []
    request.structured_output_request.grammar.validate_tokens.assert_not_called()


def test_structured_request_keeps_validated_drafts_after_reasoning_ends():
    scheduler, request = _scheduler_with_request(should_advance=True)
    scheduler.update_draft_token_ids(_drafts())
    assert request.spec_token_ids == [7, 8, 9]
    request.structured_output_request.grammar.validate_tokens.assert_called_once()


def test_plain_request_drafts_are_untouched():
    scheduler, request = _scheduler_with_request(should_advance=False)
    request.structured_output_request = None
    scheduler.update_draft_token_ids(_drafts())
    assert request.spec_token_ids == [7, 8, 9]
