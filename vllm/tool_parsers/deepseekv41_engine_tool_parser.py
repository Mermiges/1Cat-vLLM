# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 DSML tool parser (``--tool-call-parser deepseek_v41``).

V4.1 tool calls (checkpoint ``encoding/encoding.py``; the tag names carry a LEADING SPACE)::

    <｜DSML｜ calls>
    <｜DSML｜ invoke name="get_weather">
    <｜DSML｜ parameter name="location" string="true">Hangzhou</｜DSML｜ parameter>
    <｜DSML｜ parameter name="days" string="false">3</｜DSML｜ parameter>
    </｜DSML｜ invoke>
    </｜DSML｜ calls>

Upstream vLLM implements this as an adapter over its newer ``vllm.parser.engine`` framework, which 1Cat does not
have; this parser runs on 1Cat's ``ToolParser`` base and parses with the OFFICIAL grammar and decoder
(``parse_tool_calls`` / ``decode_dsml_to_arguments`` from the vendored encoding), so it accepts exactly what the
reference accepts. It is strict on purpose (1Cat issue #597: V4 on SM70 emitted DSML with a missing ``<`` and
stray characters inside parameter names): a block that does not parse, has non-JSON arguments, names an
undeclared tool or an undeclared parameter is NOT turned into a tool call -- the text is returned as content and a
warning is logged. Streaming emits content up to the block start (holding back a possible partial start tag),
buffers the block, and emits each complete call (name + full arguments) once the block closes.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.deepseek_v41_encoding import (
    dsml_token,
    eos_token,
    parse_tool_calls,
    tool_calls_block_name,
)
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser
from vllm.tool_parsers.utils import partial_tag_overlap

logger = init_logger(__name__)

CALLS_START = f"<{dsml_token}{tool_calls_block_name}>"     # "<｜DSML｜ calls>"
CALLS_END = f"</{dsml_token}{tool_calls_block_name}>"      # "</｜DSML｜ calls>"
CALLS_LEAD = "\n\n" + CALLS_START   # the reference separates content and calls by a blank line (not content)


class MalformedToolCalls(ValueError):
    pass


def _tool_name(tool: Any) -> str | None:
    fn = getattr(tool, "function", None)
    if fn is None and isinstance(tool, dict):
        fn = tool.get("function", tool)
    name = getattr(fn, "name", None) if fn is not None else None
    if name is None and isinstance(fn, dict):
        name = fn.get("name")
    return name


def _tool_properties(tool: Any) -> dict[str, Any] | None:
    fn = getattr(tool, "function", None)
    if fn is None and isinstance(tool, dict):
        fn = tool.get("function", tool)
    params = getattr(fn, "parameters", None) if fn is not None else None
    if params is None and isinstance(fn, dict):
        params = fn.get("parameters")
    if not isinstance(params, dict):
        return None
    props = params.get("properties")
    return props if isinstance(props, dict) else None


def parse_calls_block(block: str, tools: Sequence[Any] | None) -> list[tuple[str, str]]:
    """``<｜DSML｜ calls>...</｜DSML｜ calls>`` -> [(qualified name, canonical JSON arguments)]; raises
    MalformedToolCalls on anything the reference grammar rejects or the request's tools do not declare."""
    if not block.startswith(CALLS_START) or not block.endswith(CALLS_END):
        raise MalformedToolCalls("block is not delimited by the DSML calls tags")
    try:
        # the reference parser starts right after "<｜DSML｜ calls" and expects ">\n" first
        end, stop, calls = parse_tool_calls(len(CALLS_START) - 1, block)
    except (ValueError, AssertionError) as exc:
        raise MalformedToolCalls(str(exc)) from exc
    if stop != CALLS_END or end != len(block):
        raise MalformedToolCalls("text after the closing DSML calls tag or unterminated block")
    if not calls:
        raise MalformedToolCalls("empty DSML calls block")
    declared = {_tool_name(t): t for t in (tools or [])}
    result = []
    for call in calls:
        name = call["name"] if call.get("namespace") is None else f"{call['namespace']}::{call['name']}"
        try:
            args = json.loads(call["arguments"])
        except json.JSONDecodeError as exc:
            raise MalformedToolCalls(f"arguments of {name!r} are not JSON: {exc}") from exc
        if not isinstance(args, dict):
            raise MalformedToolCalls(f"arguments of {name!r} are not an object")
        if declared:
            if name not in declared:
                raise MalformedToolCalls(f"tool {name!r} is not among the request's tools {sorted(declared)}")
            props = _tool_properties(declared[name])
            unknown = sorted(set(args) - set(props)) if props is not None else []
            if unknown:
                raise MalformedToolCalls(f"tool {name!r} has undeclared parameters {unknown}")
        result.append((name, json.dumps(args, ensure_ascii=False)))
    return result


class DeepSeekV41EngineToolParser(ToolParser):
    tool_call_start_token: str = CALLS_START
    tool_call_end_token: str = CALLS_END

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self.prev_tool_call_arr: list[dict[str, Any]] = []
        self.streamed_args_for_tool: list[str] = []
        self.current_tool_id = -1
        self._sent_upto = 0
        self._block_start: int | None = None
        self._lead = ""

    def adjust_request(self, request: ChatCompletionRequest | ResponsesRequest
                       ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # ｜DSML｜ is a special token: it must survive detokenization for the parser to see the calls.
            request.skip_special_tokens = False
        return request

    def _tools_for(self, request: Any) -> Sequence[Any] | None:
        return getattr(request, "tools", None) or self.tools

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        start = model_output.find(CALLS_START)
        if start < 0:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)
        end = model_output.find(CALLS_END, start)
        tail = model_output[end + len(CALLS_END):] if end >= 0 else ""
        try:
            if end < 0:
                raise MalformedToolCalls("unterminated DSML calls block")
            if tail.strip() not in ("", eos_token):
                raise MalformedToolCalls("text after the DSML calls block")
            calls = parse_calls_block(model_output[start: end + len(CALLS_END)], self._tools_for(request))
        except MalformedToolCalls as exc:
            logger.warning("DeepSeek-V4.1 DSML tool calls rejected (returned as content): %s", exc)
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)
        tool_calls = [ToolCall(type="function", function=FunctionCall(name=name, arguments=args))
                      for name, args in calls]
        content = model_output[:start]
        if content.endswith("\n\n"):
            content = content[:-2]
        return ExtractedToolCallInformation(tools_called=True, tool_calls=tool_calls, content=content or None)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        if not previous_text:
            self._sent_upto, self._block_start = 0, None
        if self._block_start is None:
            start = current_text.find(CALLS_START, self._sent_upto)
            if start < 0:
                stop = len(current_text) - partial_tag_overlap(current_text, CALLS_LEAD)
                if stop <= self._sent_upto:
                    return None
                content, self._sent_upto = current_text[self._sent_upto: stop], stop
                return DeltaMessage(content=content)
            lead = max(start - 2 if current_text[max(start - 2, 0): start] == "\n\n" else start, self._sent_upto)
            content = current_text[self._sent_upto: lead]
            self._lead = current_text[lead:start]          # dropped if the block parses, re-emitted if rejected
            self._block_start = self._sent_upto = start
            if content:
                return DeltaMessage(content=content)
        end = current_text.find(CALLS_END, self._block_start)
        if end < 0:
            return None
        block_end = end + len(CALLS_END)
        block = current_text[self._block_start: block_end]
        self._block_start, self._sent_upto = None, block_end
        try:
            calls = parse_calls_block(block, self._tools_for(request))
        except MalformedToolCalls as exc:
            logger.warning("DeepSeek-V4.1 DSML tool calls rejected (streamed as content): %s", exc)
            return DeltaMessage(content=self._lead + block)
        deltas = []
        for name, args in calls:
            self.current_tool_id += 1
            self.prev_tool_call_arr.append({"name": name, "arguments": args})
            self.streamed_args_for_tool.append(args)
            deltas.append(DeltaToolCall(index=self.current_tool_id, type="function",
                                        id=f"call_{uuid.uuid4().hex[:24]}",
                                        function=DeltaFunctionCall(name=name, arguments=args).model_dump(
                                            exclude_none=True)))
        return DeltaMessage(tool_calls=deltas)
