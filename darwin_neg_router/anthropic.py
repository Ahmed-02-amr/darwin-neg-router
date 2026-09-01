from __future__ import annotations

import hashlib
import json
import math
import uuid
from typing import Any, Iterable

from .types import Candidate, ChatRequest, normalize_system_messages


def anthropic_chat_request(body: dict[str, Any], default_max_tokens: int) -> ChatRequest:
    """Translate an Anthropic Messages request into the router's internal contract.

    Anthropic represents tool calls/results as message content blocks, whereas
    the native Darwin and Ollama layers use OpenAI-style assistant tool calls
    followed by ``tool`` role messages.  Keeping the translation at the API
    boundary lets both public protocols exercise the same NEG and verifier
    routing implementation.
    """

    requested_model = str(body.get("model") or "darwin-neg-agent")
    messages = _anthropic_messages_to_openai(body.get("system"), body.get("messages", []))
    tools = [_anthropic_tool_to_openai(tool) for tool in body.get("tools", [])]
    tool_choice, parallel_tool_calls = _anthropic_tool_choice(body.get("tool_choice"))
    thinking = body.get("thinking")
    think = not (
        thinking is False
        or (isinstance(thinking, dict) and thinking.get("type") == "disabled")
    )
    requested_tokens = int(body.get("max_tokens") or default_max_tokens)

    return ChatRequest(
        messages=messages,
        model=requested_model,
        tools=tools,
        temperature=float(body.get("temperature", 0.0) or 0.0),
        top_p=float(body.get("top_p", 1.0) or 1.0),
        top_k=max(1, int(body.get("top_k", 1) or 1)),
        max_tokens=min(requested_tokens, default_max_tokens),
        stop=body.get("stop_sequences"),
        seed=body.get("seed"),
        think=think,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        presence_penalty=float(body.get("presence_penalty", 0.0) or 0.0),
        frequency_penalty=float(body.get("frequency_penalty", 0.0) or 0.0),
        repeat_penalty=float(body.get("repeat_penalty", 1.0) or 1.0),
    )


def anthropic_response(
    candidate: Candidate, model: str = "darwin-neg-agent", *, message_id: str | None = None
) -> dict[str, Any]:
    content = _anthropic_content(candidate)
    return {
        "id": message_id or f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _anthropic_stop_reason(candidate.finish_reason),
        "stop_sequence": None,
        "usage": {
            "input_tokens": candidate.prompt_tokens,
            "output_tokens": candidate.completion_tokens,
        },
    }


def stream_anthropic_response(response: dict[str, Any]) -> Iterable[str]:
    """Emit the ordered SSE event sequence consumed by Anthropic SDK clients."""

    start = {
        "type": "message_start",
        "message": {
            "id": response["id"],
            "type": "message",
            "role": "assistant",
            "model": response["model"],
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": response["usage"]["input_tokens"],
                "output_tokens": 0,
            },
        },
    }
    yield _sse("message_start", start)

    for index, block in enumerate(response["content"]):
        block_type = block["type"]
        if block_type == "thinking":
            yield _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                },
            )
            if block.get("thinking"):
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "thinking_delta",
                            "thinking": block["thinking"],
                        },
                    },
                )
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "signature_delta",
                        "signature": block["signature"],
                    },
                },
            )
        elif block_type == "text":
            yield _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            if block.get("text"):
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": block["text"]},
                    },
                )
        elif block_type == "tool_use":
            yield _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": block["id"],
                        "name": block["name"],
                        "input": {},
                    },
                },
            )
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(
                            block.get("input", {}), ensure_ascii=False, separators=(",", ":")
                        ),
                    },
                },
            )
        else:  # pragma: no cover - all blocks are constructed locally
            continue

        yield _sse(
            "content_block_stop", {"type": "content_block_stop", "index": index}
        )

    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": response["stop_reason"],
                "stop_sequence": response["stop_sequence"],
            },
            "usage": {"output_tokens": response["usage"]["output_tokens"]},
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})


def estimate_input_tokens(body: dict[str, Any]) -> int:
    """Return a conservative local estimate for Claude SDK context checks.

    The native llama.cpp tokenizer is behind the generation server and cannot
    tokenize Anthropic's pre-rendered tool/system prompt directly.  A UTF-8
    byte estimate is preferable to omitting the endpoint, and is deliberately
    conservative for English/code-heavy Claude Code sessions.
    """

    relevant = {
        "system": body.get("system"),
        "messages": body.get("messages", []),
        "tools": body.get("tools", []),
    }
    encoded = json.dumps(relevant, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return max(1, math.ceil(len(encoded) / 3.5))


def _anthropic_messages_to_openai(
    system: Any, messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    system_text = _system_text(system)
    if system_text:
        converted.append({"role": "system", "content": system_text})

    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            converted.append({"role": role, "content": str(content)})
            continue

        if role == "assistant":
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(str(block.get("text", "")))
                elif block_type == "thinking":
                    reasoning_parts.append(str(block.get("thinking", "")))
                elif block_type == "tool_use":
                    tool_calls.append(
                        {
                            "id": str(block.get("id") or f"call_{uuid.uuid4().hex[:24]}"),
                            "type": "function",
                            "function": {
                                "name": str(block.get("name", "")),
                                "arguments": json.dumps(
                                    block.get("input", {}), ensure_ascii=False
                                ),
                            },
                        }
                    )
            item: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
            if reasoning_parts:
                item["reasoning_content"] = "\n".join(reasoning_parts)
            if tool_calls:
                item["tool_calls"] = tool_calls
            converted.append(item)
            continue

        tool_results: list[dict[str, Any]] = []
        user_blocks: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                user_blocks.append({"type": "text", "text": str(block)})
            elif block.get("type") == "tool_result":
                result_text = _tool_result_text(block.get("content", ""))
                if block.get("is_error"):
                    result_text = f"Tool error: {result_text}"
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(block.get("tool_use_id", "")),
                        "content": result_text,
                    }
                )
            else:
                user_blocks.append(block)
        converted.extend(tool_results)
        user_content = _user_content(user_blocks)
        if user_content not in ("", []):
            converted.append({"role": "user", "content": user_content})

    return normalize_system_messages(converted)


def _anthropic_tool_to_openai(tool: dict[str, Any]) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": str(tool.get("name", "")),
        "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
    }
    if tool.get("description") is not None:
        function["description"] = str(tool["description"])
    return {"type": "function", "function": function}


def _anthropic_tool_choice(value: Any) -> tuple[str | dict[str, Any] | None, bool | None]:
    if not isinstance(value, dict):
        return value if isinstance(value, str) else None, None
    choice_type = value.get("type")
    if choice_type == "tool":
        choice: str | dict[str, Any] | None = {
            "type": "function",
            "function": {"name": str(value.get("name", ""))},
        }
    elif choice_type == "any":
        choice = "required"
    elif choice_type in {"auto", "none"}:
        choice = str(choice_type)
    else:
        choice = None
    parallel = None
    if "disable_parallel_tool_use" in value:
        parallel = not bool(value["disable_parallel_tool_use"])
    return choice, parallel


def _anthropic_content(candidate: Candidate) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if candidate.reasoning_content:
        content.append(
            {
                "type": "thinking",
                "thinking": candidate.reasoning_content,
                "signature": _thinking_signature(candidate.reasoning_content),
            }
        )
    if candidate.content:
        content.append({"type": "text", "text": candidate.content})
    for call in candidate.tool_calls:
        function = call.get("function", {})
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_arguments = {"raw_arguments": arguments}
        elif isinstance(arguments, dict):
            parsed_arguments = arguments
        else:
            parsed_arguments = {"value": arguments}
        content.append(
            {
                "type": "tool_use",
                "id": str(call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}"),
                "name": str(function.get("name", "")),
                "input": parsed_arguments,
            }
        )
    if not content:
        content.append({"type": "text", "text": ""})
    return content


def _anthropic_stop_reason(finish_reason: str | None) -> str:
    return {
        "tool_calls": "tool_use",
        "tool_use": "tool_use",
        "length": "max_tokens",
        "max_tokens": "max_tokens",
        "content_filter": "refusal",
        "stop_sequence": "stop_sequence",
    }.get(str(finish_reason), "end_turn")


def _system_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "".join(
            str(block.get("text", ""))
            for block in system
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _user_content(blocks: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    if not blocks:
        return ""
    if all(block.get("type") == "text" for block in blocks):
        return "".join(str(block.get("text", "")) for block in blocks)

    converted: list[dict[str, Any]] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            converted.append({"type": "text", "text": str(block.get("text", ""))})
        elif block_type == "image":
            source = block.get("source", {})
            if source.get("type") == "base64":
                url = f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
            else:
                url = str(source.get("url", ""))
            converted.append({"type": "image_url", "image_url": {"url": url}})
        else:
            converted.append(
                {
                    "type": "text",
                    "text": json.dumps(block, ensure_ascii=False),
                }
            )
    return converted


def _tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(json.dumps(block, ensure_ascii=False))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False)


def _thinking_signature(reasoning: str) -> str:
    digest = hashlib.sha256(reasoning.encode("utf-8")).hexdigest()
    return f"darwin-neg-local-v1:{digest}"


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
