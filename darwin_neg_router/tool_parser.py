from __future__ import annotations

import json
import re
import uuid
from typing import Any


_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def split_thinking(text: str) -> tuple[str, str]:
    match = re.search(r"<think>\s*(.*?)\s*</think>\s*", text, re.DOTALL)
    if not match:
        return "", text.strip()
    reasoning = match.group(1).strip()
    visible = (text[: match.start()] + text[match.end() :]).strip()
    return reasoning, visible


def parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    for call in _CALL_RE.finditer(text):
        args: dict[str, Any] = {}
        for param in _PARAM_RE.finditer(call.group(2)):
            value = param.group(2).strip()
            try:
                args[param.group(1)] = json.loads(value)
            except json.JSONDecodeError:
                args[param.group(1)] = value
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": call.group(1), "arguments": json.dumps(args)},
            }
        )
    visible = _CALL_RE.sub("", text).strip()
    return visible, calls

