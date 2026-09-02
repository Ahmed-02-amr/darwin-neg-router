from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any


def normalize_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge system messages at index zero for strict Qwen chat templates.

    Anthropic clients normally send system instructions in the top-level
    ``system`` field, but some OpenAI/Claude compatibility layers also place a
    system-role item later in ``messages``. Qwen rejects that ordering. Keeping
    all non-system messages stable while consolidating the system text makes the
    boundary tolerant without changing tool/result sequencing.
    """

    system_parts: list[str] = []
    conversation: list[dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        if item.get("role") != "system":
            conversation.append(item)
            continue
        content = item.get("content", "")
        if isinstance(content, str):
            rendered = content
        else:
            rendered = json.dumps(content, ensure_ascii=False)
        if rendered.strip():
            system_parts.append(rendered)
    if system_parts:
        conversation.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    return conversation


@dataclass
class ChatRequest:
    messages: list[dict[str, Any]]
    model: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 1
    max_tokens: int = 8192
    stop: str | list[str] | None = None
    seed: int | None = None
    think: bool = True
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repeat_penalty: float = 1.0
    routing_profile: str | None = None


@dataclass
class Candidate:
    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def selection_text(self) -> str:
        if self.tool_calls:
            return f"tool_calls={self.tool_calls!r}\ncontent={self.content}"
        return self.content
