from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import replace
from typing import Any

from .types import Candidate, ChatRequest


_COUNT_FIELDS = (
    "duplicates_removed",
    "invalid_calls_removed",
    "parallel_overflow_removed",
    "stalled_calls_blocked",
    "recovery_inferences",
    "long_form_retries",
)


def canonical_tool_signature(call: dict[str, Any]) -> str | None:
    """Return a stable action signature without retaining arguments in telemetry."""

    function = call.get("function")
    if not isinstance(function, dict):
        return None
    name = str(function.get("name") or "").strip()
    if not name:
        return None
    arguments = function.get("arguments", "{}")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = re.sub(r"\s+", " ", arguments.strip())
    try:
        rendered = json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        rendered = repr(arguments)
    return f"{name}\0{rendered}"


def sanitize_candidate_tools(
    request: ChatRequest,
    candidate: Candidate,
    *,
    max_parallel_tool_calls: int,
) -> Candidate:
    """Keep unique valid actions while enforcing the caller's parallel contract.

    Exact duplicates are never useful within one assistant message. Distinct
    actions remain ordered and are preserved up to a deliberately generous
    safety ceiling. When the client explicitly disables parallel tools, its
    one-action contract takes precedence over that ceiling.
    """

    raw_calls = list(candidate.tool_calls or [])
    if not raw_calls:
        return candidate

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    invalid = 0
    duplicates = 0
    for call in raw_calls:
        signature = canonical_tool_signature(call)
        if signature is None:
            invalid += 1
            continue
        if signature in seen:
            duplicates += 1
            continue
        seen.add(signature)
        unique.append(call)

    limit = 1 if request.parallel_tool_calls is False else max(1, max_parallel_tool_calls)
    kept = unique[:limit]
    overflow = max(0, len(unique) - len(kept))
    report = dict(candidate.metadata.get("tool_guard") or {})
    report.update(
        {
            "raw_tool_calls": len(raw_calls),
            "unique_tool_calls": len(unique),
            "kept_tool_calls": len(kept),
            "duplicates_removed": int(report.get("duplicates_removed", 0)) + duplicates,
            "invalid_calls_removed": int(report.get("invalid_calls_removed", 0)) + invalid,
            "parallel_overflow_removed": int(report.get("parallel_overflow_removed", 0))
            + overflow,
            "parallel_limit": limit,
            "upstream_finish_reason": candidate.finish_reason,
        }
    )
    metadata = dict(candidate.metadata)
    metadata["tool_guard"] = report
    finish_reason = "tool_calls" if kept else (
        "stop" if candidate.finish_reason == "tool_calls" else candidate.finish_reason
    )
    return replace(
        candidate,
        tool_calls=kept,
        finish_reason=finish_reason,
        metadata=metadata,
    )


def stalled_tool_signatures(
    messages: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    *,
    unchanged_result_limit: int,
) -> set[str]:
    """Find actions whose most recent completed retries returned the same result.

    A real user message resets the history window. Tool-result messages do not.
    Polling and retries whose output changes therefore remain available, while a
    third identical action after two identical results is considered stalled.
    """

    limit = max(2, unchanged_result_limit)
    results = _action_results_since_last_user_message(messages)
    stalled: set[str] = set()
    for call in calls:
        signature = canonical_tool_signature(call)
        if signature is None:
            continue
        recent = results.get(signature, [])[-limit:]
        if len(recent) == limit and len(set(recent)) == 1:
            stalled.add(signature)
    return stalled


def remove_stalled_tools(candidate: Candidate, signatures: set[str]) -> Candidate:
    if not signatures or not candidate.tool_calls:
        return candidate
    kept: list[dict[str, Any]] = []
    blocked_names: set[str] = set()
    for call in candidate.tool_calls:
        if canonical_tool_signature(call) not in signatures:
            kept.append(call)
            continue
        function = call.get("function") or {}
        blocked_names.add(str(function.get("name") or "unknown"))

    blocked = len(candidate.tool_calls) - len(kept)
    report = dict(candidate.metadata.get("tool_guard") or {})
    report["stalled_calls_blocked"] = int(report.get("stalled_calls_blocked", 0)) + blocked
    report["stalled_tool_names"] = sorted(blocked_names)
    metadata = dict(candidate.metadata)
    metadata["tool_guard"] = report
    return replace(
        candidate,
        tool_calls=kept,
        finish_reason="tool_calls" if kept else "stop",
        metadata=metadata,
    )


def merge_guard_usage(first: Candidate, second: Candidate, **flags: int | bool) -> Candidate:
    """Return the second result while accounting for all guarded inference work."""

    first_report = dict(first.metadata.get("tool_guard") or {})
    second_report = dict(second.metadata.get("tool_guard") or {})
    merged = dict(second_report)
    for field in _COUNT_FIELDS:
        merged[field] = int(first_report.get(field, 0)) + int(second_report.get(field, 0))
    for key, value in flags.items():
        if isinstance(value, bool):
            merged[key] = value
        else:
            merged[key] = int(merged.get(key, 0)) + int(value)
    if first_report.get("stalled_tool_names") or second_report.get("stalled_tool_names"):
        merged["stalled_tool_names"] = sorted(
            set(first_report.get("stalled_tool_names") or [])
            | set(second_report.get("stalled_tool_names") or [])
        )
    metadata = dict(second.metadata)
    metadata["tool_guard"] = merged
    return replace(
        second,
        prompt_tokens=first.prompt_tokens + second.prompt_tokens,
        completion_tokens=first.completion_tokens + second.completion_tokens,
        metadata=metadata,
    )


def _action_results_since_last_user_message(
    messages: list[dict[str, Any]],
) -> dict[str, list[str]]:
    pending: dict[str, str] = {}
    results: dict[str, list[str]] = defaultdict(list)
    for message in messages:
        role = message.get("role")
        if role == "user":
            pending.clear()
            results.clear()
            continue
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                signature = canonical_tool_signature(call)
                call_id = str(call.get("id") or "")
                if signature is not None and call_id:
                    pending[call_id] = signature
            continue
        if role != "tool":
            continue
        signature = pending.get(str(message.get("tool_call_id") or ""))
        if signature is None:
            continue
        results[signature].append(_result_fingerprint(message.get("content", "")))
    return results


def _result_fingerprint(content: Any) -> str:
    if not isinstance(content, str):
        try:
            content = json.dumps(content, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            content = repr(content)
    normalized = re.sub(r"\s+", " ", content.strip())
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()
