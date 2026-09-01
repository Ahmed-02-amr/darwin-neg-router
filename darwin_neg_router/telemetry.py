from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from .types import Candidate


class TelemetryStore:
    """Thread-safe, prompt-free runtime telemetry for the desktop controller."""

    def __init__(self, history_size: int = 100):
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._history: deque[dict[str, Any]] = deque(maxlen=max(10, history_size))
        self._requests = 0
        self._errors = 0
        self._routed_requests = 0
        self._inference_calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._latency_seconds = 0.0
        self._neg_steps = 0
        self._neg_activations = 0
        self._neg_guided_steps = 0

    def record(self, candidate: Candidate, model: str, latency_seconds: float) -> None:
        metadata = candidate.metadata or {}
        routing = metadata.get("routing") or {}
        neg = metadata.get("neg") or {}
        calls = max(1, int(routing.get("inference_calls", 1) or 1))
        neg_steps = max(0, int(neg.get("steps", 0) or 0))
        neg_activations = max(0, int(neg.get("activations", 0) or 0))
        neg_guided = max(0, int(neg.get("guided_steps", 0) or 0))
        entry = {
            "timestamp": time.time(),
            "model": model,
            "latency_seconds": max(0.0, float(latency_seconds)),
            "prompt_tokens": max(0, int(candidate.prompt_tokens)),
            "completion_tokens": max(0, int(candidate.completion_tokens)),
            "inference_calls": calls,
            "ensemble": bool(routing.get("ensemble", False)),
            "route_reasons": list(routing.get("reasons") or []),
            "finish_reason": candidate.finish_reason,
            "tool_calls": len(candidate.tool_calls),
            "neg_signal": neg.get("signal") or metadata.get("neg_signal"),
            "neg_steps": neg_steps,
            "neg_activations": neg_activations,
            "neg_activation_rate": float(neg.get("activation_rate", 0.0) or 0.0),
            "neg_guided_steps": neg_guided,
            "neg_eval_ms": float(neg.get("eval_ms", 0.0) or 0.0),
        }
        with self._lock:
            self._requests += 1
            self._routed_requests += int(entry["ensemble"])
            self._inference_calls += calls
            self._prompt_tokens += entry["prompt_tokens"]
            self._completion_tokens += entry["completion_tokens"]
            self._latency_seconds += entry["latency_seconds"]
            self._neg_steps += neg_steps
            self._neg_activations += neg_activations
            self._neg_guided_steps += neg_guided
            self._history.appendleft(entry)

    def record_error(self) -> None:
        with self._lock:
            self._errors += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            requests = self._requests
            latency = self._latency_seconds
            neg_steps = self._neg_steps
            return {
                "started_at": self._started_at,
                "uptime_seconds": max(0.0, time.time() - self._started_at),
                "requests": requests,
                "errors": self._errors,
                "routed_requests": self._routed_requests,
                "routing_rate": self._routed_requests / requests if requests else 0.0,
                "inference_calls": self._inference_calls,
                "average_calls": self._inference_calls / requests if requests else 0.0,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "tokens_per_second": self._completion_tokens / latency if latency else 0.0,
                "average_latency_seconds": latency / requests if requests else 0.0,
                "neg_steps": neg_steps,
                "neg_activations": self._neg_activations,
                "neg_activation_rate": self._neg_activations / neg_steps if neg_steps else 0.0,
                "neg_guided_steps": self._neg_guided_steps,
                "recent": list(self._history),
            }

