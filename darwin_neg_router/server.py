from __future__ import annotations

import argparse
import json
import time
import uuid
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

from .anthropic import (
    anthropic_chat_request,
    anthropic_response,
    estimate_input_tokens,
    stream_anthropic_response,
)
from .backends import Backend, OllamaBackend, OpenAIBackend, TransformersNEGBackend
from .config import Settings
from .gpqa import GPQAEnsembler
from .router import SelectiveRouter, last_user_text
from .routing_policy import VALID_ROUTING_PROFILES
from .telemetry import TelemetryStore
from .types import Candidate, ChatRequest


def build_backend(kind: str, model: str, settings: Settings) -> Backend:
    if kind == "ollama":
        return OllamaBackend(settings.ollama_url, model)
    if kind in {"openai", "vllm", "nvfp4"}:
        return OpenAIBackend(settings.upstream_url, settings.upstream_api_key, model)
    if kind in {"native", "native-neg"}:
        native_model = settings.native_model if model in {"", "auto"} else model
        return OpenAIBackend(
            settings.native_url,
            settings.upstream_api_key,
            native_model,
            native_neg=True,
        )
    if kind in {"transformers", "neg"}:
        return TransformersNEGBackend(
            settings.hf_model, load_in_4bit=settings.load_in_4bit, max_context=settings.max_context
        )
    raise ValueError(f"Unsupported backend: {kind}")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    primary = build_backend(settings.primary_backend, settings.primary_model, settings)
    verifier = build_backend(settings.verifier_backend, settings.verifier_model, settings)
    router = SelectiveRouter(
        primary,
        verifier,
        candidate_count=settings.candidates,
        candidate_temperature=settings.candidate_temperature,
        complexity_threshold=settings.complexity_threshold,
        neg_activation_threshold=settings.neg_activation_threshold,
        neg_min_activations=settings.neg_min_activations,
        route_tool_calls=settings.route_tool_calls,
        review_max_tokens=settings.review_max_tokens,
        truncation_recovery_tokens=settings.truncation_recovery_tokens,
        tool_phase_max_tokens=settings.tool_phase_max_tokens,
        max_parallel_tool_calls=settings.max_parallel_tool_calls,
        unchanged_tool_result_limit=settings.unchanged_tool_result_limit,
    )
    gpqa = GPQAEnsembler(
        primary,
        primary,
        solver_tokens=settings.gpqa_solver_tokens,
        review_tokens=settings.gpqa_review_tokens,
    )
    app = FastAPI(title="Darwin NEG Router", version="0.4.4")
    app.state.settings = settings
    app.state.router = router
    app.state.gpqa = gpqa
    app.state.telemetry = TelemetryStore()

    def authorize(authorization: str | None, x_api_key: str | None = None) -> None:
        if not settings.api_key:
            return
        if authorization != f"Bearer {settings.api_key}" and x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

    def run_request(request: ChatRequest, requested_model: str, ensemble_option: Any) -> Candidate:
        if requested_model in {"darwin-neg-gpqa", "darwin-neg-gpqa20"}:
            question = last_user_text(request.messages)
            if not question:
                raise HTTPException(status_code=400, detail="GPQA profile requires a user question")
            try:
                return gpqa.solve(
                    question,
                    mode="full20" if requested_model.endswith("gpqa20") else "adaptive20",
                    seed=int(request.seed or 0),
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        explicit_budget: int | None = None
        if requested_model in {"darwin-neg-ensemble20", "darwin-neg-agent20"}:
            explicit_budget = settings.max_ensemble_inferences
        elif isinstance(ensemble_option, int) and not isinstance(ensemble_option, bool):
            explicit_budget = max(4, min(settings.max_ensemble_inferences, ensemble_option))
        elif isinstance(ensemble_option, str) and ensemble_option.lower() in {"20", "20x", "full"}:
            explicit_budget = settings.max_ensemble_inferences
        force = bool(ensemble_option) or explicit_budget is not None
        return router.chat(
            request,
            force_ensemble=force,
            total_inferences=explicit_budget,
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "protocols": ["openai-chat-completions", "anthropic-messages"],
            "anthropic_base_url": f"http://{settings.host}:{settings.port}",
            "primary_backend": settings.primary_backend,
            "primary_model": settings.primary_model,
            "verifier_model": settings.verifier_model,
            "true_neg": settings.primary_backend in {"transformers", "neg", "native", "native-neg"},
            "entropy_routing": settings.primary_backend in {
                "ollama",
                "transformers",
                "neg",
                "openai",
                "vllm",
                "nvfp4",
                "native",
                "native-neg",
            },
            "tool_guard": {
                "tool_phase_max_tokens": settings.tool_phase_max_tokens,
                "max_parallel_tool_calls": settings.max_parallel_tool_calls,
                "unchanged_result_limit": settings.unchanged_tool_result_limit,
                "long_form_max_tokens": settings.default_max_tokens,
            },
            "review_max_tokens": settings.review_max_tokens,
            "truncation_recovery": {
                "enabled": True,
                "max_tokens": settings.truncation_recovery_tokens,
                "max_attempts_per_generation": 1,
            },
            "adaptive_voter_weighting": {
                "enabled": True,
                "profiles": sorted(VALID_ROUTING_PROFILES),
                "verifier_blinded_to_sampling": True,
            },
            "gpqa": {
                "solver_max_tokens": settings.gpqa_solver_tokens,
                "review_max_tokens": settings.gpqa_review_tokens,
            },
        }

    @app.get("/v1/models")
    def models(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(authorization, x_api_key)
        created = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": "Darwin-NEG",
                    "object": "model",
                    "type": "model",
                    "display_name": "Darwin NEG",
                    "created": created,
                    "created_at": "2026-09-01T00:00:00Z",
                    "owned_by": "local",
                },
                {
                    "id": "darwin-neg-auto",
                    "object": "model",
                    "type": "model",
                    "display_name": "Darwin NEG Auto",
                    "created": created,
                    "created_at": "2026-09-01T00:00:00Z",
                    "owned_by": "local",
                },
                {
                    "id": "darwin-neg-agent",
                    "object": "model",
                    "type": "model",
                    "display_name": "Darwin NEG Agent",
                    "created": created,
                    "created_at": "2026-09-01T00:00:00Z",
                    "owned_by": "local",
                },
                {
                    "id": "darwin-neg-ensemble20",
                    "object": "model",
                    "type": "model",
                    "display_name": "Darwin NEG Ensemble 20",
                    "created": created,
                    "created_at": "2026-09-01T00:00:00Z",
                    "owned_by": "local",
                },
                {
                    "id": "darwin-neg-agent20",
                    "object": "model",
                    "type": "model",
                    "display_name": "Darwin NEG Agent 20",
                    "created": created,
                    "created_at": "2026-09-01T00:00:00Z",
                    "owned_by": "local",
                },
                {
                    "id": "darwin-neg-gpqa",
                    "object": "model",
                    "type": "model",
                    "display_name": "Darwin NEG GPQA",
                    "created": created,
                    "created_at": "2026-09-01T00:00:00Z",
                    "owned_by": "local",
                },
                {
                    "id": "darwin-neg-gpqa20",
                    "object": "model",
                    "type": "model",
                    "display_name": "Darwin NEG GPQA 20",
                    "created": created,
                    "created_at": "2026-09-01T00:00:00Z",
                    "owned_by": "local",
                },
            ],
            "has_more": False,
            "first_id": "Darwin-NEG",
            "last_id": "darwin-neg-gpqa20",
        }

    @app.get("/telemetry")
    def telemetry(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        authorize(authorization)
        return app.state.telemetry.snapshot()

    @app.post("/v1/chat/completions")
    def chat_completions(
        body: dict[str, Any],
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Any:
        authorize(authorization, x_api_key)
        request_started = time.perf_counter()
        requested_model = str(body.get("model", "darwin-neg-auto"))
        request = ChatRequest(
            messages=body.get("messages", []),
            model=requested_model,
            tools=body.get("tools", []),
            temperature=float(body.get("temperature", 0.0) or 0.0),
            top_p=float(body.get("top_p", 1.0) or 1.0),
            top_k=max(1, int(body.get("top_k", 1) or 1)),
            max_tokens=min(
                int(body.get("max_tokens") or body.get("max_completion_tokens") or settings.default_max_tokens),
                settings.default_max_tokens,
            ),
            stop=body.get("stop"),
            seed=body.get("seed"),
            think=bool(body.get("think", True)),
            tool_choice=body.get("tool_choice"),
            parallel_tool_calls=body.get("parallel_tool_calls"),
            presence_penalty=float(body.get("presence_penalty", 0.0) or 0.0),
            frequency_penalty=float(body.get("frequency_penalty", 0.0) or 0.0),
            repeat_penalty=float(body.get("repeat_penalty", 1.0) or 1.0),
            routing_profile=_routing_profile(body),
        )
        try:
            result = run_request(request, requested_model, body.get("ensemble", False))
        except Exception:
            app.state.telemetry.record_error()
            raise
        app.state.telemetry.record(result, requested_model, time.perf_counter() - request_started)
        response = _openai_response(result, requested_model)
        if body.get("stream", False):
            return StreamingResponse(
                _stream_response(response), media_type="text/event-stream"
            )
        return response

    @app.post("/v1/messages")
    def messages(
        body: dict[str, Any],
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Any:
        """Anthropic Messages compatibility endpoint for Claude Code clients."""

        authorize(authorization, x_api_key)
        request_started = time.perf_counter()
        requested_model = str(body.get("model") or "darwin-neg-agent")
        if not isinstance(body.get("messages"), list):
            raise HTTPException(status_code=400, detail="messages must be an array")
        if "max_tokens" not in body or int(body.get("max_tokens") or 0) <= 0:
            raise HTTPException(status_code=400, detail="max_tokens must be a positive integer")
        request = anthropic_chat_request(body, settings.default_max_tokens)
        request.routing_profile = _routing_profile(body)
        try:
            result = run_request(request, requested_model, body.get("ensemble", False))
        except Exception:
            app.state.telemetry.record_error()
            raise
        app.state.telemetry.record(result, requested_model, time.perf_counter() - request_started)
        response = anthropic_response(result, requested_model)
        if body.get("stream", False):
            return StreamingResponse(
                stream_anthropic_response(response), media_type="text/event-stream"
            )
        return response

    @app.post("/v1/messages/count_tokens")
    def count_tokens(
        body: dict[str, Any],
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> dict[str, int]:
        authorize(authorization, x_api_key)
        return {"input_tokens": estimate_input_tokens(body)}

    return app


def _routing_profile(body: dict[str, Any]) -> str | None:
    darwin = body.get("darwin") if isinstance(body.get("darwin"), dict) else {}
    metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
    value = (
        darwin.get("routing_profile")
        or metadata.get("darwin_routing_profile")
        or body.get("routing_profile")
    )
    if value is None or not str(value).strip():
        return None
    profile = str(value).strip().lower()
    if profile not in VALID_ROUTING_PROFILES:
        valid = ", ".join(sorted(VALID_ROUTING_PROFILES))
        raise HTTPException(status_code=400, detail=f"routing_profile must be one of: {valid}")
    return profile


def _openai_response(candidate: Candidate, model: str = "darwin-neg-auto") -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": candidate.content}
    if candidate.reasoning_content:
        message["reasoning_content"] = candidate.reasoning_content
    if candidate.tool_calls:
        message["tool_calls"] = candidate.tool_calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": candidate.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": candidate.prompt_tokens,
            "completion_tokens": candidate.completion_tokens,
            "total_tokens": candidate.prompt_tokens + candidate.completion_tokens,
        },
        "darwin": candidate.metadata,
    }


def _stream_response(response: dict[str, Any]):
    choice = response["choices"][0]
    message = choice["message"]
    delta: dict[str, Any] = {"role": "assistant"}
    if message.get("content") is not None:
        delta["content"] = message.get("content") or ""
    if message.get("reasoning_content"):
        delta["reasoning_content"] = message["reasoning_content"]
    if message.get("tool_calls"):
        delta["tool_calls"] = [
            {"index": index, **tool_call}
            for index, tool_call in enumerate(message["tool_calls"])
        ]
    chunk = {
        "id": response["id"],
        "object": "chat.completion.chunk",
        "created": response["created"],
        "model": response["model"],
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    done = {
        **chunk,
        "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}],
    }
    yield f"data: {json.dumps(done)}\n\n"
    yield "data: [DONE]\n\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Darwin NEG routing gateway")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    settings = Settings.from_env()
    import uvicorn

    uvicorn.run(
        create_app(settings),
        host=args.host or settings.host,
        port=args.port or settings.port,
    )


if __name__ == "__main__":
    main()
