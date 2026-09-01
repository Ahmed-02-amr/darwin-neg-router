from __future__ import annotations

import os
from dataclasses import dataclass


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 11435
    primary_backend: str = "ollama"
    primary_model: str = "darwin-9b-neg:agent"
    verifier_backend: str = "ollama"
    verifier_model: str = "ornith-1.5:agent"
    ollama_url: str = "http://127.0.0.1:11434"
    upstream_url: str = "http://127.0.0.1:8000/v1"
    upstream_api_key: str = "EMPTY"
    native_url: str = "http://127.0.0.1:11436/v1"
    native_model: str = "darwin-9b-neg-native"
    hf_model: str = "FINAL-Bench/Darwin-9B-NEG"
    max_context: int = 65536
    default_max_tokens: int = 16384
    candidates: int = 3
    max_ensemble_inferences: int = 20
    candidate_temperature: float = 0.45
    complexity_threshold: int = 3
    neg_activation_threshold: float = 0.05
    neg_min_activations: int = 16
    route_tool_calls: bool = True
    tool_phase_max_tokens: int = 4096
    max_parallel_tool_calls: int = 32
    unchanged_tool_result_limit: int = 2
    api_key: str = ""
    load_in_4bit: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            host=os.getenv("DARWIN_HOST", cls.host),
            port=int(os.getenv("DARWIN_PORT", cls.port)),
            primary_backend=os.getenv("DARWIN_PRIMARY_BACKEND", cls.primary_backend).lower(),
            primary_model=os.getenv("DARWIN_PRIMARY_MODEL", cls.primary_model),
            verifier_backend=os.getenv("DARWIN_VERIFIER_BACKEND", cls.verifier_backend).lower(),
            verifier_model=os.getenv("DARWIN_VERIFIER_MODEL", cls.verifier_model),
            ollama_url=os.getenv("OLLAMA_URL", cls.ollama_url).rstrip("/"),
            upstream_url=os.getenv("DARWIN_UPSTREAM_URL", cls.upstream_url).rstrip("/"),
            upstream_api_key=os.getenv("DARWIN_UPSTREAM_API_KEY", cls.upstream_api_key),
            native_url=os.getenv("DARWIN_NATIVE_URL", cls.native_url).rstrip("/"),
            native_model=os.getenv("DARWIN_NATIVE_MODEL", cls.native_model),
            hf_model=os.getenv("DARWIN_HF_MODEL", cls.hf_model),
            max_context=int(os.getenv("DARWIN_MAX_CONTEXT", cls.max_context)),
            default_max_tokens=int(os.getenv("DARWIN_MAX_TOKENS", cls.default_max_tokens)),
            candidates=max(1, int(os.getenv("DARWIN_CANDIDATES", cls.candidates))),
            max_ensemble_inferences=max(
                2,
                min(20, int(os.getenv("DARWIN_MAX_ENSEMBLE_INFERENCES", cls.max_ensemble_inferences))),
            ),
            candidate_temperature=float(
                os.getenv("DARWIN_CANDIDATE_TEMPERATURE", cls.candidate_temperature)
            ),
            complexity_threshold=int(
                os.getenv("DARWIN_COMPLEXITY_THRESHOLD", cls.complexity_threshold)
            ),
            neg_activation_threshold=float(
                os.getenv("DARWIN_NEG_ACTIVATION_THRESHOLD", cls.neg_activation_threshold)
            ),
            neg_min_activations=max(
                1, int(os.getenv("DARWIN_NEG_MIN_ACTIVATIONS", cls.neg_min_activations))
            ),
            route_tool_calls=_bool("DARWIN_ROUTE_TOOL_CALLS", cls.route_tool_calls),
            tool_phase_max_tokens=max(
                512, int(os.getenv("DARWIN_TOOL_PHASE_MAX_TOKENS", cls.tool_phase_max_tokens))
            ),
            max_parallel_tool_calls=max(
                1, int(os.getenv("DARWIN_MAX_PARALLEL_TOOL_CALLS", cls.max_parallel_tool_calls))
            ),
            unchanged_tool_result_limit=max(
                2,
                int(
                    os.getenv(
                        "DARWIN_UNCHANGED_TOOL_RESULT_LIMIT",
                        cls.unchanged_tool_result_limit,
                    )
                ),
            ),
            api_key=os.getenv("DARWIN_API_KEY", ""),
            load_in_4bit=_bool("DARWIN_LOAD_IN_4BIT", cls.load_in_4bit),
        )
