from __future__ import annotations

import json
import math
import threading
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .neg import NEGEntropyMonitor, NEGLogitsProcessor
from .tool_parser import parse_tool_calls, split_thinking
from .types import Candidate, ChatRequest, normalize_system_messages


class Backend(ABC):
    @abstractmethod
    def chat(self, request: ChatRequest) -> Candidate:
        raise NotImplementedError


class OllamaBackend(Backend):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat(self, request: ChatRequest) -> Candidate:
        options: dict[str, Any] = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "num_predict": request.max_tokens,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "repeat_penalty": request.repeat_penalty,
        }
        if request.seed is not None:
            options["seed"] = request.seed
        use_logprobs = not request.tools
        body: dict[str, Any] = {
            "model": self.model,
            "messages": normalize_system_messages(request.messages),
            "stream": False,
            "think": request.think,
            # Ollama's native API exposes token probabilities even though its
            # OpenAI-compatible endpoint does not.  These are the only useful
            # uncertainty observations available without modifying Ollama's
            # generation graph to expose the released hidden-state NEG head.
            "logprobs": use_logprobs,
            "options": options,
        }
        if use_logprobs:
            body["top_logprobs"] = 20
        if request.tools:
            body["tools"] = request.tools
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=1800) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        message = payload.get("message", {})
        tool_calls = _normalize_ollama_tool_calls(message.get("tool_calls", []))
        stats = _entropy_stats_from_tokens(payload.get("logprobs", []))
        return Candidate(
            content=message.get("content", ""),
            reasoning_content=message.get("thinking", ""),
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            prompt_tokens=int(payload.get("prompt_eval_count", 0)),
            completion_tokens=int(payload.get("eval_count", 0)),
            metadata={
                "backend": "ollama",
                "model": self.model,
                "neg": stats,
                "neg_threshold": 1.175187349319458,
                "neg_signal": "ollama_top20_distribution_entropy",
                "neg_mode": "runtime_surrogate" if use_logprobs else "unavailable_with_tools",
            },
        )


class OpenAIBackend(Backend):
    """Backend for vLLM/TensorRT-LLM and other OpenAI-compatible servers."""

    def __init__(
        self, base_url: str, api_key: str, model: str, *, native_neg: bool = False
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.native_neg = native_neg

    def chat(self, request: ChatRequest) -> Candidate:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": normalize_system_messages(request.messages),
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "max_tokens": request.max_tokens,
            "seed": request.seed,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": request.think},
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "repeat_penalty": request.repeat_penalty,
        }
        if not self.native_neg:
            body["logprobs"] = True
            body["top_logprobs"] = 20
        if request.tools:
            body["tools"] = request.tools
            if request.tool_choice is not None:
                body["tool_choice"] = request.tool_choice
            if request.parallel_tool_calls is not None:
                body["parallel_tool_calls"] = request.parallel_tool_calls
        if request.stop is not None:
            body["stop"] = request.stop
        # A router-facing benchmark/client may explicitly select one of the
        # general router's task policies. Never forward this gateway extension
        # into the native llama.cpp server used behind the router.
        if request.routing_profile and not self.native_neg:
            body["darwin"] = {"routing_profile": request.routing_profile}
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=1800) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Upstream OpenAI request failed ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Upstream OpenAI request failed: {exc}") from exc

        choice = payload["choices"][0]
        message = choice.get("message", {})
        usage = payload.get("usage", {})
        released_neg = payload.get("neg") if self.native_neg else None
        stats = released_neg or _entropy_stats(choice.get("logprobs"))
        return Candidate(
            content=message.get("content") or "",
            reasoning_content=message.get("reasoning_content") or "",
            tool_calls=_normalize_openai_tool_calls(message.get("tool_calls", [])),
            finish_reason=choice.get("finish_reason") or "stop",
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            metadata={
                "backend": "native-neg" if self.native_neg else "openai-compatible",
                "model": self.model,
                "neg": stats,
                "neg_threshold": float(stats.get("threshold", 1.175187349319458)),
                "neg_signal": stats.get(
                    "signal",
                    "top20_distribution_entropy",
                ),
                "neg_mode": "released_head" if released_neg else "runtime_surrogate",
                "darwin": payload.get("darwin", {}),
            },
        )


class TransformersNEGBackend(Backend):
    def __init__(self, model_id: str, *, load_in_4bit: bool = True, max_context: int = 163840):
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit
        self.max_context = max_context
        self._lock = threading.Lock()
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA PyTorch is required for the 9B NEG checkpoint. Run scripts/setup-cuda.ps1 first."
            )
        kwargs: dict[str, Any] = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }
        if self.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.truncation_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs).eval()
        neg_path = hf_hub_download(self.model_id, "neg_modules.safetensors")
        self.neg = NEGEntropyMonitor(self.model, Path(neg_path))
        self._loaded = True

    def chat(self, request: ChatRequest) -> Candidate:
        with self._lock:
            self._load()
            return self._chat_locked(request)

    def _chat_locked(self, request: ChatRequest) -> Candidate:
        import torch
        from transformers import LogitsProcessorList

        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
            "enable_thinking": request.think,
        }
        if request.tools:
            kwargs["tools"] = request.tools
        inputs = self.tokenizer.apply_chat_template(normalize_system_messages(request.messages), **kwargs)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        if input_ids.shape[1] + request.max_tokens > self.max_context:
            raise ValueError(
                f"Prompt ({input_ids.shape[1]} tokens) plus output allowance "
                f"({request.max_tokens}) exceeds DARWIN_MAX_CONTEXT={self.max_context}. "
                "Compact the CodePilot session or lower max_tokens."
            )
        input_ids = input_ids.to(self.model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        sampled = request.temperature > 0
        self.neg.reset()
        generation: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": request.max_tokens,
            "do_sample": sampled,
            "logits_processor": LogitsProcessorList(
                [NEGLogitsProcessor(self.neg, sampled=sampled)]
            ),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if sampled:
            generation.update(
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=max(1, request.top_k),
            )
        if request.seed is not None:
            torch.manual_seed(request.seed)
            torch.cuda.manual_seed_all(request.seed)
        with torch.inference_mode():
            output = self.model.generate(**generation)
        new_tokens = output[0, input_ids.shape[1] :]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        reasoning, visible = split_thinking(raw)
        visible, tool_calls = parse_tool_calls(visible)
        return Candidate(
            content=visible,
            reasoning_content=reasoning,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            prompt_tokens=int(input_ids.shape[1]),
            completion_tokens=int(new_tokens.shape[0]),
            metadata={
                "backend": "transformers-neg",
                "model": self.model_id,
                "neg": self.neg.stats.as_dict(),
                "neg_threshold": self.neg.threshold,
                "neg_top_k": self.neg.top_k,
                "neg_temperature_scale": self.neg.temperature_scale,
            },
        )


def _normalize_ollama_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        function = call.get("function", {})
        arguments = function.get("arguments", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        normalized.append(
            {
                "id": call.get("id", f"call_ollama_{index}"),
                "type": "function",
                "function": {
                    "name": function.get("name", ""),
                    "arguments": arguments,
                },
            }
        )
    return normalized


def _normalize_openai_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        function = call.get("function", {})
        arguments = function.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        normalized.append(
            {
                "id": call.get("id", f"call_upstream_{index}"),
                "type": "function",
                "function": {
                    "name": function.get("name", ""),
                    "arguments": arguments,
                },
            }
        )
    return normalized


def _entropy_stats(logprobs: dict[str, Any] | None) -> dict[str, float | int | str]:
    """Estimate full-distribution entropy from top-20 log probabilities.

    The unreported probability mass is treated as one bucket, making this a
    conservative lower bound. It is still a direct uncertainty signal and is
    more actionable than the released scalar gate's no-op greedy top-k mask.
    """
    return _entropy_stats_from_tokens((logprobs or {}).get("content", []) or [])


def _entropy_stats_from_tokens(
    tokens: list[dict[str, Any]] | None,
) -> dict[str, float | int | str]:
    """Estimate entropy for Ollama- or OpenAI-shaped top-20 token data."""
    threshold = 1.175187349319458
    values: list[float] = []
    for token in tokens or []:
        probabilities = [
            math.exp(float(item["logprob"]))
            for item in token.get("top_logprobs", [])
            if item.get("logprob") is not None and math.isfinite(float(item["logprob"]))
        ]
        reported_mass = min(1.0, sum(probabilities))
        missing_mass = max(0.0, 1.0 - reported_mass)
        entropy = -sum(p * math.log(max(p, 1e-30)) for p in probabilities)
        if missing_mass:
            entropy -= missing_mass * math.log(missing_mass)
        values.append(entropy)
    activations = sum(value > threshold for value in values)
    return {
        "source": "top20_lower_bound",
        "steps": len(values),
        "activations": activations,
        "activation_rate": activations / len(values) if values else 0.0,
        "entropy_mean": sum(values) / len(values) if values else 0.0,
        "entropy_max": max(values, default=0.0),
    }
