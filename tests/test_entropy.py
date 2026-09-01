import math
import json

from darwin_neg_router.backends import OllamaBackend, OpenAIBackend, _entropy_stats
from darwin_neg_router.types import ChatRequest


def test_entropy_routes_flat_distribution() -> None:
    flat = math.log(0.05)
    stats = _entropy_stats(
        {"content": [{"top_logprobs": [{"logprob": flat} for _ in range(20)]}]}
    )
    assert stats["activations"] == 1
    assert stats["entropy_mean"] > 2.9


def test_entropy_does_not_route_confident_distribution() -> None:
    stats = _entropy_stats(
        {
            "content": [
                {
                    "top_logprobs": [
                        {"logprob": math.log(0.99)},
                        {"logprob": math.log(0.01)},
                    ]
                }
            ]
        }
    )
    assert stats["activations"] == 0
    assert stats["entropy_mean"] < 0.1


def test_openai_backend_converts_response_and_entropy(monkeypatch) -> None:
    payload = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": "",
                    "reasoning_content": "inspect it",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": {"path": "a.py"}},
                        }
                    ],
                },
                "logprobs": {
                    "content": [
                        {"top_logprobs": [{"logprob": math.log(0.05)} for _ in range(20)]}
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(payload).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    backend = OpenAIBackend("http://localhost:8000/v1", "EMPTY", "darwin")
    result = backend.chat(
        ChatRequest(messages=[{"role": "user", "content": "inspect"}], model="darwin")
    )
    assert result.reasoning_content == "inspect it"
    assert result.tool_calls[0]["function"]["arguments"] == '{"path": "a.py"}'
    assert result.metadata["neg"]["activations"] == 1


def test_ollama_backend_uses_native_logprobs_for_entropy(monkeypatch) -> None:
    payload = {
        "message": {"role": "assistant", "content": "answer", "thinking": "reason"},
        "logprobs": [
            {"top_logprobs": [{"token": str(i), "logprob": math.log(0.05)} for i in range(20)]}
        ],
        "prompt_eval_count": 12,
        "eval_count": 3,
    }
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(payload).encode()

    def urlopen(request, **_kwargs):
        captured.update(json.loads(request.data))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    result = OllamaBackend("http://localhost:11434", "darwin").chat(
        ChatRequest(messages=[{"role": "user", "content": "solve"}], model="darwin")
    )
    assert captured["logprobs"] is True
    assert captured["top_logprobs"] == 20
    assert result.metadata["neg"]["activations"] == 1
    assert result.metadata["neg_mode"] == "runtime_surrogate"
