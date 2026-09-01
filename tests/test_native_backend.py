import json

from darwin_neg_router.backends import OpenAIBackend
from darwin_neg_router.types import ChatRequest


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return json.dumps(
            {
                "choices": [
                    {
                        "message": {"content": "ok", "reasoning_content": "checked"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                "neg": {
                    "signal": "released_hidden_state_head",
                    "steps": 2,
                    "activations": 1,
                    "activation_rate": 0.5,
                    "threshold": 1.175187349319458,
                },
            }
        ).encode()


def test_native_backend_uses_released_head_telemetry(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    backend = OpenAIBackend(
        "http://127.0.0.1:11436/v1",
        "",
        "darwin-9b-neg-native",
        native_neg=True,
    )
    result = backend.chat(
        ChatRequest(
            messages=[{"role": "user", "content": "hi"}],
            model="test",
            tools=[{"type": "function", "function": {"name": "read_file"}}],
            tool_choice="auto",
            parallel_tool_calls=False,
            repeat_penalty=1.05,
        )
    )

    assert "logprobs" not in captured["body"]
    assert captured["body"]["tool_choice"] == "auto"
    assert captured["body"]["parallel_tool_calls"] is False
    assert captured["body"]["repeat_penalty"] == 1.05
    assert result.metadata["backend"] == "native-neg"
    assert result.metadata["neg_mode"] == "released_head"
    assert result.metadata["neg"]["activation_rate"] == 0.5
