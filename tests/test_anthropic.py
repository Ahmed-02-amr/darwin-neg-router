import json
from dataclasses import replace

from fastapi.testclient import TestClient

from darwin_neg_router.anthropic import (
    anthropic_chat_request,
    anthropic_response,
    stream_anthropic_response,
)
from darwin_neg_router.backends import Backend
from darwin_neg_router.config import Settings
from darwin_neg_router.types import Candidate, ChatRequest


class FakeBackend(Backend):
    def __init__(self, responses: list[Candidate]):
        self.responses = responses
        self.requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> Candidate:
        self.requests.append(replace(request))
        return self.responses.pop(0)


def test_translates_anthropic_tools_results_and_system_prompt() -> None:
    body = {
        "model": "darwin-neg-agent",
        "max_tokens": 4096,
        "system": [{"type": "text", "text": "You are a coding agent."}],
        "tools": [
            {
                "name": "Read",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "messages": [
            {"role": "user", "content": "Inspect the project."},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "I should inspect first.", "signature": "x"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Read",
                        "input": {"path": "README.md"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{"type": "text", "text": "Project documentation"}],
                    }
                ],
            },
        ],
    }

    request = anthropic_chat_request(body, 16384)

    assert request.messages[0] == {"role": "system", "content": "You are a coding agent."}
    assert request.messages[2]["reasoning_content"] == "I should inspect first."
    assert request.messages[2]["tool_calls"][0]["function"]["arguments"] == '{"path": "README.md"}'
    assert request.messages[3] == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "content": "Project documentation",
    }
    assert request.tools[0]["function"]["parameters"]["required"] == ["path"]
    assert request.tool_choice == "auto"
    assert request.parallel_tool_calls is False
    assert request.think is True


def test_late_system_message_is_merged_at_the_beginning() -> None:
    request = anthropic_chat_request(
        {
            "model": "darwin-neg-agent",
            "max_tokens": 256,
            "system": "Primary Claude Code contract.",
            "messages": [
                {"role": "user", "content": "hello?"},
                {"role": "system", "content": "Compatibility-layer skill instructions."},
            ],
        },
        1024,
    )

    assert [message["role"] for message in request.messages] == ["system", "user"]
    assert request.messages[0]["content"] == (
        "Primary Claude Code contract.\n\nCompatibility-layer skill instructions."
    )


def test_anthropic_response_and_stream_include_thinking_text_and_tool_use() -> None:
    candidate = Candidate(
        content="I will inspect it.",
        reasoning_content="Repository evidence is needed.",
        tool_calls=[
            {
                "id": "toolu_read_1",
                "type": "function",
                "function": {"name": "Read", "arguments": '{"path":"README.md"}'},
            }
        ],
        finish_reason="tool_calls",
        prompt_tokens=100,
        completion_tokens=20,
    )
    response = anthropic_response(candidate, "darwin-neg-agent", message_id="msg_test")

    assert [block["type"] for block in response["content"]] == [
        "thinking",
        "text",
        "tool_use",
    ]
    assert response["content"][2]["input"] == {"path": "README.md"}
    assert response["stop_reason"] == "tool_use"

    events = list(stream_anthropic_response(response))
    names = [event.splitlines()[0] for event in events]
    assert names[0] == "event: message_start"
    assert "event: content_block_start" in names
    assert "event: content_block_delta" in names
    assert names[-2] == "event: message_delta"
    assert names[-1] == "event: message_stop"
    tool_delta = next(event for event in events if '"type": "input_json_delta"' in event)
    payload = json.loads(tool_delta.split("data: ", 1)[1])
    assert json.loads(payload["delta"]["partial_json"]) == {"path": "README.md"}


def test_messages_endpoint_accepts_anthropic_api_key_and_streams(monkeypatch) -> None:
    backend = FakeBackend(
        [
            Candidate(
                reasoning_content="Use the requested skill.",
                tool_calls=[
                    {
                        "id": "toolu_skill_1",
                        "type": "function",
                        "function": {"name": "Skill", "arguments": "{}"},
                    }
                ],
                finish_reason="tool_calls",
                prompt_tokens=12,
                completion_tokens=8,
            )
        ]
    )
    monkeypatch.setattr("darwin_neg_router.server.build_backend", lambda *_args, **_kwargs: backend)
    from darwin_neg_router.server import create_app

    app = create_app(Settings(api_key="local-secret"))
    client = TestClient(app)
    response = client.post(
        "/v1/messages",
        headers={
            "x-api-key": "local-secret",
            "anthropic-version": "2023-06-01",
        },
        json={
            "model": "darwin-neg-agent",
            "max_tokens": 1024,
            "stream": True,
            "tools": [
                {
                    "name": "Skill",
                    "description": "List or load skills",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            "messages": [{"role": "user", "content": "What skills are available?"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: message_start" in response.text
    assert '"type": "tool_use"' in response.text
    assert '"name": "Skill"' in response.text
    assert "event: message_stop" in response.text
    assert backend.requests[0].tools[0]["function"]["name"] == "Skill"


def test_count_tokens_endpoint_is_available_to_anthropic_clients(monkeypatch) -> None:
    backend = FakeBackend([])
    monkeypatch.setattr("darwin_neg_router.server.build_backend", lambda *_args, **_kwargs: backend)
    from darwin_neg_router.server import create_app

    client = TestClient(create_app(Settings()))
    response = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "darwin-neg-agent",
            "messages": [{"role": "user", "content": "Count this input."}],
        },
    )
    assert response.status_code == 200
    assert response.json()["input_tokens"] > 0
