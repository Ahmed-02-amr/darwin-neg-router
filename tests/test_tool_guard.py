from dataclasses import replace

from darwin_neg_router.backends import Backend
from darwin_neg_router.router import SelectiveRouter
from darwin_neg_router.types import Candidate, ChatRequest


class FakeBackend(Backend):
    def __init__(self, responses: list[Candidate]):
        self.responses = responses
        self.requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> Candidate:
        self.requests.append(replace(request))
        return self.responses.pop(0)


def call(call_id: str, name: str, arguments: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def tool_request(
    messages: list[dict] | None = None,
    *,
    parallel: bool | None = None,
) -> ChatRequest:
    return ChatRequest(
        messages=messages or [{"role": "user", "content": "Use the tools."}],
        model="test",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "WebFetch",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Search",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Poll",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
        max_tokens=16384,
        parallel_tool_calls=parallel,
    )


def test_duplicate_burst_is_collapsed_without_shrinking_client_allowance() -> None:
    repeated = [
        call(f"call_{index}", "WebFetch", '{"url":"https://example.com"}')
        for index in range(780)
    ]
    backend = FakeBackend(
        [
            Candidate(
                tool_calls=repeated,
                finish_reason="length",
                completion_tokens=16384,
            )
        ]
    )

    result = SelectiveRouter(backend, FakeBackend([])).chat(tool_request())

    assert len(result.tool_calls) == 1
    assert result.finish_reason == "tool_calls"
    assert result.metadata["tool_guard"]["raw_tool_calls"] == 780
    assert result.metadata["tool_guard"]["duplicates_removed"] == 779
    assert backend.requests[0].max_tokens == 4096
    assert result.metadata["routing"]["inference_calls"] == 1


def test_many_distinct_parallel_actions_are_preserved_in_order() -> None:
    actions = [
        call(f"call_{index}", "Read", f'{{"path":"src/file_{index}.py"}}')
        for index in range(24)
    ]
    backend = FakeBackend([Candidate(tool_calls=actions)])

    result = SelectiveRouter(backend, FakeBackend([])).chat(
        tool_request(parallel=True)
    )

    assert result.tool_calls == actions
    assert result.metadata["tool_guard"]["parallel_overflow_removed"] == 0


def test_explicit_disable_parallel_tools_enforces_one_action() -> None:
    actions = [
        call("call_1", "Read", '{"path":"a.py"}'),
        call("call_2", "Read", '{"path":"b.py"}'),
        call("call_3", "Read", '{"path":"c.py"}'),
    ]
    backend = FakeBackend([Candidate(tool_calls=actions)])

    result = SelectiveRouter(backend, FakeBackend([])).chat(
        tool_request(parallel=False)
    )

    assert result.tool_calls == actions[:1]
    assert result.metadata["tool_guard"]["parallel_limit"] == 1
    assert result.metadata["tool_guard"]["parallel_overflow_removed"] == 2


def test_third_unchanged_action_is_recovered_with_a_different_tool() -> None:
    repeated = call("call_3", "WebFetch", '{"url":"https://blocked.example"}')
    alternate = call("call_4", "Search", '{"query":"gold spot price"}')
    messages = [
        {"role": "user", "content": "Find the current gold price."},
        {"role": "assistant", "content": "", "tool_calls": [
            call("call_1", "WebFetch", '{"url":"https://blocked.example"}')
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "403 Forbidden"},
        {"role": "assistant", "content": "", "tool_calls": [
            call("call_2", "WebFetch", '{"url":"https://blocked.example"}')
        ]},
        {"role": "tool", "tool_call_id": "call_2", "content": "403 Forbidden"},
    ]
    backend = FakeBackend(
        [
            Candidate(tool_calls=[repeated], prompt_tokens=100, completion_tokens=20),
            Candidate(tool_calls=[alternate], prompt_tokens=110, completion_tokens=25),
        ]
    )

    result = SelectiveRouter(backend, FakeBackend([])).chat(tool_request(messages))

    assert result.tool_calls == [alternate]
    assert len(backend.requests) == 2
    assert "Tool-loop recovery" in backend.requests[1].messages[0]["content"]
    assert result.metadata["tool_guard"]["stalled_calls_blocked"] == 1
    assert result.metadata["tool_guard"]["recovery_inferences"] == 1
    assert result.metadata["routing"]["inference_calls"] == 2
    assert result.prompt_tokens == 210
    assert result.completion_tokens == 45


def test_same_polling_action_remains_available_when_results_change() -> None:
    repeated = call("call_3", "Poll", '{"job_id":"build-1"}')
    messages = [
        {"role": "user", "content": "Wait for the build."},
        {"role": "assistant", "content": "", "tool_calls": [
            call("call_1", "Poll", '{"job_id":"build-1"}')
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "running: 10%"},
        {"role": "assistant", "content": "", "tool_calls": [
            call("call_2", "Poll", '{"job_id":"build-1"}')
        ]},
        {"role": "tool", "tool_call_id": "call_2", "content": "running: 80%"},
    ]
    backend = FakeBackend([Candidate(tool_calls=[repeated])])

    result = SelectiveRouter(backend, FakeBackend([])).chat(tool_request(messages))

    assert result.tool_calls == [repeated]
    assert len(backend.requests) == 1


def test_tool_phase_limit_falls_back_to_full_allowance_for_long_form_output() -> None:
    backend = FakeBackend(
        [
            Candidate(
                content="partial long answer",
                finish_reason="length",
                prompt_tokens=100,
                completion_tokens=4096,
            ),
            Candidate(
                content="complete long answer",
                finish_reason="stop",
                prompt_tokens=100,
                completion_tokens=6000,
            ),
        ]
    )

    result = SelectiveRouter(backend, FakeBackend([])).chat(tool_request())

    assert [request.max_tokens for request in backend.requests] == [4096, 16384]
    assert result.content == "complete long answer"
    assert result.prompt_tokens == 200
    assert result.completion_tokens == 10096
    assert result.metadata["tool_guard"]["long_form_retries"] == 1
    assert result.metadata["routing"]["inference_calls"] == 2
