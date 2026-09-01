import json

from darwin_neg_router.server import _openai_response, _stream_response
from darwin_neg_router.types import Candidate


def test_streaming_tool_calls_include_openai_index() -> None:
    response = _openai_response(
        Candidate(
            reasoning_content="I should inspect the skill list.",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "Skill", "arguments": "{}"},
                }
            ],
            finish_reason="tool_calls",
        ),
        "Darwin-NEG",
    )
    events = list(_stream_response(response))
    first = json.loads(events[0].removeprefix("data: "))
    final = json.loads(events[1].removeprefix("data: "))

    call = first["choices"][0]["delta"]["tool_calls"][0]
    assert call["index"] == 0
    assert call["function"] == {"name": "Skill", "arguments": "{}"}
    assert first["choices"][0]["delta"]["reasoning_content"]
    assert final["choices"][0]["finish_reason"] == "tool_calls"
    assert events[-1] == "data: [DONE]\n\n"
