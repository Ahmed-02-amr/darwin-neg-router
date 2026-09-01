from dataclasses import replace

import pytest

from darwin_neg_router.backends import Backend
from darwin_neg_router.router import SelectiveRouter, complexity_score
from darwin_neg_router.types import Candidate, ChatRequest


class FakeBackend(Backend):
    def __init__(self, responses: list[Candidate]):
        self.responses = responses
        self.requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> Candidate:
        self.requests.append(replace(request))
        return self.responses.pop(0)


def request(text: str) -> ChatRequest:
    return ChatRequest(messages=[{"role": "user", "content": text}], model="test")


def test_simple_request_stays_single_pass() -> None:
    primary = FakeBackend([Candidate(content="hello")])
    verifier = FakeBackend([])
    result = SelectiveRouter(primary, verifier).chat(request("Say hello"))
    assert result.content == "hello"
    assert result.metadata["routing"]["ensemble"] is False
    assert len(primary.requests) == 1


def test_complex_request_generates_candidates_and_uses_verifier() -> None:
    primary = FakeBackend(
        [Candidate(content="a"), Candidate(content="b"), Candidate(content="c")]
    )
    verifier = FakeBackend(
        [
            Candidate(content='{"winner": 1, "confidence": 0.9, "reason": "better"}'),
            Candidate(content="refined b"),
        ]
    )
    router = SelectiveRouter(primary, verifier, complexity_threshold=2)
    result = router.chat(request("Implement and debug this multi-file repository architecture."))
    assert result.content == "refined b"
    assert result.metadata["routing"]["winner"] == 1
    assert [r.temperature for r in primary.requests] == [0.0, 0.45, 0.55]
    assert [r.top_k for r in primary.requests] == [1, 40, 40]
    assert "repository_mapper" in primary.requests[1].messages[0]["content"]
    assert "implementation_engineer" in primary.requests[2].messages[0]["content"]


def test_default_threshold_routes_multi_concern_repository_task() -> None:
    primary = FakeBackend(
        [Candidate(content="a"), Candidate(content="b"), Candidate(content="c")]
    )
    verifier = FakeBackend(
        [Candidate(content='{"winner": 0}'), Candidate(content="refined a")]
    )
    result = SelectiveRouter(primary, verifier).chat(
        request("Implement and debug this multi-file repository change.")
    )
    assert result.metadata["routing"]["reasons"] == ["complex_request"]
    assert result.metadata["routing"]["inference_calls"] == 5


def test_neg_uncertainty_routes_even_for_short_request() -> None:
    primary = FakeBackend(
        [
            Candidate(
                content="a",
                metadata={"neg": {"activation_rate": 0.2, "activations": 20}},
            ),
            Candidate(content="b"),
        ]
    )
    verifier = FakeBackend(
        [Candidate(content='{"winner": 0}'), Candidate(content="refined a")]
    )
    result = SelectiveRouter(primary, verifier, candidate_count=2).chat(request("Solve it"))
    assert result.metadata["routing"]["reasons"] == ["neg_uncertainty"]


def test_short_high_activation_trace_does_not_route() -> None:
    primary = FakeBackend(
        [Candidate(content="a", metadata={"neg": {"activation_rate": 0.2, "activations": 3}})]
    )
    result = SelectiveRouter(primary, FakeBackend([])).chat(request("Solve it"))
    assert result.metadata["routing"]["inference_calls"] == 1


def test_tool_result_continuation_stays_single_pass_despite_uncertainty() -> None:
    primary = FakeBackend(
        [
            Candidate(
                content="I might not be sure, but here is the result.",
                metadata={"neg": {"activation_rate": 0.4, "activations": 80}},
            )
        ]
    )
    follow_up = ChatRequest(
        messages=[
            {"role": "user", "content": "Inspect README.md"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "Read", "arguments": '{"path":"README.md"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "documentation"},
        ],
        model="test",
    )
    result = SelectiveRouter(primary, FakeBackend([])).chat(follow_up)
    assert result.metadata["routing"]["ensemble"] is False
    assert result.metadata["routing"]["inference_calls"] == 1


def test_low_impact_tool_call_is_not_ensembled_only_for_neg_uncertainty() -> None:
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "tavily_search", "arguments": '{"query":"gold"}'},
    }
    primary = FakeBackend(
        [
            Candidate(
                tool_calls=[call],
                metadata={"neg": {"activation_rate": 0.4, "activations": 80}},
            )
        ]
    )
    result = SelectiveRouter(primary, FakeBackend([])).chat(request("Search the web"))
    assert result.tool_calls == [call]
    assert result.metadata["routing"]["ensemble"] is False


def test_refiner_cannot_turn_selected_tool_call_into_plain_text() -> None:
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
    }
    primary = FakeBackend(
        [
            Candidate(tool_calls=[tool_call]),
            Candidate(content="alternative"),
        ]
    )
    verifier = FakeBackend(
        [
            Candidate(content='{"winner": 0}'),
            Candidate(content='{"name":"read_file","path":"README.md"}'),
        ]
    )
    result = SelectiveRouter(primary, verifier, candidate_count=2).chat(
        request("Use read_file"), force_ensemble=True
    )

    assert result.tool_calls == [tool_call]
    assert result.metadata["routing"]["refinement_mode"] == "selected_tool_call_preserved"


def test_evaluator_meta_prose_is_rejected_in_favor_of_selected_answer() -> None:
    primary = FakeBackend([Candidate(content="useful answer"), Candidate(content="alternative")])
    verifier = FakeBackend(
        [
            Candidate(content='{"winner": 0}'),
            Candidate(
                content="Looking at the selected candidate, the evaluator's assessment is correct. "
                "No refinement is needed."
            ),
        ]
    )
    result = SelectiveRouter(primary, verifier, candidate_count=2).chat(
        request("Solve it"), force_ensemble=True
    )
    assert result.content == "useful answer"
    assert result.metadata["routing"]["refinement_mode"] == "selected_fallback_evaluator_leak"


def test_complexity_score_is_explainable() -> None:
    assert complexity_score(request("Implement a refactor of this architecture.\n```py\nx = 1\n```")) >= 4


def test_explicit_20x_budget_means_15_candidates_three_reviews_evaluator_refiner() -> None:
    primary = FakeBackend(
        [
            Candidate(content=f"candidate {i}", prompt_tokens=10, completion_tokens=2)
            for i in range(15)
        ]
    )
    verifier = FakeBackend(
        [
            Candidate(
                content='{"winner": 14, "confidence": 0.7}',
                prompt_tokens=100,
                completion_tokens=4,
            )
            for _ in range(3)
        ]
        + [
            Candidate(
                content='{"winner": 14, "confidence": 0.8}',
                prompt_tokens=200,
                completion_tokens=8,
            ),
            Candidate(content="refined candidate 14", prompt_tokens=300, completion_tokens=12),
        ]
    )
    router = SelectiveRouter(primary, verifier)
    result = router.chat(request("Solve carefully"), total_inferences=20)
    assert result.content == "refined candidate 14"
    assert len(primary.requests) == 15
    assert len(verifier.requests) == 5
    assert result.metadata["routing"]["inference_calls"] == 20
    assert result.metadata["routing"]["candidate_count"] == 15
    assert result.metadata["routing"]["reviewer_count"] == 3
    assert all(item.top_k == 1 for item in primary.requests[1:7])
    assert all(item.top_k == 20 for item in primary.requests[7:])
    assert result.prompt_tokens == 15 * 10 + 3 * 100 + 200 + 300
    assert result.completion_tokens == 15 * 2 + 3 * 4 + 8 + 12


def test_explicit_budget_requires_meaningful_candidate_diversity() -> None:
    router = SelectiveRouter(FakeBackend([Candidate(content="a")]), FakeBackend([]))
    with pytest.raises(ValueError, match="at least four"):
        router.chat(request("Solve carefully"), total_inferences=2)
