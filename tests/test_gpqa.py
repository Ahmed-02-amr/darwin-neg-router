from darwin_neg_router.backends import Backend
from darwin_neg_router.gpqa import (
    GPQAEnsembler,
    canonicalize_answer,
    deterministic_orderings,
    extract_answer,
    parse_multiple_choice,
)
from darwin_neg_router.types import Candidate, ChatRequest


QUESTION = """Which value is prime?

A. 8
B. 9
C. 11
D. 12"""


class FixedBackend(Backend):
    def __init__(self, answers: list[str]):
        self.answers = answers
        self.requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> Candidate:
        self.requests.append(request)
        return Candidate(content=f"FINAL: {self.answers.pop(0)}", completion_tokens=5)


def test_parse_gpqa_final_option_block() -> None:
    value = parse_multiple_choice("embedded a) one b) two\n\n" + QUESTION)
    assert value.stem.endswith("Which value is prime?")
    assert value.options == ("8", "9", "11", "12")


def test_balanced_deterministic_orderings() -> None:
    problem = parse_multiple_choice(QUESTION)
    first = deterministic_orderings(problem)
    second = deterministic_orderings(problem)
    assert first == second
    assert len(set(first)) == 4
    assert all(sorted(ordering) == [0, 1, 2, 3] for ordering in first)


def test_answer_parsing_and_canonicalization() -> None:
    assert extract_answer("work\nFINAL: **C**") == 2
    assert extract_answer("work\n### Final Answer\n\nB") == 1
    assert canonicalize_answer(2, (3, 2, 0, 1)) == 0


def test_unanimous_adaptive_stops_after_four_permutations() -> None:
    problem = parse_multiple_choice(QUESTION)
    orderings = deterministic_orderings(problem)
    canonical = 2
    presented = ["ABCD"[ordering.index(canonical)] for ordering in orderings]
    backend = FixedBackend(presented)
    result = GPQAEnsembler(backend).solve(QUESTION, mode="adaptive20")
    assert result.content == "FINAL: C"
    assert result.metadata["inference_calls"] == 4
    assert result.metadata["stop_reason"] == "unanimous"


def test_unparsed_fourth_vote_forces_review() -> None:
    problem = parse_multiple_choice(QUESTION)
    orderings = deterministic_orderings(problem)
    canonical = 2
    presented = ["ABCD"[ordering.index(canonical)] for ordering in orderings[:3]]
    backend = FixedBackend(presented + ["not-parseable", "C", "C"])
    result = GPQAEnsembler(backend).solve(QUESTION, mode="adaptive20")
    assert result.metadata["inference_calls"] == 6
    assert result.metadata["stop_reason"] == "three_to_one_guarded_review"


def test_disagreeing_reviewers_cannot_erase_three_vote_consensus() -> None:
    problem = parse_multiple_choice(QUESTION)
    orderings = deterministic_orderings(problem)
    canonical = 2
    majority = ["ABCD"[ordering.index(canonical)] for ordering in orderings[:3]]
    minority_canonical = 1
    minority = "ABCD"[orderings[3].index(minority_canonical)]
    backend = FixedBackend(majority + [minority, "A", "B"])
    result = GPQAEnsembler(backend).solve(QUESTION, mode="adaptive20")
    assert result.content == "FINAL: C"


def test_agreeing_reviewers_can_overturn_tied_reviewed_consensus() -> None:
    problem = parse_multiple_choice(QUESTION)
    orderings = deterministic_orderings(problem)
    initial_majority = 2
    alternative = 1
    majority = ["ABCD"[ordering.index(initial_majority)] for ordering in orderings[:3]]
    minority = "ABCD"[orderings[3].index(alternative)]
    backend = FixedBackend(majority + [minority, "B", "B"])
    result = GPQAEnsembler(backend).solve(QUESTION, mode="adaptive20")
    assert result.content == "FINAL: B"


def test_full_schedule_uses_twenty_calls() -> None:
    backend = FixedBackend(list("ABCD") * 5)
    result = GPQAEnsembler(backend).solve(QUESTION, mode="full20")
    assert result.metadata["inference_calls"] == 20
    assert len(backend.requests) == 20
