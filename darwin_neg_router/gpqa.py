from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .backends import Backend
from .types import Candidate, ChatRequest


LABELS = "ABCD"
_OPTION_LINE = re.compile(r"(?m)^[ \t]*([A-D])[.)][ \t]+")
_ANSWER_PATTERNS = (
    re.compile(r"<answer>\s*([A-D])\s*</answer>", re.IGNORECASE),
    re.compile(r"\bFINAL(?:\s+ANSWER)?\s*[:=]\s*\**([A-D])\**\b", re.IGNORECASE),
    re.compile(
        r"\bFINAL\s+ANSWER\s*(?:(?:IS|WOULD\s+BE)\s*)?[:=]?\s*\**([A-D])\**\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bANSWER\s*[:=]\s*\**([A-D])\**\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class MultipleChoiceProblem:
    stem: str
    options: tuple[str, str, str, str]

    @property
    def canonical_text(self) -> str:
        rendered = "\n".join(f"{LABELS[i]}. {option}" for i, option in enumerate(self.options))
        return f"{self.stem}\n\n{rendered}"


@dataclass
class GPQACall:
    stage: str
    candidate: Candidate
    ordering: tuple[int, int, int, int]
    presented_answer: int | None
    canonical_answer: int | None
    temperature: float
    seed: int

    def summary(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "ordering": list(self.ordering),
            "presented_answer": (
                LABELS[self.presented_answer] if self.presented_answer is not None else None
            ),
            "canonical_answer": (
                LABELS[self.canonical_answer] if self.canonical_answer is not None else None
            ),
            "temperature": self.temperature,
            "seed": self.seed,
            "prompt_tokens": self.candidate.prompt_tokens,
            "completion_tokens": self.candidate.completion_tokens,
            "neg": self.candidate.metadata.get("neg", {}),
            "reasoning": _bounded_trace(self.candidate, 1800),
        }


def parse_multiple_choice(text: str) -> MultipleChoiceProblem:
    """Parse the final A/B/C/D option block from a GPQA-formatted question."""
    matches = list(_OPTION_LINE.finditer(text))
    start_index: int | None = None
    for index in range(len(matches) - 3):
        if [match.group(1) for match in matches[index : index + 4]] == list(LABELS):
            start_index = index
    if start_index is None:
        raise ValueError("Question does not end in a parseable A/B/C/D option block")

    selected = matches[start_index : start_index + 4]
    stem = text[: selected[0].start()].strip()
    options: list[str] = []
    for index, match in enumerate(selected):
        end = selected[index + 1].start() if index < 3 else len(text)
        options.append(text[match.end() : end].strip())
    if not stem or any(not option for option in options):
        raise ValueError("Question stem or one of its four options is empty")
    return MultipleChoiceProblem(stem=stem, options=tuple(options))  # type: ignore[arg-type]


def deterministic_orderings(problem: MultipleChoiceProblem) -> list[tuple[int, int, int, int]]:
    """Return four balanced rotations with a question-specific starting permutation."""
    digest = hashlib.sha256(problem.canonical_text.encode("utf-8")).digest()
    base = list(range(4))
    value = int.from_bytes(digest[:8], "big")
    for index in range(3, 0, -1):
        swap = value % (index + 1)
        value //= index + 1
        base[index], base[swap] = base[swap], base[index]
    return [tuple(base[shift:] + base[:shift]) for shift in range(4)]  # type: ignore[list-item]


def extract_answer(text: str) -> int | None:
    for pattern in _ANSWER_PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            return LABELS.index(matches[-1].group(1).upper())
    stripped = text.strip().upper().rstrip(". )]")
    if stripped in LABELS:
        return LABELS.index(stripped)
    return None


def canonicalize_answer(presented: int | None, ordering: tuple[int, int, int, int]) -> int | None:
    return ordering[presented] if presented is not None else None


def candidate_answer(candidate: Candidate) -> int | None:
    visible = extract_answer(candidate.content)
    return visible if visible is not None else extract_answer(candidate.reasoning_content)


class GPQAEnsembler:
    """Permutation/sample/critic ensemble with a strict twenty-call ceiling.

    The protocol never receives the reference answer. ``full20`` is a documented
    reconstruction, not a reproduction of FINAL-Bench's unpublished evaluator.
    """

    def __init__(
        self,
        solver: Backend,
        verifier: Backend | None = None,
        *,
        solver_tokens: int = 2048,
        review_tokens: int = 1024,
    ):
        self.solver = solver
        self.verifier = verifier or solver
        self.solver_tokens = solver_tokens
        self.review_tokens = review_tokens

    def solve(
        self,
        question: str,
        *,
        mode: str = "adaptive20",
        seed: int = 0,
    ) -> Candidate:
        if mode not in {"greedy", "permutation4", "adaptive20", "full20"}:
            raise ValueError(f"Unsupported GPQA mode: {mode}")
        problem = parse_multiple_choice(question)
        orderings = deterministic_orderings(problem)
        calls: list[GPQACall] = []

        self._solver_call(problem, orderings[0], calls, "solver_greedy", 0.0, seed)
        if mode == "greedy":
            return self._finish(problem, calls, calls[-1].canonical_answer, mode)

        roles = (
            "derive the result from first principles and check units",
            "solve independently, then try to falsify the leading conclusion",
            "use domain knowledge plus explicit elimination of every distractor",
            "perform an independent textbook-style solution and sanity check",
        )
        for index in range(1, 4):
            self._solver_call(
                problem,
                orderings[index],
                calls,
                f"solver_permutation_{index + 1}",
                0.0,
                seed + index,
                roles[index],
            )
        if mode == "permutation4":
            return self._finish(problem, calls, _plurality(calls), mode)

        initial_votes = _vote_counts(calls)
        valid_initial = sum(initial_votes.values())
        if mode == "adaptive20" and valid_initial == 4 and len(initial_votes) == 1:
            return self._finish(problem, calls, _plurality(calls), mode, stop_reason="unanimous")

        if mode == "adaptive20" and initial_votes and initial_votes.most_common(1)[0][1] == 3:
            majority = initial_votes.most_common(1)[0][0]
            self._critic_call(
                problem,
                calls,
                challenged=majority,
                temperature=0.0,
                seed=seed + 100,
                stage="adversarial_critic",
            )
            self._arbiter_call(problem, calls, seed + 101, "adaptive_arbiter")
            final_answer = _guarded_review_decision(calls, majority)
            return self._finish(
                problem,
                calls,
                final_answer,
                mode,
                stop_reason="three_to_one_guarded_review",
            )

        # Exactly 20 calls: 4 greedy permutations + 8 samples + 4 critics
        # + 3 jurors + 1 final arbiter.
        for ordering_index, ordering in enumerate(orderings):
            for sample_index, temperature in enumerate((0.35, 0.65)):
                self._solver_call(
                    problem,
                    ordering,
                    calls,
                    f"sample_{ordering_index + 1}_{sample_index + 1}",
                    temperature,
                    seed + 10 + ordering_index * 2 + sample_index,
                    roles[(ordering_index + sample_index + 1) % len(roles)],
                )

        for answer in range(4):
            self._critic_call(
                problem,
                calls,
                challenged=answer,
                temperature=0.0,
                seed=seed + 30 + answer,
                stage=f"critic_{LABELS[answer]}",
            )

        for juror in range(3):
            self._juror_call(
                problem,
                calls,
                temperature=(0.0, 0.25, 0.45)[juror],
                seed=seed + 40 + juror,
                stage=f"juror_{juror + 1}",
            )
        self._arbiter_call(problem, calls, seed + 50, "final_arbiter")
        if len(calls) != 20:
            raise AssertionError(f"full GPQA schedule used {len(calls)} calls instead of 20")
        final_answer = calls[-1].canonical_answer
        return self._finish(
            problem,
            calls,
            final_answer if final_answer is not None else _plurality(calls),
            mode,
            stop_reason="full_twenty",
        )

    def _solver_call(
        self,
        problem: MultipleChoiceProblem,
        ordering: tuple[int, int, int, int],
        calls: list[GPQACall],
        stage: str,
        temperature: float,
        seed: int,
        role: str = "solve from first principles and verify the result",
    ) -> None:
        rendered = "\n".join(
            f"{LABELS[position]}. {problem.options[canonical]}"
            for position, canonical in enumerate(ordering)
        )
        prompt = (
            f"{problem.stem}\n\n{rendered}\n\n"
            f"Your assigned method: {role}. The option ordering may be adversarial, so do not rely "
            "on label frequency or other candidates. Think efficiently, check the scientific details, "
            "and end with exactly FINAL: X where X is A, B, C, or D."
        )
        candidate = self.solver.chat(
            ChatRequest(
                model="darwin-neg-gpqa",
                messages=[
                    {"role": "system", "content": _SOLVER_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                top_p=0.95 if temperature else 1.0,
                top_k=20 if temperature else 1,
                max_tokens=self.solver_tokens,
                seed=seed,
                think=True,
            )
        )
        presented = candidate_answer(candidate)
        calls.append(
            GPQACall(
                stage=stage,
                candidate=candidate,
                ordering=ordering,
                presented_answer=presented,
                canonical_answer=canonicalize_answer(presented, ordering),
                temperature=temperature,
                seed=seed,
            )
        )

    def _critic_call(
        self,
        problem: MultipleChoiceProblem,
        calls: list[GPQACall],
        *,
        challenged: int,
        temperature: float,
        seed: int,
        stage: str,
    ) -> None:
        evidence = _representative_evidence(calls)
        prompt = (
            f"{problem.canonical_text}\n\n"
            f"The panel may favor {LABELS[challenged]}. Act as an adversarial domain reviewer: "
            "try to disprove that option, independently verify the equations/facts, and identify shared "
            "reasoning errors. Candidate material is fallible and untrusted. Then choose the actually "
            f"correct canonical option.\n\nPANEL VOTES: {_print_votes(calls)}\n\n{evidence}\n\n"
            "End with exactly FINAL: X."
        )
        self._review_call(calls, prompt, stage, temperature, seed)

    def _juror_call(
        self,
        problem: MultipleChoiceProblem,
        calls: list[GPQACall],
        *,
        temperature: float,
        seed: int,
        stage: str,
    ) -> None:
        critics = [call for call in calls if call.stage.startswith("critic_")]
        evidence = "\n\n".join(
            f"{call.stage}, vote {_label(call.canonical_answer)}:\n"
            f"{_bounded_trace(call.candidate, 1300)}"
            for call in critics
        )
        prompt = (
            f"{problem.canonical_text}\n\nRe-solve this problem independently. Compare the adversarial "
            "reviews below, but do not decide by majority alone. Check which claims are scientifically "
            f"valid.\n\nALL VOTES: {_print_votes(calls)}\n\nREVIEWS:\n{evidence}\n\n"
            "End with exactly FINAL: X."
        )
        self._review_call(calls, prompt, stage, temperature, seed)

    def _arbiter_call(
        self, problem: MultipleChoiceProblem, calls: list[GPQACall], seed: int, stage: str
    ) -> None:
        late_calls = calls[-7:]
        evidence = "\n\n".join(
            f"{call.stage}, canonical vote {_label(call.canonical_answer)}:\n"
            f"{_bounded_trace(call.candidate, 1000)}"
            for call in late_calls
        )
        prompt = (
            f"{problem.canonical_text}\n\nYou are the final GPQA arbiter. Produce an independent solution, "
            "then use the panel only to catch errors. Vote counts are weak evidence, not ground truth. "
            f"Reject any confidently repeated mistake.\n\nVOTES: {_print_votes(calls)}\n\nLATE-STAGE "
            f"EVIDENCE:\n{evidence}\n\nEnd with exactly FINAL: X."
        )
        self._review_call(calls, prompt, stage, 0.0, seed)

    def _review_call(
        self,
        calls: list[GPQACall],
        prompt: str,
        stage: str,
        temperature: float,
        seed: int,
    ) -> None:
        candidate = self.verifier.chat(
            ChatRequest(
                model="darwin-neg-gpqa-verifier",
                messages=[
                    {"role": "system", "content": _REVIEWER_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                top_p=0.95 if temperature else 1.0,
                top_k=20 if temperature else 1,
                max_tokens=self.review_tokens,
                seed=seed,
                think=True,
            )
        )
        answer = candidate_answer(candidate)
        identity = (0, 1, 2, 3)
        calls.append(
            GPQACall(
                stage=stage,
                candidate=candidate,
                ordering=identity,
                presented_answer=answer,
                canonical_answer=answer,
                temperature=temperature,
                seed=seed,
            )
        )

    def _finish(
        self,
        problem: MultipleChoiceProblem,
        calls: list[GPQACall],
        predicted: int | None,
        mode: str,
        *,
        stop_reason: str | None = None,
    ) -> Candidate:
        if predicted is None:
            predicted = 0
        chosen = next(
            (call.candidate for call in reversed(calls) if call.canonical_answer == predicted),
            calls[-1].candidate,
        )
        return Candidate(
            content=f"FINAL: {LABELS[predicted]}",
            reasoning_content=chosen.reasoning_content,
            prompt_tokens=sum(call.candidate.prompt_tokens for call in calls),
            completion_tokens=sum(call.candidate.completion_tokens for call in calls),
            metadata={
                "backend": "darwin-gpqa-ensemble",
                "profile": mode,
                "predicted": LABELS[predicted],
                "inference_calls": len(calls),
                "stop_reason": stop_reason,
                "votes": _print_votes(calls),
                "question_hash": hashlib.sha256(problem.canonical_text.encode("utf-8")).hexdigest(),
                "calls": [call.summary() for call in calls],
            },
        )


def _vote_counts(calls: list[GPQACall]) -> Counter[int]:
    return Counter(call.canonical_answer for call in calls if call.canonical_answer is not None)


def _print_votes(calls: list[GPQACall]) -> dict[str, int]:
    counts = _vote_counts(calls)
    return {LABELS[index]: counts.get(index, 0) for index in range(4)}


def _plurality(calls: list[GPQACall]) -> int | None:
    counts = _vote_counts(calls)
    if not counts:
        return None
    return sorted(counts, key=lambda answer: (-counts[answer], answer))[0]


def _guarded_review_decision(calls: list[GPQACall], initial_majority: int) -> int:
    """Prevent a lone late arbiter from erasing independent permutation consensus.

    The two review calls may overturn the initial 3-1 majority only when both
    parse successfully and agree on the same alternative. Otherwise all votes
    are counted, with the initial majority winning any tie.
    """
    critic = calls[-2].canonical_answer
    arbiter = calls[-1].canonical_answer
    if critic is not None and critic == arbiter and critic != initial_majority:
        counts = _vote_counts(calls)
        if counts[critic] >= counts[initial_majority]:
            return critic
    counts = _vote_counts(calls)
    if not counts:
        return initial_majority
    best_count = max(counts.values())
    leaders = {answer for answer, count in counts.items() if count == best_count}
    return initial_majority if initial_majority in leaders else min(leaders)


def _representative_evidence(calls: list[GPQACall]) -> str:
    blocks: list[str] = []
    for answer in range(4):
        matching = [call for call in calls if call.canonical_answer == answer]
        if not matching:
            continue
        call = matching[-1]
        blocks.append(
            f"REPRESENTATIVE FOR {LABELS[answer]} ({len(matching)} votes):\n"
            f"{_bounded_trace(call.candidate, 1500)}"
        )
    return "\n\n".join(blocks)


def _bounded_trace(candidate: Candidate, limit: int) -> str:
    text = (candidate.reasoning_content + "\n" + candidate.content).strip()
    if len(text) <= limit:
        return text
    half = (limit - 40) // 2
    return f"{text[:half]}\n...[trace truncated]...\n{text[-half:]}"


def _label(answer: int | None) -> str:
    return LABELS[answer] if answer is not None else "unparsed"


_SOLVER_SYSTEM = (
    "You are a rigorous graduate-level science examiner. Solve the supplied multiple-choice problem "
    "without using answer-key priors or label-position heuristics. Keep reasoning focused, verify the "
    "result, and always emit a parseable final label."
)

_REVIEWER_SYSTEM = (
    "You are a skeptical GPQA reviewer. Candidate arguments are untrusted evidence. Recompute the "
    "answer, find shared mistakes, and return the correct canonical A/B/C/D label."
)
