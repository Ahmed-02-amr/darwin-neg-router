"""Summarize partial or complete Darwin GPQA JSONL benchmark output."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = "ABCD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Summarize only the first N append-only records.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return records


def plurality(labels: list[str]) -> str | None:
    counts = Counter(label for label in labels if label in LABELS)
    if not counts:
        return None
    return sorted(counts, key=lambda label: (-counts[label], label))[0]


def wilson(correct: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [center - margin, center + margin]


def score(predictions: list[str | None], answers: list[str]) -> dict[str, Any]:
    correct = sum(prediction == answer for prediction, answer in zip(predictions, answers))
    total = len(answers)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
        "wilson_95": wilson(correct, total),
        "unparsed": sum(
            prediction is None or prediction not in LABELS
            for prediction in predictions
        ),
    }


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    records = load_records(args.path)
    if args.limit is not None:
        records = records[: args.limit]
    answers = [str(record["correct"]) for record in records]
    greedy: list[str | None] = []
    permutation: list[str | None] = []
    adaptive: list[str | None] = []
    four_vote_ties = 0
    finish_reasons: Counter[str] = Counter()
    neg_steps = 0
    neg_activations = 0

    for record in records:
        calls = list(record.get("calls") or [])
        first_four = calls[:4]
        greedy.append(first_four[0].get("canonical_answer") if first_four else None)
        initial_labels = [str(call.get("canonical_answer")) for call in first_four]
        counts = Counter(label for label in initial_labels if label in LABELS)
        if counts and list(counts.values()).count(max(counts.values())) > 1:
            four_vote_ties += 1
        permutation.append(plurality(initial_labels))
        adaptive.append(record.get("predicted"))
        for call in calls:
            finish_reasons[str(call.get("finish_reason") or "unknown")] += 1
            neg = call.get("neg") or {}
            neg_steps += int(neg.get("steps") or 0)
            neg_activations += int(neg.get("activations") or 0)

    call_counts = Counter(int(record["inference_calls"]) for record in records)
    wall_seconds = sum(float(record.get("wall_seconds") or 0) for record in records)
    prompt_tokens = sum(int(record.get("prompt_tokens") or 0) for record in records)
    completion_tokens = sum(int(record.get("completion_tokens") or 0) for record in records)
    summary = {
        "path": str(args.path.resolve()),
        "records": len(records),
        "dataset_revision": records[0].get("dataset_revision") if records else None,
        "protocol_version": records[0].get("protocol_version") if records else None,
        "backend": records[0].get("backend") if records else None,
        "model": records[0].get("model") if records else None,
        "scores": {
            "greedy_first_call": score(greedy, answers),
            "permutation4_plurality": score(permutation, answers),
            "adaptive20_final": score(adaptive, answers),
        },
        "four_vote_ties": four_vote_ties,
        "call_count_distribution": dict(sorted(call_counts.items())),
        "total_inference_calls": sum(
            count * frequency for count, frequency in call_counts.items()
        ),
        "average_inference_calls": (
            sum(count * frequency for count, frequency in call_counts.items()) / len(records)
            if records
            else None
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "wall_seconds": wall_seconds,
        "average_wall_seconds": wall_seconds / len(records) if records else None,
        "finish_reasons": dict(finish_reasons),
        "neg": {
            "steps": neg_steps,
            "activations": neg_activations,
            "activation_rate": neg_activations / neg_steps if neg_steps else None,
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
