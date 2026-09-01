"""Reproducible GPQA-Diamond runner for Ollama or the native Darwin NEG server."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from darwin_neg_router.backends import OllamaBackend, OpenAIBackend
from darwin_neg_router.gpqa import GPQAEnsembler, LABELS, parse_multiple_choice


DEFAULT_DATASET = (
    Path(__file__).parent / "data" / "fingertap-gpqa-diamond" / "test" / "gpqa_diamond.parquet"
)
DATASET_REVISION = "68be7564497676e07a77a042fdb587deb88c51c3"
PROTOCOL_VERSION = "darwin-gpqa-reconstruction-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a label-blind Darwin GPQA ensemble on GPQA-Diamond"
    )
    parser.add_argument("--parquet", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--backend", choices=("ollama", "native"), default="native")
    parser.add_argument("--model", default="darwin-9b-neg-native")
    parser.add_argument(
        "--verifier-model",
        default=None,
        help="Optional cross-model verifier; omit for a pure Darwin self-ensemble",
    )
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--native-url", default="http://127.0.0.1:11436/v1")
    parser.add_argument(
        "--mode",
        choices=("greedy", "permutation4", "adaptive20", "full20"),
        default="adaptive20",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--solver-tokens", type=int, default=2048)
    parser.add_argument("--review-tokens", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/gpqa.jsonl"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all question/answer rows without performing inference",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    frame = pd.read_parquet(path, columns=["question", "answer"])
    rows = frame.to_dict(orient="records")
    for index, row in enumerate(rows):
        parse_multiple_choice(str(row["question"]))
        answer = str(row["answer"]).strip().upper()
        if answer not in LABELS:
            raise ValueError(f"Row {index} has invalid answer label {answer!r}")
        row["answer"] = answer
    return rows


def existing_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    result: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result.append(json.loads(line))
    return result


def main() -> None:
    args = parse_args()
    rows = load_rows(args.parquet)
    if args.validate_only:
        print(f"validated_rows={len(rows)} source={args.parquet}")
        return

    selected = list(enumerate(rows))[args.offset :]
    if args.limit is not None:
        selected = selected[: args.limit]
    prior = existing_records(args.output) if args.resume else []
    selected_indices = {index for index, _row in selected}
    prior = [record for record in prior if int(record["index"]) in selected_indices]
    done = {int(record["index"]) for record in prior}
    if args.backend == "native":
        solver = OpenAIBackend(args.native_url, "", args.model, native_neg=True)
        verifier = (
            OpenAIBackend(args.native_url, "", args.verifier_model, native_neg=True)
            if args.verifier_model
            else solver
        )
    else:
        solver = OllamaBackend(args.ollama_url, args.model)
        verifier = OllamaBackend(args.ollama_url, args.verifier_model) if args.verifier_model else solver
    ensemble = GPQAEnsembler(
        solver,
        verifier,
        solver_tokens=args.solver_tokens,
        review_tokens=args.review_tokens,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    correct = sum(int(bool(record["is_correct"])) for record in prior)
    attempted = len(prior)
    calls = sum(int(record["inference_calls"]) for record in prior)
    tokens = sum(int(record["completion_tokens"]) for record in prior)
    started = time.perf_counter()
    with args.output.open("a" if args.resume else "w", encoding="utf-8") as output:
        for position, (index, row) in enumerate(selected, start=1):
            if index in done:
                continue
            item_started = time.perf_counter()
            result = ensemble.solve(
                row["question"],
                mode=args.mode,
                seed=100_003 * (index + 1),
            )
            predicted = str(result.metadata["predicted"])
            is_correct = predicted == row["answer"]
            attempted += 1
            correct += int(is_correct)
            calls += int(result.metadata["inference_calls"])
            tokens += result.completion_tokens
            record = {
                "index": index,
                "dataset": "fingertap/GPQA-Diamond",
                "dataset_revision": DATASET_REVISION,
                "protocol_version": PROTOCOL_VERSION,
                "question_hash": result.metadata["question_hash"],
                "model": args.model,
                "backend": args.backend,
                "verifier_model": args.verifier_model or args.model,
                "mode": args.mode,
                "correct": row["answer"],
                "predicted": predicted,
                "is_correct": is_correct,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "inference_calls": result.metadata["inference_calls"],
                "stop_reason": result.metadata["stop_reason"],
                "votes": result.metadata["votes"],
                "calls": result.metadata["calls"],
                "wall_seconds": time.perf_counter() - item_started,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            elapsed = time.perf_counter() - started
            print(
                f"{position}/{len(selected)} index={index} calls={record['inference_calls']} "
                f"correct={is_correct} accuracy={correct / attempted:.3f} "
                f"avg_calls={calls / attempted:.2f} tok_s={tokens / max(elapsed, 1e-9):.1f}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "attempted": attempted,
                "correct": correct,
                "accuracy": correct / attempted if attempted else None,
                "calls": calls,
                "average_calls": calls / attempted if attempted else None,
                "completion_tokens": tokens,
                "wall_seconds": elapsed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
