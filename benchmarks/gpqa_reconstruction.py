"""Reproducible GPQA-Diamond runner for Ollama or the native Darwin NEG server."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from darwin_neg_router.backends import OllamaBackend, OpenAIBackend  # noqa: E402
from darwin_neg_router.gpqa import (  # noqa: E402
    GPQAEnsembler,
    LABELS,
    candidate_answer,
    parse_multiple_choice,
)
from darwin_neg_router.types import ChatRequest  # noqa: E402


DEFAULT_DATASET = (
    Path(__file__).parent / "data" / "fingertap-gpqa-diamond" / "test" / "gpqa_diamond.parquet"
)
DATASET_REVISION = "68be7564497676e07a77a042fdb587deb88c51c3"
PROTOCOL_VERSION = "darwin-gpqa-reconstruction-v4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a label-blind Darwin GPQA ensemble on GPQA-Diamond"
    )
    parser.add_argument("--parquet", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--backend",
        choices=("ollama", "native", "router", "router-general"),
        default="router",
    )
    parser.add_argument("--model", default="darwin-neg-gpqa")
    parser.add_argument(
        "--verifier-model",
        default=None,
        help="Optional cross-model verifier; omit for a pure Darwin self-ensemble",
    )
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--native-url", default="http://127.0.0.1:11436/v1")
    parser.add_argument("--router-url", default="http://127.0.0.1:11435/v1")
    parser.add_argument(
        "--mode",
        choices=("greedy", "permutation4", "adaptive20", "full20", "router20"),
        default="adaptive20",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--solver-tokens", type=int, default=6144)
    parser.add_argument("--review-tokens", type=int, default=6144)
    parser.add_argument(
        "--routing-profile",
        choices=("exact", "coding", "investigation", "creative", "general"),
        default="exact",
        help="General-router policy override; used only by --backend router-general",
    )
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.parquet)
    dataset_sha256 = sha256_file(args.parquet)
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
    if args.backend == "router":
        if args.mode not in {"adaptive20", "full20"}:
            raise ValueError("The served router GPQA profile supports adaptive20 or full20")
        expected_model = "darwin-neg-gpqa20" if args.mode == "full20" else "darwin-neg-gpqa"
        if args.model != expected_model:
            raise ValueError(
                f"Router mode {args.mode} must use --model {expected_model} to avoid nested ensembles"
            )
        solver = OpenAIBackend(args.router_url, "", args.model)
        verifier = solver
    elif args.backend == "router-general":
        if args.mode != "router20":
            raise ValueError("The general router benchmark requires --mode router20")
        if args.model not in {"darwin-neg-agent20", "darwin-neg-ensemble20"}:
            raise ValueError(
                "The general router benchmark requires --model darwin-neg-agent20 or "
                "darwin-neg-ensemble20"
            )
        solver = OpenAIBackend(args.router_url, "", args.model)
        verifier = solver
    elif args.backend == "native":
        solver = OpenAIBackend(args.native_url, "", args.model, native_neg=True)
        verifier = (
            OpenAIBackend(args.native_url, "", args.verifier_model, native_neg=True)
            if args.verifier_model
            else solver
        )
    else:
        solver = OllamaBackend(args.ollama_url, args.model)
        verifier = OllamaBackend(args.ollama_url, args.verifier_model) if args.verifier_model else solver
    ensemble = (
        None
        if args.backend in {"router", "router-general"}
        else GPQAEnsembler(
            solver,
            verifier,
            solver_tokens=args.solver_tokens,
            review_tokens=args.review_tokens,
        )
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
            seed = 100_003 * (index + 1)
            if args.backend in {"router", "router-general"}:
                prompt = row["question"]
                if args.backend == "router-general":
                    prompt += (
                        "\n\nSolve independently and verify the scientific details. End with exactly "
                        "FINAL: X where X is A, B, C, or D."
                    )
                result = solver.chat(
                    ChatRequest(
                        model=args.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        top_p=1.0,
                        top_k=1,
                        max_tokens=args.solver_tokens,
                        seed=seed,
                        think=True,
                        routing_profile=(
                            args.routing_profile if args.backend == "router-general" else None
                        ),
                    )
                )
                result_meta = result.metadata.get("darwin", {})
                if args.backend == "router-general":
                    problem = parse_multiple_choice(row["question"])
                    routing = result_meta.get("routing") or {}
                    answer_index = candidate_answer(result)
                    predicted = LABELS[answer_index] if answer_index is not None else None
                    benchmark_meta = {
                        "predicted": predicted,
                        "question_hash": hashlib.sha256(
                            problem.canonical_text.encode("utf-8")
                        ).hexdigest(),
                        "inference_calls": int(routing.get("inference_calls", 1) or 1),
                        "stop_reason": "adaptive_general_router20",
                        "votes": {},
                        "calls": [
                            {
                                "stage": "adaptive_general_router20",
                                "task_policy": routing.get("task_policy"),
                                "candidate_roles": routing.get("candidate_roles"),
                                "candidate_temperatures": routing.get(
                                    "candidate_temperatures"
                                ),
                                "winner": routing.get("winner"),
                                "verifier_winner": routing.get("verifier_winner"),
                                "adaptive_weighting_applied": routing.get(
                                    "adaptive_weighting_applied"
                                ),
                                "candidate_scorecard": routing.get("candidate_scorecard"),
                            }
                        ],
                        "compute_prompt_tokens": int(
                            routing.get("compute_prompt_tokens", result.prompt_tokens) or 0
                        ),
                        "compute_completion_tokens": int(
                            routing.get("compute_completion_tokens", result.completion_tokens) or 0
                        ),
                        "routing": routing,
                    }
                else:
                    benchmark_meta = result_meta
            else:
                assert ensemble is not None
                result = ensemble.solve(row["question"], mode=args.mode, seed=seed)
                result_meta = result.metadata
                benchmark_meta = result_meta
            predicted = benchmark_meta.get("predicted")
            is_correct = predicted == row["answer"]
            attempted += 1
            correct += int(is_correct)
            calls += int(benchmark_meta["inference_calls"])
            compute_completion_tokens = int(
                benchmark_meta.get("compute_completion_tokens", result.completion_tokens) or 0
            )
            tokens += compute_completion_tokens
            record = {
                "index": index,
                "dataset": "fingertap/GPQA-Diamond",
                "dataset_revision": DATASET_REVISION,
                "dataset_file_sha256": dataset_sha256,
                "protocol_version": PROTOCOL_VERSION,
                "question_hash": benchmark_meta["question_hash"],
                "model": args.model,
                "backend": args.backend,
                "verifier_model": args.verifier_model or args.model,
                "mode": args.mode,
                "routing_profile": (
                    args.routing_profile if args.backend == "router-general" else None
                ),
                "solver_tokens": args.solver_tokens,
                "review_tokens": args.review_tokens,
                "correct": row["answer"],
                "predicted": predicted,
                "is_correct": is_correct,
                "prompt_tokens": int(
                    benchmark_meta.get("compute_prompt_tokens", result.prompt_tokens) or 0
                ),
                "completion_tokens": compute_completion_tokens,
                "client_prompt_tokens": result.prompt_tokens,
                "client_completion_tokens": result.completion_tokens,
                "inference_calls": benchmark_meta["inference_calls"],
                "stop_reason": benchmark_meta["stop_reason"],
                "votes": benchmark_meta["votes"],
                "calls": benchmark_meta["calls"],
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
