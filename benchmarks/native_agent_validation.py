"""Small executable coding, tool-call, and throughput validation for the native NEG stack."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


CODING_TASKS = (
    {
        "name": "first_index",
        "prompt": (
            "Return only Python code defining first_index(xs, target). xs is sorted and may contain "
            "duplicates. Return the first matching index or -1. Use O(log n) time. No imports or type hints."
        ),
        "tests": [
            "first_index([], 3) == -1",
            "first_index([1, 2, 2, 2, 7], 2) == 1",
            "first_index([1, 3, 5], 4) == -1",
            "first_index([1, 1], 1) == 0",
        ],
    },
    {
        "name": "merge_intervals",
        "prompt": (
            "Return only Python code defining merge_intervals(intervals). Merge overlapping or touching "
            "[start, end] integer intervals, return sorted lists, and do not mutate the input. No imports "
            "or type hints."
        ),
        "tests": [
            "merge_intervals([]) == []",
            "merge_intervals([[1, 3], [2, 6], [8, 10], [10, 12]]) == [[1, 6], [8, 12]]",
            "merge_intervals([[5, 7], [1, 2], [3, 4]]) == [[1, 2], [3, 4], [5, 7]]",
            "(lambda x: (merge_intervals(x), x))([[1, 4], [2, 3]]) == ([[1, 4]], [[1, 4], [2, 3]])",
        ],
    },
    {
        "name": "deep_merge",
        "prompt": (
            "Return only Python code defining deep_merge(left, right). Recursively merge dictionaries "
            "without mutating either input; right wins for non-dictionary conflicts and lists are replaced, "
            "not concatenated. No imports or type hints."
        ),
        "tests": [
            "deep_merge({'a': 1}, {'b': 2}) == {'a': 1, 'b': 2}",
            "deep_merge({'a': {'x': 1, 'y': 2}}, {'a': {'y': 3}}) == {'a': {'x': 1, 'y': 3}}",
            "deep_merge({'a': [1], 'b': {'x': 1}}, {'a': [2], 'b': 4}) == {'a': [2], 'b': 4}",
            "(lambda l, r: (deep_merge(l, r), l, r))({'a': {'x': 1}}, {'a': {'y': 2}}) == "
            "({'a': {'x': 1, 'y': 2}}, {'a': {'x': 1}}, {'a': {'y': 2}})",
        ],
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-url", default="http://127.0.0.1:11436/v1")
    parser.add_argument("--router-url", default="http://127.0.0.1:11435/v1")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def post(url: str, body: dict[str, Any]) -> tuple[dict[str, Any], float]:
    request = urllib.request.Request(
        f"{url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1200) as response:
        payload = json.loads(response.read())
    return payload, time.perf_counter() - started


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    code = blocks[-1].strip() if blocks else text.strip()
    tree = ast.parse(code)
    if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)):
        raise ValueError("generated code contains an import")
    return code


def run_code_tests(code: str, tests: list[str]) -> tuple[bool, str]:
    harness = r'''
import json, sys
payload = json.loads(sys.stdin.read())
safe = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "float": float, "int": int, "len": len,
    "isinstance": isinstance, "list": list, "max": max, "min": min, "range": range, "reversed": reversed,
    "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
}
namespace = {"__builtins__": safe}
exec(compile(payload["code"], "<model>", "exec"), namespace, namespace)
for expression in payload["tests"]:
    if eval(expression, namespace, namespace) is not True:
        raise AssertionError(expression)
print("pass")
'''
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", harness],
            input=json.dumps({"code": code, "tests": tests}),
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "execution timeout"
    detail = (result.stdout + result.stderr).strip()
    return result.returncode == 0, detail[-1200:]


def coding_benchmark(native_url: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, task in enumerate(CODING_TASKS):
        payload, wall = post(
            native_url,
            {
                "model": "darwin-9b-neg-native",
                "messages": [{"role": "user", "content": task["prompt"]}],
                "temperature": 0,
                "max_tokens": 4096,
                "seed": 7001 + index,
            },
        )
        message = payload["choices"][0]["message"]
        try:
            code = extract_code(message.get("content") or "")
            passed, detail = run_code_tests(code, task["tests"])
        except (SyntaxError, ValueError) as exc:
            passed, detail, code = False, str(exc), message.get("content") or ""
        results.append(
            {
                "name": task["name"],
                "passed": passed,
                "detail": detail,
                "wall_seconds": wall,
                "completion_tokens": payload.get("usage", {}).get("completion_tokens", 0),
                "neg": payload.get("neg", {}),
                "code": code,
            }
        )
    return results


def tool_benchmark(router_url: str) -> list[dict[str, Any]]:
    cases = (
        ("read_file", "Read src/app.py.", {"path": "src/app.py"}),
        ("search_code", "Search the repository for class Database.", {"query": "class Database"}),
        ("run_tests", "Run tests/test_router.py.", {"target": "tests/test_router.py"}),
    )
    results: list[dict[str, Any]] = []
    for name, prompt, expected in cases:
        properties = {key: {"type": "string"} for key in expected}
        payload, wall = post(
            router_url,
            {
                "model": "darwin-neg-agent",
                "messages": [{"role": "user", "content": prompt}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": f"Agent tool {name}",
                            "parameters": {
                                "type": "object",
                                "properties": properties,
                                "required": list(expected),
                            },
                        },
                    }
                ],
                "tool_choice": {"type": "function", "function": {"name": name}},
                "parallel_tool_calls": False,
                "temperature": 0,
                "max_tokens": 1024,
            },
        )
        calls = payload["choices"][0]["message"].get("tool_calls") or []
        actual_name = calls[0]["function"]["name"] if calls else None
        try:
            actual_args = json.loads(calls[0]["function"]["arguments"]) if calls else None
        except json.JSONDecodeError:
            actual_args = None
        results.append(
            {
                "name": name,
                "passed": actual_name == name and actual_args == expected,
                "actual_name": actual_name,
                "actual_arguments": actual_args,
                "wall_seconds": wall,
                "routing": payload.get("darwin", {}).get("routing", {}),
            }
        )
    return results


def performance_benchmark(native_url: str) -> dict[str, Any]:
    payload, wall = post(
        native_url,
        {
            "model": "darwin-9b-neg-native",
            "messages": [
                {
                    "role": "user",
                    "content": "Produce a detailed numbered code-review checklist with at least 100 items.",
                }
            ],
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 512,
            "seed": 991,
        },
    )
    tokens = int(payload.get("usage", {}).get("completion_tokens", 0))
    return {
        "wall_seconds": wall,
        "completion_tokens": tokens,
        "tokens_per_second": tokens / wall,
        "finish_reason": payload["choices"][0]["finish_reason"],
        "neg": payload.get("neg", {}),
    }


def main() -> None:
    args = parse_args()
    coding = coding_benchmark(args.native_url)
    tools = tool_benchmark(args.router_url)
    performance = performance_benchmark(args.native_url)
    result = {
        "coding": coding,
        "coding_passed": sum(int(item["passed"]) for item in coding),
        "coding_total": len(coding),
        "tools": tools,
        "tools_passed": sum(int(item["passed"]) for item in tools),
        "tools_total": len(tools),
        "performance": performance,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
