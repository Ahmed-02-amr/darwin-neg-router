from darwin_neg_router.telemetry import TelemetryStore
from darwin_neg_router.types import Candidate


def test_telemetry_aggregates_without_prompt_content() -> None:
    store = TelemetryStore()
    store.record(
        Candidate(
            content="secret answer",
            prompt_tokens=100,
            completion_tokens=50,
            metadata={
                "neg": {
                    "signal": "released_hidden_state_head",
                    "steps": 50,
                    "activations": 5,
                    "activation_rate": 0.1,
                    "guided_steps": 5,
                    "eval_ms": 30,
                },
                "routing": {
                    "ensemble": True,
                    "inference_calls": 5,
                    "reasons": ["neg_uncertainty"],
                    "compute_prompt_tokens": 500,
                    "compute_completion_tokens": 250,
                    "task_policy": {"name": "exact", "confidence": 0.9},
                    "candidate_temperatures": [0.0, 0.2],
                    "winner": 0,
                    "verifier_winner": 1,
                    "adaptive_weighting_applied": True,
                },
                "truncation_recovery": {
                    "attempted": True,
                    "succeeded": True,
                },
            },
        ),
        "darwin-neg-agent",
        2.0,
    )
    value = store.snapshot()
    assert value["requests"] == 1
    assert value["inference_calls"] == 5
    assert value["prompt_tokens"] == 500
    assert value["completion_tokens"] == 250
    assert value["tokens_per_second"] == 125
    assert value["recent"][0]["client_input_tokens"] == 100
    assert value["recent"][0]["client_output_tokens"] == 50
    assert value["neg_activation_rate"] == 0.1
    assert value["recent"][0]["route_reasons"] == ["neg_uncertainty"]
    assert value["truncation_recoveries"] == 1
    assert value["recent"][0]["truncation_recovery_succeeded"] is True
    assert value["task_profiles"] == {"exact": 1}
    assert value["adaptive_selection_changes"] == 1
    assert value["recent"][0]["selected_temperature"] == 0.0
    assert "secret answer" not in str(value)


def test_telemetry_counts_errors() -> None:
    store = TelemetryStore()
    store.record_error()
    assert store.snapshot()["errors"] == 1
