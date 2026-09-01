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
                },
            },
        ),
        "darwin-neg-agent",
        2.0,
    )
    value = store.snapshot()
    assert value["requests"] == 1
    assert value["inference_calls"] == 5
    assert value["completion_tokens"] == 50
    assert value["tokens_per_second"] == 25
    assert value["neg_activation_rate"] == 0.1
    assert value["recent"][0]["route_reasons"] == ["neg_uncertainty"]
    assert "secret answer" not in str(value)


def test_telemetry_counts_errors() -> None:
    store = TelemetryStore()
    store.record_error()
    assert store.snapshot()["errors"] == 1
