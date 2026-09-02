from darwin_neg_router.routing_policy import classify_task
from darwin_neg_router.types import ChatRequest


def request(text: str, *, tools: list[dict] | None = None, profile: str | None = None) -> ChatRequest:
    return ChatRequest(
        messages=[{"role": "user", "content": text}],
        model="test",
        tools=tools or [],
        routing_profile=profile,
    )


def tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "parameters": {}}}


def test_classifier_distinguishes_primary_task_families() -> None:
    assert classify_task(request("Calculate this integral and check the units.")).name == "exact"
    assert classify_task(request("Implement this repository refactor and update its tests.")).name == "coding"
    assert classify_task(request("Research and compare the latest evidence.")).name == "investigation"
    assert classify_task(request("Brainstorm five original story concepts.")).name == "creative"
    assert classify_task(request("Explain this clearly.")).name == "general"


def test_code_tools_keep_mixed_debugging_work_in_coding_profile() -> None:
    policy = classify_task(
        request(
            "Investigate why this creative UI implementation fails.",
            tools=[tool("Read"), tool("apply_patch"), tool("run_tests")],
        )
    )
    assert policy.name == "coding"
    assert "code_tools" in policy.signals


def test_research_tools_select_investigation_profile() -> None:
    policy = classify_task(request("Find the answer.", tools=[tool("ddg_web_search")]))
    assert policy.name == "investigation"
    assert "research_tools" in policy.signals


def test_full_codepilot_tool_catalog_does_not_turn_web_search_into_coding() -> None:
    policy = classify_task(
        request(
            "Search the web for the current gold price.",
            tools=[tool("Read"), tool("apply_patch"), tool("ddg_web_search")],
        )
    )
    assert policy.name == "investigation"


def test_explicit_profile_override_is_authoritative() -> None:
    policy = classify_task(request("Implement a function.", profile="creative"))
    assert policy.name == "creative"
    assert policy.signals == ("explicit_override",)


def test_profiles_change_diversity_and_reliability_priors() -> None:
    exact = classify_task(request("Compute the exact probability."))
    creative = classify_task(request("Brainstorm original concepts."))

    assert exact.candidate_temperature(3, 0.45) < creative.candidate_temperature(3, 0.45)
    assert exact.reliability_prior(0.0) > exact.reliability_prior(0.75)
    assert creative.reliability_prior(0.65) > creative.reliability_prior(0.0)
