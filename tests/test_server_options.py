import pytest
from fastapi import HTTPException

from darwin_neg_router.server import _routing_profile


def test_openai_and_anthropic_routing_profile_overrides() -> None:
    assert _routing_profile({"darwin": {"routing_profile": "Exact"}}) == "exact"
    assert (
        _routing_profile({"metadata": {"darwin_routing_profile": "investigation"}})
        == "investigation"
    )
    assert _routing_profile({}) is None


def test_invalid_routing_profile_is_rejected() -> None:
    with pytest.raises(HTTPException, match="routing_profile must be one of"):
        _routing_profile({"darwin": {"routing_profile": "guessy"}})
