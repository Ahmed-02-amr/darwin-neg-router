from __future__ import annotations

from pathlib import Path

from darwin_neg_router.config import Settings
from desktop_app.darwin_neg_control import (
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_MAX_TOKENS,
    KV_CACHE_TYPE_K,
    KV_CACHE_TYPE_V,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_general_router_uses_long_context_defaults() -> None:
    settings = Settings()
    assert settings.max_context == 163840
    assert settings.default_max_tokens == 43008
    assert settings.max_context - settings.default_max_tokens == 120832
    assert settings.review_max_tokens == 3072
    assert settings.gpqa_solver_tokens == 6144
    assert settings.gpqa_review_tokens == 6144
    assert settings.truncation_recovery_tokens == 2048


def test_desktop_profile_matches_router_and_uses_q8_kv() -> None:
    assert DEFAULT_CONTEXT_SIZE == 163840
    assert DEFAULT_MAX_TOKENS == 43008
    assert KV_CACHE_TYPE_K == "q8_0"
    assert KV_CACHE_TYPE_V == "q8_0"


def test_native_launcher_forces_q8_kv_and_flash_attention() -> None:
    launcher = (PROJECT_ROOT / "scripts" / "start-native-neg.ps1").read_text(
        encoding="utf-8"
    )
    assert "[int]$ContextSize = 163840" in launcher
    assert "[string]$CacheTypeK = 'q8_0'" in launcher
    assert "[string]$CacheTypeV = 'q8_0'" in launcher
    assert "--cache-type-k $CacheTypeK" in launcher
    assert "--cache-type-v $CacheTypeV" in launcher
    assert "--flash-attn on" in launcher
