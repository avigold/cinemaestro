"""Tests for configuration system — key propagation, TOML loading, etc."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from cinemaestro.config import CinemaestroConfig, load_config


@pytest.fixture(autouse=True)
def _isolate_from_real_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests don't pick up the real .env file or user config."""
    monkeypatch.chdir(tmp_path)
    # Clear any API key env vars that might leak in
    for var in [
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "RUNWAY_API_KEY",
        "KLING_API_KEY", "FAL_KEY", "REPLICATE_API_TOKEN",
        "ELEVENLABS_API_KEY", "STABILITY_API_KEY",
    ]:
        monkeypatch.delenv(var, raising=False)


class TestConfigKeyPropagation:
    def test_runway_key_creates_provider(self) -> None:
        config = CinemaestroConfig(runway_api_key="rk-test-1234567890")
        assert "runway" in config.visual.providers
        assert config.visual.providers["runway"].enabled is True
        assert config.visual.providers["runway"].api_key == "rk-test-1234567890"
        assert config.visual.providers["runway"].default_model == "gen4_turbo"

    def test_elevenlabs_key_creates_tts_provider(self) -> None:
        config = CinemaestroConfig(elevenlabs_api_key="el-test-key")
        assert "elevenlabs" in config.audio.tts.providers
        assert config.audio.tts.providers["elevenlabs"].enabled is True
        assert config.audio.tts.providers["elevenlabs"].api_key == "el-test-key"

    def test_stability_key_creates_music_and_sfx(self) -> None:
        config = CinemaestroConfig(stability_api_key="sk-stab-test")
        assert "stable_audio" in config.audio.music.providers
        assert config.audio.music.providers["stable_audio"].enabled is True
        assert "stable_audio" in config.audio.sfx.providers
        assert config.audio.sfx.providers["stable_audio"].enabled is True

    def test_multiple_keys(self) -> None:
        config = CinemaestroConfig(
            runway_api_key="rk-test",
            kling_api_key="kl-test",
            fal_key="fal-test",
            elevenlabs_api_key="el-test",
        )
        assert len([p for p in config.visual.providers.values() if p.enabled]) >= 3
        assert "elevenlabs" in config.audio.tts.providers

    def test_comfyui_always_available(self) -> None:
        config = CinemaestroConfig()
        assert "comfyui" in config.visual.providers
        assert config.visual.providers["comfyui"].enabled is True

    def test_env_var_reading(self) -> None:
        with patch.dict(os.environ, {"RUNWAY_API_KEY": "env-runway-key"}):
            config = CinemaestroConfig()
            assert config.runway_api_key == "env-runway-key"
            assert config.visual.providers["runway"].api_key == "env-runway-key"

    def test_no_key_no_provider(self) -> None:
        config = CinemaestroConfig()
        # These should not have providers enabled (no keys)
        assert "runway" not in config.visual.providers or not config.visual.providers.get("runway", None)
        assert "elevenlabs" not in config.audio.tts.providers


class TestConfigSummary:
    def test_summary_masks_keys(self) -> None:
        config = CinemaestroConfig(anthropic_api_key="sk-ant-1234567890abcdef")
        summary = config.summary()
        masked = summary["api_keys"]["anthropic"]
        assert "1234567890abcdef" not in masked
        assert masked.startswith("sk-a")
        assert masked.endswith("cdef")

    def test_summary_shows_not_set(self) -> None:
        config = CinemaestroConfig()
        summary = config.summary()
        assert summary["api_keys"]["anthropic"] == "(not set)"


class TestAvailableProviders:
    def test_available_with_keys(self) -> None:
        config = CinemaestroConfig(
            runway_api_key="test",
            elevenlabs_api_key="test",
        )
        available = config.get_available_providers()
        assert "runway" in available["visual"]
        assert "elevenlabs" in available["tts"]

    def test_comfyui_always_in_visual(self) -> None:
        config = CinemaestroConfig()
        available = config.get_available_providers()
        assert "comfyui" in available["visual"]


class TestTomlLoading:
    def test_load_config_from_toml(self, tmp_path: Path) -> None:
        toml_content = """
max_budget_usd = 100.0

[story]
llm_provider = "openai"
model = "gpt-4o"

[visual]
default_provider = "kling"
character_consistency_strategy = "native"
"""
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(toml_content)

        config = load_config(config_file=config_file)
        assert config.story.llm_provider == "openai"
        assert config.story.model == "gpt-4o"
        assert config.visual.default_provider == "kling"
        assert config.max_budget_usd == 100.0

    def test_load_project_toml(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        toml_content = """
max_budget_usd = 25.0

[story]
model = "claude-opus-4-20250514"
"""
        (project_dir / "cinemaestro.toml").write_text(toml_content)

        config = load_config(project_dir=project_dir)
        assert config.story.model == "claude-opus-4-20250514"
        assert config.max_budget_usd == 25.0
