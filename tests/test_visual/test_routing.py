"""Tests for shot-to-provider routing."""

from __future__ import annotations

from cinemaestro.config import CinemaestroConfig, ProviderConfig, VisualConfig
from cinemaestro.core.scene_graph import CameraMovement, Shot, ShotType
from cinemaestro.visual.strategies.routing import ShotRouter


class TestShotRouter:
    def test_default_provider(self) -> None:
        config = CinemaestroConfig()
        router = ShotRouter(config)
        shot = Shot(
            shot_id="test",
            shot_type=ShotType.MEDIUM,
            visual_description="test shot",
        )
        assert router.route(shot) == config.visual.default_provider

    def test_explicit_provider_hint(self) -> None:
        config = CinemaestroConfig()
        router = ShotRouter(config)
        shot = Shot(
            shot_id="test",
            shot_type=ShotType.MEDIUM,
            visual_description="test shot",
            generation_hints={"provider": "comfyui"},
        )
        assert router.route(shot) == "comfyui"

    def test_character_closeup_prefers_kling(self) -> None:
        config = CinemaestroConfig(
            visual=VisualConfig(
                providers={
                    "kling": ProviderConfig(enabled=True),
                }
            )
        )
        router = ShotRouter(config)
        shot = Shot(
            shot_id="test",
            shot_type=ShotType.CLOSE_UP,
            visual_description="character close-up",
            characters_present=["char_1"],
        )
        assert router.route(shot) == "kling"

    def test_wide_shot_no_chars_prefers_runway(self) -> None:
        config = CinemaestroConfig(
            visual=VisualConfig(
                providers={
                    "runway": ProviderConfig(enabled=True),
                }
            )
        )
        router = ShotRouter(config)
        shot = Shot(
            shot_id="test",
            shot_type=ShotType.WIDE,
            visual_description="establishing shot",
            characters_present=[],
        )
        assert router.route(shot) == "runway"

    def test_fallback(self) -> None:
        config = CinemaestroConfig(
            visual=VisualConfig(
                default_provider="runway",
                fallback_provider="comfyui",
            )
        )
        router = ShotRouter(config)
        assert router.get_fallback("runway") == "comfyui"
        assert router.get_fallback("comfyui") is None
