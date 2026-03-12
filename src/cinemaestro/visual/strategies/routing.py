"""Shot-to-provider routing logic.

Determines which video/image generation provider to use for each shot
based on shot characteristics, provider capabilities, and cost.
"""

from __future__ import annotations

from cinemaestro.config import CinemaestroConfig
from cinemaestro.core.scene_graph import Shot, ShotType


class ShotRouter:
    """Routes shots to the optimal video generation provider."""

    def __init__(self, config: CinemaestroConfig) -> None:
        self.config = config
        self.default = config.visual.default_provider
        self.fallback = config.visual.fallback_provider

    def route(self, shot: Shot) -> str:
        """Determine the best provider for a given shot.

        Routing logic:
        - Dialogue close-ups with characters -> provider with best character ref
        - Wide/establishing shots without characters -> cheapest high-quality provider
        - Shots with explicit provider hints -> honor the hint
        - Everything else -> default provider
        """
        # Explicit override
        if "provider" in shot.generation_hints:
            return shot.generation_hints["provider"]

        has_characters = len(shot.characters_present) > 0

        # Character-heavy shots benefit from native reference support
        if has_characters and shot.shot_type in (
            ShotType.CLOSE_UP,
            ShotType.EXTREME_CLOSE_UP,
            ShotType.MEDIUM_CLOSE,
            ShotType.MEDIUM,
            ShotType.OVER_THE_SHOULDER,
            ShotType.TWO_SHOT,
        ):
            # Kling's Elements system is best for character consistency
            if self._provider_enabled("kling"):
                return "kling"
            return self.default

        # Wide/establishing shots — any provider works, prefer Runway for cinematic quality
        if shot.shot_type in (
            ShotType.EXTREME_WIDE,
            ShotType.WIDE,
            ShotType.AERIAL,
        ) and not has_characters:
            if self._provider_enabled("runway"):
                return "runway"
            return self.default

        return self.default

    def get_fallback(self, primary: str) -> str | None:
        """Get the fallback provider for a given primary provider."""
        if primary == self.fallback:
            return None
        return self.fallback

    def _provider_enabled(self, name: str) -> bool:
        providers = self.config.visual.providers
        return name in providers and providers[name].enabled
