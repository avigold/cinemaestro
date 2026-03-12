"""Story writer protocol."""

from __future__ import annotations

from typing import Protocol

from cinemaestro.core.scene_graph import SceneGraph


class StoryWriter(Protocol):
    """Generates a SceneGraph from a story concept."""

    async def write(
        self,
        concept: str,
        target_duration_seconds: float = 120.0,
        genre: str = "",
        tone: str = "",
        style: str = "",
        existing_characters: list[str] | None = None,
    ) -> SceneGraph: ...
