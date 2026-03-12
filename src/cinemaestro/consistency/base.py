"""Consistency checker protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class ConsistencyScore:
    """Consistency verification result for a single shot."""

    shot_id: str
    character_id: str
    mean_similarity: float  # 0.0 - 1.0
    min_similarity: float
    max_similarity: float
    frames_checked: int
    frames_below_threshold: int
    passed: bool
    failing_frame_indices: list[int] = field(default_factory=list)


@dataclass
class ConsistencyReport:
    """Full consistency report for a film."""

    scores: list[ConsistencyScore] = field(default_factory=list)
    overall_score: float = 0.0
    shots_needing_repair: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.shots_needing_repair) == 0


class ConsistencyChecker(Protocol):
    """Verifies character identity consistency across video frames."""

    async def check_shot(
        self,
        video_path: Path,
        character_id: str,
        reference_embedding_path: Path,
        threshold: float = 0.55,
        sample_interval: float = 0.5,
    ) -> ConsistencyScore: ...

    async def repair_shot(
        self,
        video_path: Path,
        character_id: str,
        reference_image: Path,
        failing_frames: list[int],
        output_path: Path,
    ) -> Path: ...
