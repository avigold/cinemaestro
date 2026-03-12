"""Pipeline state management and checkpointing.

Tracks which stages have completed and which shots have been generated,
enabling crash recovery and partial re-runs.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class StageStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ShotStatus(BaseModel):
    shot_id: str
    status: StageStatus = StageStatus.PENDING
    provider: str = ""
    generation_id: str = ""
    cost_usd: float = 0.0
    consistency_score: float | None = None
    repaired: bool = False
    error: str = ""


class StageState(BaseModel):
    name: str
    status: StageStatus = StageStatus.PENDING
    started_at: str = ""
    completed_at: str = ""
    error: str = ""


class PipelineState(BaseModel):
    """Persistent state of a pipeline run — saved to disk for crash recovery."""

    stages: dict[str, StageState] = Field(default_factory=dict)
    shots: dict[str, ShotStatus] = Field(default_factory=dict)
    total_cost_usd: float = 0.0
    current_stage: str = ""

    def mark_stage_started(self, stage: str) -> None:
        from datetime import datetime, timezone

        self.stages[stage] = StageState(
            name=stage,
            status=StageStatus.IN_PROGRESS,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self.current_stage = stage

    def mark_stage_completed(self, stage: str) -> None:
        from datetime import datetime, timezone

        if stage in self.stages:
            self.stages[stage].status = StageStatus.COMPLETED
            self.stages[stage].completed_at = datetime.now(timezone.utc).isoformat()

    def mark_stage_failed(self, stage: str, error: str) -> None:
        if stage in self.stages:
            self.stages[stage].status = StageStatus.FAILED
            self.stages[stage].error = error

    def is_stage_completed(self, stage: str) -> bool:
        return self.stages.get(stage, StageState(name=stage)).status == StageStatus.COMPLETED

    def mark_shot(self, shot_id: str, **kwargs: object) -> None:
        if shot_id not in self.shots:
            self.shots[shot_id] = ShotStatus(shot_id=shot_id)
        for k, v in kwargs.items():
            setattr(self.shots[shot_id], k, v)

    def add_cost(self, amount: float) -> None:
        self.total_cost_usd += amount

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> PipelineState:
        if not path.exists():
            return cls()
        return cls.model_validate_json(path.read_text())
