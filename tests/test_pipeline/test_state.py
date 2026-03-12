"""Tests for pipeline state management."""

from __future__ import annotations

from pathlib import Path

from cinemaestro.pipeline.state import PipelineState, StageStatus


class TestPipelineState:
    def test_mark_stage_lifecycle(self) -> None:
        state = PipelineState()
        state.mark_stage_started("story")
        assert state.stages["story"].status == StageStatus.IN_PROGRESS
        assert state.current_stage == "story"

        state.mark_stage_completed("story")
        assert state.stages["story"].status == StageStatus.COMPLETED
        assert state.is_stage_completed("story")

    def test_mark_stage_failed(self) -> None:
        state = PipelineState()
        state.mark_stage_started("visual")
        state.mark_stage_failed("visual", "API error")
        assert state.stages["visual"].status == StageStatus.FAILED
        assert state.stages["visual"].error == "API error"

    def test_save_and_load(self, tmp_path: Path) -> None:
        state = PipelineState()
        state.mark_stage_started("story")
        state.mark_stage_completed("story")
        state.mark_shot("shot1", status=StageStatus.COMPLETED, provider="runway")
        state.add_cost(0.50)

        path = tmp_path / "state.json"
        state.save(path)

        loaded = PipelineState.load(path)
        assert loaded.is_stage_completed("story")
        assert loaded.shots["shot1"].provider == "runway"
        assert loaded.total_cost_usd == 0.50

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        state = PipelineState.load(tmp_path / "nonexistent.json")
        assert state.total_cost_usd == 0.0
        assert len(state.stages) == 0

    def test_cost_tracking(self) -> None:
        state = PipelineState()
        state.add_cost(0.10)
        state.add_cost(0.25)
        state.add_cost(0.15)
        assert abs(state.total_cost_usd - 0.50) < 0.001
