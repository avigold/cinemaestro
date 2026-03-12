"""Tests for visual prompt construction."""

from __future__ import annotations

from cinemaestro.core.scene_graph import SceneGraph
from cinemaestro.visual.prompt_builder import PromptBuilder


class TestPromptBuilder:
    def test_build_shot_prompt(self, sample_scene_graph: SceneGraph) -> None:
        builder = PromptBuilder(sample_scene_graph)
        shots = sample_scene_graph.all_shots()

        prompt = builder.build_shot_prompt(shots[0])
        assert "wide shot" in prompt.lower()
        assert "cafe" in prompt.lower()
        assert "cinematic" in prompt.lower()

    def test_prompt_includes_character_desc(
        self, sample_scene_graph: SceneGraph
    ) -> None:
        builder = PromptBuilder(sample_scene_graph)
        shot = sample_scene_graph.all_shots()[1]  # Alice entering

        prompt = builder.build_shot_prompt(shot)
        assert "brown hair" in prompt.lower()  # Alice's description

    def test_prompt_includes_style(self, sample_scene_graph: SceneGraph) -> None:
        builder = PromptBuilder(sample_scene_graph)
        shots = sample_scene_graph.all_shots()
        prompt = builder.build_shot_prompt(shots[0])
        assert "photorealistic" in prompt.lower()

    def test_build_first_frame_prompt(
        self, sample_scene_graph: SceneGraph
    ) -> None:
        builder = PromptBuilder(sample_scene_graph)
        shot = sample_scene_graph.all_shots()[1]

        prompt = builder.build_first_frame_prompt(shot)
        assert "sharp focus" in prompt.lower()
        assert "brown hair" in prompt.lower()
