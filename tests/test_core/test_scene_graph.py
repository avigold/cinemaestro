"""Tests for SceneGraph data model."""

from __future__ import annotations

import json

import yaml

from cinemaestro.core.scene_graph import SceneGraph, ShotType


class TestSceneGraph:
    def test_all_shots(self, sample_scene_graph: SceneGraph) -> None:
        shots = sample_scene_graph.all_shots()
        assert len(shots) == 4
        assert shots[0].shot_id == "act1_s1_shot1"
        assert shots[-1].shot_id == "act1_s1_shot4"

    def test_shots_for_character(self, sample_scene_graph: SceneGraph) -> None:
        alice_shots = sample_scene_graph.shots_for_character("char_alice")
        assert len(alice_shots) == 2
        assert all("char_alice" in s.characters_present for s in alice_shots)

        bob_shots = sample_scene_graph.shots_for_character("char_bob")
        assert len(bob_shots) == 2

    def test_character_ids(self, sample_scene_graph: SceneGraph) -> None:
        ids = sample_scene_graph.character_ids()
        assert ids == {"char_alice", "char_bob"}

    def test_total_duration(self, sample_scene_graph: SceneGraph) -> None:
        assert sample_scene_graph.total_duration_seconds == 14.0

    def test_total_shots(self, sample_scene_graph: SceneGraph) -> None:
        assert sample_scene_graph.total_shots == 4

    def test_total_dialogue_lines(self, sample_scene_graph: SceneGraph) -> None:
        assert sample_scene_graph.total_dialogue_lines == 2

    def test_yaml_roundtrip(self, sample_scene_graph: SceneGraph) -> None:
        """SceneGraph should survive YAML serialization/deserialization."""
        data = sample_scene_graph.model_dump(mode="json")
        yaml_str = yaml.dump(data, default_flow_style=False)
        loaded = yaml.safe_load(yaml_str)
        restored = SceneGraph(**loaded)

        assert restored.title == sample_scene_graph.title
        assert restored.total_shots == sample_scene_graph.total_shots
        assert len(restored.characters) == len(sample_scene_graph.characters)

    def test_json_roundtrip(self, sample_scene_graph: SceneGraph) -> None:
        json_str = sample_scene_graph.model_dump_json()
        restored = SceneGraph.model_validate_json(json_str)
        assert restored.total_shots == sample_scene_graph.total_shots

    def test_get_location(self, sample_scene_graph: SceneGraph) -> None:
        loc = sample_scene_graph.get_location("loc_cafe")
        assert loc is not None
        assert loc.name == "Corner Cafe"

        assert sample_scene_graph.get_location("nonexistent") is None

    def test_get_character(self, sample_scene_graph: SceneGraph) -> None:
        char = sample_scene_graph.get_character("char_alice")
        assert char is not None
        assert char.name == "Alice"

        assert sample_scene_graph.get_character("nonexistent") is None

    def test_shot_has_dialogue(self, sample_scene_graph: SceneGraph) -> None:
        shots = sample_scene_graph.all_shots()
        assert not shots[0].has_dialogue  # establishing shot
        assert shots[2].has_dialogue  # Bob's dialogue
