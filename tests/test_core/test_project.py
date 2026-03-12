"""Tests for Project workspace management."""

from __future__ import annotations

from pathlib import Path

from cinemaestro.core.project import Project
from cinemaestro.core.scene_graph import SceneGraph


class TestProject:
    def test_initialize(self, tmp_path: Path) -> None:
        project = Project(tmp_path / "my_film")
        project.initialize()

        assert project.root.exists()
        assert project.characters_dir.exists()
        assert project.shots_dir.exists()
        assert project.dialogue_dir.exists()
        assert project.music_dir.exists()
        assert project.sfx_dir.exists()
        assert project.assembly_dir.exists()
        assert project.export_dir.exists()

    def test_save_and_load_scene_graph(
        self, tmp_path: Path, sample_scene_graph: SceneGraph
    ) -> None:
        project = Project(tmp_path / "my_film")
        project.initialize()

        project.save_scene_graph(sample_scene_graph)
        assert project.scene_graph_path.exists()

        loaded = project.load_scene_graph()
        assert loaded is not None
        assert loaded.title == sample_scene_graph.title
        assert loaded.total_shots == sample_scene_graph.total_shots

    def test_load_nonexistent_scene_graph(self, tmp_path: Path) -> None:
        project = Project(tmp_path / "empty")
        assert project.load_scene_graph() is None

    def test_shot_dir(self, tmp_path: Path) -> None:
        project = Project(tmp_path / "my_film")
        project.initialize()

        shot_dir = project.shot_dir("act1_s1_shot1")
        assert shot_dir.exists()
        assert shot_dir.name == "act1_s1_shot1"
