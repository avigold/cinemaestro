"""Project workspace management.

A Project represents a single film production with its own directory structure
for scenes, shots, audio, and assembly artifacts.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from cinemaestro.core.scene_graph import SceneGraph


class ProjectMetadata(BaseModel):
    """Metadata stored in project directory."""

    title: str = ""
    created_at: str = ""
    config_file: str = "cinemaestro.toml"
    scene_graph_file: str = "scene_graph.yaml"


class Project:
    """Manages the filesystem layout of a film project."""

    def __init__(self, project_dir: Path) -> None:
        self.root = project_dir

    @property
    def characters_dir(self) -> Path:
        return self.root / "characters"

    @property
    def shots_dir(self) -> Path:
        return self.root / "shots"

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    @property
    def dialogue_dir(self) -> Path:
        return self.audio_dir / "dialogue"

    @property
    def music_dir(self) -> Path:
        return self.audio_dir / "music"

    @property
    def sfx_dir(self) -> Path:
        return self.audio_dir / "sfx"

    @property
    def assembly_dir(self) -> Path:
        return self.root / "assembly"

    @property
    def export_dir(self) -> Path:
        return self.root / "export"

    @property
    def scene_graph_path(self) -> Path:
        return self.root / "scene_graph.yaml"

    @property
    def pipeline_state_path(self) -> Path:
        return self.root / "pipeline_state.json"

    @property
    def config_path(self) -> Path:
        return self.root / "cinemaestro.toml"

    def initialize(self) -> None:
        """Create the project directory structure."""
        for d in [
            self.root,
            self.characters_dir,
            self.shots_dir,
            self.audio_dir,
            self.dialogue_dir,
            self.music_dir,
            self.sfx_dir,
            self.assembly_dir,
            self.export_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def shot_dir(self, shot_id: str) -> Path:
        d = self.shots_dir / shot_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_scene_graph(self, scene_graph: SceneGraph) -> None:
        self.scene_graph_path.write_text(
            yaml.dump(
                scene_graph.model_dump(mode="json"),
                default_flow_style=False,
                sort_keys=False,
                width=120,
            )
        )

    def load_scene_graph(self) -> SceneGraph | None:
        if not self.scene_graph_path.exists():
            return None
        data = yaml.safe_load(self.scene_graph_path.read_text())
        return SceneGraph(**data)

    @property
    def exists(self) -> bool:
        return self.root.exists() and self.scene_graph_path.exists()
