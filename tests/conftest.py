"""Shared test fixtures for Cinemaestro."""

from __future__ import annotations

from pathlib import Path

import pytest

from cinemaestro.core.scene_graph import (
    Act,
    CameraMovement,
    CharacterAppearance,
    DialogueLine,
    Location,
    Scene,
    SceneGraph,
    Shot,
    ShotType,
    SoundEffect,
    TransitionType,
)


@pytest.fixture
def sample_scene_graph() -> SceneGraph:
    """A minimal but complete SceneGraph for testing."""
    return SceneGraph(
        title="Test Film",
        logline="A test film for unit testing.",
        genre="drama",
        tone="neutral",
        style="photorealistic",
        target_duration_seconds=30.0,
        characters=[
            CharacterAppearance(
                character_id="char_alice",
                name="Alice",
                physical_description="young woman with short brown hair and green eyes",
                wardrobe="blue denim jacket over white t-shirt",
            ),
            CharacterAppearance(
                character_id="char_bob",
                name="Bob",
                physical_description="tall man with grey beard and glasses",
                wardrobe="brown corduroy blazer",
            ),
        ],
        locations=[
            Location(
                location_id="loc_cafe",
                name="Corner Cafe",
                description="cozy corner cafe with warm lighting and wooden tables",
            ),
        ],
        acts=[
            Act(
                act_number=1,
                title="Meeting",
                scenes=[
                    Scene(
                        scene_id="act1_s1",
                        scene_number=1,
                        title="First Encounter",
                        description="Alice enters the cafe and notices Bob.",
                        location_id="loc_cafe",
                        time_of_day="afternoon",
                        mood="curious",
                        music_direction="soft piano, contemplative",
                        shots=[
                            Shot(
                                shot_id="act1_s1_shot1",
                                shot_type=ShotType.WIDE,
                                camera_movement=CameraMovement.STATIC,
                                duration_seconds=4.0,
                                visual_description="Interior of a warm cafe, afternoon light streaming through windows",
                                characters_present=[],
                                mood="peaceful",
                                lighting="warm natural",
                                location_id="loc_cafe",
                                time_of_day="afternoon",
                            ),
                            Shot(
                                shot_id="act1_s1_shot2",
                                shot_type=ShotType.MEDIUM,
                                camera_movement=CameraMovement.DOLLY_IN,
                                duration_seconds=3.0,
                                visual_description="Alice pushes open the cafe door and looks around",
                                characters_present=["char_alice"],
                                mood="curious",
                                lighting="warm natural",
                                location_id="loc_cafe",
                                time_of_day="afternoon",
                            ),
                            Shot(
                                shot_id="act1_s1_shot3",
                                shot_type=ShotType.CLOSE_UP,
                                camera_movement=CameraMovement.STATIC,
                                duration_seconds=3.0,
                                visual_description="Bob looks up from his book, noticing Alice",
                                characters_present=["char_bob"],
                                dialogue=[
                                    DialogueLine(
                                        character_id="char_bob",
                                        text="Can I help you?",
                                        emotion="curious",
                                    ),
                                ],
                                mood="intrigued",
                                lighting="warm natural",
                                location_id="loc_cafe",
                                time_of_day="afternoon",
                            ),
                            Shot(
                                shot_id="act1_s1_shot4",
                                shot_type=ShotType.OVER_THE_SHOULDER,
                                camera_movement=CameraMovement.STATIC,
                                duration_seconds=4.0,
                                visual_description="Alice smiles and walks toward Bob's table",
                                characters_present=["char_alice", "char_bob"],
                                dialogue=[
                                    DialogueLine(
                                        character_id="char_alice",
                                        text="I think you can.",
                                        emotion="confident",
                                    ),
                                ],
                                sound_effects=[
                                    SoundEffect(
                                        description="chair scraping on floor",
                                        intensity=0.3,
                                    ),
                                ],
                                mood="warm",
                                lighting="warm natural",
                                location_id="loc_cafe",
                                time_of_day="afternoon",
                                transition_out=TransitionType.FADE_TO_BLACK,
                            ),
                        ],
                    ),
                ],
            ),
        ],
        music_theme="gentle acoustic, hopeful",
        color_grade="warm golden tones",
    )


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary project directory."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    return project_dir
