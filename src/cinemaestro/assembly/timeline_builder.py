"""Converts a SceneGraph + generated assets into a Timeline for assembly."""

from __future__ import annotations

import logging
from pathlib import Path

from cinemaestro.core.project import Project
from cinemaestro.core.scene_graph import SceneGraph, Shot
from cinemaestro.core.timeline import Clip, Timeline, Track

logger = logging.getLogger(__name__)


class TimelineBuilder:
    """Builds a Timeline from a SceneGraph and generated asset files."""

    def __init__(self, project: Project) -> None:
        self.project = project

    def build(self, scene_graph: SceneGraph) -> Timeline:
        """Build a complete timeline from the scene graph and generated assets."""
        timeline = Timeline(
            film_title=scene_graph.title,
            resolution=(1920, 1080),
        )

        video_track = timeline.get_or_create_track("video_main", "video")
        dialogue_track = timeline.get_or_create_track("dialogue_main", "dialogue")
        music_track = timeline.get_or_create_track("music_main", "music")
        sfx_track = timeline.get_or_create_track("sfx_main", "sfx")

        current_time = 0.0

        for shot in scene_graph.all_shots():
            shot_dir = self.project.shots_dir / shot.shot_id

            # Video clip
            video_file = self._find_video_file(shot_dir, shot.shot_id)
            if video_file:
                video_clip = Clip(
                    clip_id=f"v_{shot.shot_id}",
                    source_path=str(video_file),
                    start_time=current_time,
                    duration=shot.duration_seconds,
                    transition_in=shot.transition_in.value,
                    transition_out=shot.transition_out.value,
                )
                video_track.add_clip(video_clip)

            # Dialogue clips
            dialogue_offset = 0.0
            for i, line in enumerate(shot.dialogue):
                dialogue_file = (
                    self.project.dialogue_dir
                    / f"{shot.shot_id}_line{i}.wav"
                )
                if dialogue_file.exists():
                    from pydub import AudioSegment

                    audio = AudioSegment.from_file(str(dialogue_file))
                    line_duration = len(audio) / 1000.0

                    dialogue_clip = Clip(
                        clip_id=f"d_{shot.shot_id}_{i}",
                        source_path=str(dialogue_file),
                        start_time=current_time + dialogue_offset,
                        duration=line_duration,
                        metadata={"character": line.character_id, "text": line.text},
                    )
                    dialogue_track.add_clip(dialogue_clip)
                    dialogue_offset += line_duration + 0.3  # small pause between lines

            # SFX clips
            for i, sfx in enumerate(shot.sound_effects):
                sfx_file = self.project.sfx_dir / f"{shot.shot_id}_sfx{i}.wav"
                if sfx_file.exists():
                    sfx_clip = Clip(
                        clip_id=f"sfx_{shot.shot_id}_{i}",
                        source_path=str(sfx_file),
                        start_time=current_time + sfx.timestamp_hint,
                        duration=sfx.duration_hint or 3.0,
                        volume=sfx.intensity,
                    )
                    sfx_track.add_clip(sfx_clip)

            current_time += shot.duration_seconds

        # Music — lay across full timeline for now
        self._add_music_tracks(scene_graph, music_track)

        return timeline

    def _find_video_file(self, shot_dir: Path, shot_id: str) -> Path | None:
        """Find the best video file for a shot (corrected > raw)."""
        corrected = shot_dir / "corrected.mp4"
        if corrected.exists():
            return corrected

        raw = shot_dir / "raw.mp4"
        if raw.exists():
            return raw

        # Check for any video file
        for ext in (".mp4", ".webm", ".mov"):
            for f in shot_dir.glob(f"*{ext}"):
                return f

        logger.warning("No video file found for shot %s", shot_id)
        return None

    def _add_music_tracks(self, scene_graph: SceneGraph, music_track: Track) -> None:
        """Add music clips to the timeline based on scene music directions."""
        current_time = 0.0
        for scene in scene_graph.all_scenes():
            music_file = self.project.music_dir / f"{scene.scene_id}.wav"
            if music_file.exists():
                music_clip = Clip(
                    clip_id=f"music_{scene.scene_id}",
                    source_path=str(music_file),
                    start_time=current_time,
                    duration=scene.duration_seconds,
                    transition_in="dissolve",
                    transition_in_duration=1.0,
                    transition_out="dissolve",
                    transition_out_duration=1.0,
                )
                music_track.add_clip(music_clip)
            current_time += scene.duration_seconds
