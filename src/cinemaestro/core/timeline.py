"""Timeline models for video assembly.

The Timeline is the assembly-stage representation: an ordered list of tracks
(video, dialogue, music, sfx) with clips placed at specific timestamps.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class Clip(BaseModel):
    """A single media clip placed on a timeline track."""

    clip_id: str
    source_path: str  # path to media file
    start_time: float  # position on timeline in seconds
    duration: float
    in_point: float = 0.0  # trim: start within source
    out_point: float | None = None  # trim: end within source
    volume: float = 1.0  # audio volume multiplier
    opacity: float = 1.0  # video opacity
    transition_in: str = "cut"
    transition_in_duration: float = 0.0
    transition_out: str = "cut"
    transition_out_duration: float = 0.0
    metadata: dict[str, str] = Field(default_factory=dict)


class Track(BaseModel):
    """A single track in the timeline (video, dialogue, music, or sfx)."""

    track_id: str
    track_type: str  # "video", "dialogue", "music", "sfx"
    clips: list[Clip] = Field(default_factory=list)
    volume: float = 1.0  # track-level volume

    def add_clip(self, clip: Clip) -> None:
        self.clips.append(clip)
        self.clips.sort(key=lambda c: c.start_time)

    @property
    def duration(self) -> float:
        if not self.clips:
            return 0.0
        last = max(self.clips, key=lambda c: c.start_time + c.duration)
        return last.start_time + last.duration


class Timeline(BaseModel):
    """Complete timeline for a film — ready for assembly into final video."""

    film_title: str = ""
    fps: float = 24.0
    resolution: tuple[int, int] = (1920, 1080)
    tracks: list[Track] = Field(default_factory=list)

    def get_track(self, track_type: str) -> Track | None:
        for track in self.tracks:
            if track.track_type == track_type:
                return track
        return None

    def get_or_create_track(self, track_id: str, track_type: str) -> Track:
        existing = self.get_track(track_type)
        if existing:
            return existing
        track = Track(track_id=track_id, track_type=track_type)
        self.tracks.append(track)
        return track

    @property
    def duration(self) -> float:
        if not self.tracks:
            return 0.0
        return max(track.duration for track in self.tracks)
