"""Tests for Timeline models."""

from __future__ import annotations

from cinemaestro.core.timeline import Clip, Timeline, Track


class TestTimeline:
    def test_track_duration(self) -> None:
        track = Track(track_id="v1", track_type="video")
        track.add_clip(Clip(clip_id="c1", source_path="a.mp4", start_time=0, duration=4))
        track.add_clip(Clip(clip_id="c2", source_path="b.mp4", start_time=4, duration=3))
        assert track.duration == 7.0

    def test_clips_sorted_by_start_time(self) -> None:
        track = Track(track_id="v1", track_type="video")
        track.add_clip(Clip(clip_id="c2", source_path="b.mp4", start_time=5, duration=3))
        track.add_clip(Clip(clip_id="c1", source_path="a.mp4", start_time=0, duration=4))
        assert track.clips[0].clip_id == "c1"

    def test_timeline_duration(self) -> None:
        timeline = Timeline(film_title="Test")
        video = timeline.get_or_create_track("v1", "video")
        video.add_clip(Clip(clip_id="c1", source_path="a.mp4", start_time=0, duration=10))

        audio = timeline.get_or_create_track("a1", "dialogue")
        audio.add_clip(Clip(clip_id="d1", source_path="d.wav", start_time=2, duration=3))

        assert timeline.duration == 10.0

    def test_get_or_create_track(self) -> None:
        timeline = Timeline()
        track1 = timeline.get_or_create_track("v1", "video")
        track2 = timeline.get_or_create_track("v2", "video")
        assert track1 is track2  # same type returns existing

    def test_empty_timeline(self) -> None:
        timeline = Timeline()
        assert timeline.duration == 0.0
