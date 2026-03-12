"""Video editing and assembly using MoviePy and FFmpeg."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from cinemaestro.config import AssemblyConfig, ExportConfig
from cinemaestro.core.timeline import Timeline

logger = logging.getLogger(__name__)


class VideoEditor:
    """Assembles a Timeline into a final video file."""

    def __init__(
        self,
        assembly_config: AssemblyConfig,
        export_config: ExportConfig,
    ) -> None:
        self.assembly_config = assembly_config
        self.export_config = export_config

    def assemble(
        self,
        timeline: Timeline,
        audio_mix_path: Path | None,
        output_path: Path,
    ) -> Path:
        """Assemble video clips from timeline into a single video with audio."""
        from moviepy import (
            AudioFileClip,
            CompositeVideoClip,
            VideoFileClip,
            concatenate_videoclips,
        )

        video_track = timeline.get_track("video")
        if not video_track or not video_track.clips:
            raise ValueError("No video clips in timeline")

        # Load and sequence video clips
        video_clips = []
        for clip in video_track.clips:
            if not Path(clip.source_path).exists():
                logger.warning("Video file missing: %s", clip.source_path)
                continue

            vc = VideoFileClip(clip.source_path)

            # Trim to specified duration
            if clip.duration and clip.duration < vc.duration:
                vc = vc.subclipped(clip.in_point, clip.in_point + clip.duration)

            # Apply transitions
            if clip.transition_in == "dissolve" and clip.transition_in_duration > 0:
                vc = vc.with_effects([
                    self._fade_in(clip.transition_in_duration)
                ])
            if clip.transition_out == "dissolve" and clip.transition_out_duration > 0:
                vc = vc.with_effects([
                    self._fade_out(clip.transition_out_duration)
                ])

            video_clips.append(vc)

        if not video_clips:
            raise ValueError("No valid video clips found")

        # Concatenate all clips
        final_video = concatenate_videoclips(video_clips, method="compose")

        # Add audio mix
        if audio_mix_path and audio_mix_path.exists():
            audio = AudioFileClip(str(audio_mix_path))
            # Match audio duration to video
            if audio.duration > final_video.duration:
                audio = audio.subclipped(0, final_video.duration)
            final_video = final_video.with_audio(audio)

        # Render
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_video.write_videofile(
            str(output_path),
            fps=self.assembly_config.fps,
            codec=self.export_config.codec,
            audio_codec=self.export_config.audio_codec,
            preset="medium",
            threads=4,
        )

        # Clean up
        final_video.close()
        for vc in video_clips:
            vc.close()

        logger.info("Assembled video: %s", output_path)
        return output_path

    def add_subtitles(
        self,
        video_path: Path,
        subtitle_path: Path,
        output_path: Path,
    ) -> Path:
        """Burn subtitles into a video using FFmpeg."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"subtitles={subtitle_path}",
            "-c:a", "copy",
            "-y",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path

    @staticmethod
    def _fade_in(duration: float):  # type: ignore[no-untyped-def]
        from moviepy.video.fx import FadeIn
        return FadeIn(duration)

    @staticmethod
    def _fade_out(duration: float):  # type: ignore[no-untyped-def]
        from moviepy.video.fx import FadeOut
        return FadeOut(duration)


class SubtitleGenerator:
    """Generates SRT subtitle files from dialogue in the timeline."""

    def generate_srt(self, timeline: Timeline, output_path: Path) -> Path:
        """Generate an SRT subtitle file from dialogue clips."""
        dialogue_track = timeline.get_track("dialogue")
        if not dialogue_track:
            return output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []

        for i, clip in enumerate(dialogue_track.clips, 1):
            start = self._format_srt_time(clip.start_time)
            end = self._format_srt_time(clip.start_time + clip.duration)
            text = clip.metadata.get("text", "")
            character = clip.metadata.get("character", "")

            if text:
                display = f"{character.upper()}: {text}" if character else text
                lines.append(f"{i}")
                lines.append(f"{start} --> {end}")
                lines.append(display)
                lines.append("")

        output_path.write_text("\n".join(lines))
        return output_path

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
