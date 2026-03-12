"""Audio mixing — combines dialogue, music, and SFX into a final mix."""

from __future__ import annotations

import logging
from pathlib import Path

from pydub import AudioSegment

from cinemaestro.config import AudioConfig
from cinemaestro.core.timeline import Timeline

logger = logging.getLogger(__name__)


class AudioMixer:
    """Mixes audio tracks (dialogue, music, SFX) into a final stereo mix."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config

    def mix(self, timeline: Timeline, output_path: Path) -> Path:
        """Mix all audio tracks from the timeline into a single file.

        Priority order: dialogue > SFX > music
        """
        duration_ms = int(timeline.duration * 1000)
        if duration_ms == 0:
            raise ValueError("Timeline has zero duration")

        # Create silent base track
        mix = AudioSegment.silent(duration=duration_ms, frame_rate=self.config.sample_rate)

        # Layer tracks
        for track in timeline.tracks:
            if track.track_type == "video":
                continue

            volume_multiplier = track.volume
            if track.track_type == "dialogue":
                volume_multiplier *= self.config.dialogue_volume
            elif track.track_type == "music":
                volume_multiplier *= self.config.music_volume
            elif track.track_type == "sfx":
                volume_multiplier *= self.config.sfx_volume

            for clip in track.clips:
                if not clip.source_path or not Path(clip.source_path).exists():
                    logger.warning("Missing audio file: %s", clip.source_path)
                    continue

                audio = AudioSegment.from_file(clip.source_path)

                # Apply clip volume
                volume_db = self._volume_to_db(clip.volume * volume_multiplier)
                audio = audio + volume_db

                # Trim if needed
                if clip.in_point > 0:
                    audio = audio[int(clip.in_point * 1000):]
                if clip.out_point is not None:
                    end_ms = int((clip.out_point - clip.in_point) * 1000)
                    audio = audio[:end_ms]

                # Apply crossfades for transitions
                if clip.transition_in == "dissolve" and clip.transition_in_duration > 0:
                    fade_ms = int(clip.transition_in_duration * 1000)
                    audio = audio.fade_in(fade_ms)
                if clip.transition_out == "dissolve" and clip.transition_out_duration > 0:
                    fade_ms = int(clip.transition_out_duration * 1000)
                    audio = audio.fade_out(fade_ms)

                # Overlay at the correct position
                position_ms = int(clip.start_time * 1000)
                mix = mix.overlay(audio, position=position_ms)

        # Master volume
        master_db = self._volume_to_db(self.config.master_volume)
        mix = mix + master_db

        # Normalize to prevent clipping
        mix = self._normalize(mix)

        # Export
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mix.export(str(output_path), format="wav")
        logger.info("Mixed audio exported to %s (%.1fs)", output_path, len(mix) / 1000)

        return output_path

    @staticmethod
    def _volume_to_db(volume: float) -> float:
        """Convert a 0-1 volume multiplier to dB adjustment."""
        if volume <= 0:
            return -120
        import math
        return 20 * math.log10(volume)

    @staticmethod
    def _normalize(audio: AudioSegment, target_dbfs: float = -3.0) -> AudioSegment:
        """Normalize audio to target dBFS."""
        change = target_dbfs - audio.dBFS
        return audio + change
