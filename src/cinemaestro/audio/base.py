"""Audio provider protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class AudioResult:
    """Result from an audio generation call."""

    provider: str
    output_path: Path | None = None
    duration_seconds: float = 0.0
    sample_rate: int = 44100
    cost_usd: float = 0.0
    error_message: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class TextToSpeech(Protocol):
    """Generates speech audio from text with a specific voice."""

    provider_name: str
    supports_voice_cloning: bool
    supports_emotion_control: bool

    async def synthesize(
        self,
        text: str,
        voice_id: str | None = None,
        voice_reference: Path | None = None,
        emotion: str = "neutral",
        speed: float = 1.0,
        output_path: Path | None = None,
    ) -> AudioResult: ...

    async def clone_voice(
        self,
        reference_audio: Path,
        voice_name: str,
    ) -> str: ...  # returns voice_id


@runtime_checkable
class MusicGenerator(Protocol):
    """Generates music from a text description."""

    provider_name: str

    async def generate(
        self,
        prompt: str,
        duration_seconds: float = 30.0,
        output_path: Path | None = None,
        **kwargs: str,
    ) -> AudioResult: ...


@runtime_checkable
class SFXGenerator(Protocol):
    """Generates sound effects from a text description."""

    provider_name: str

    async def generate(
        self,
        description: str,
        duration_seconds: float = 5.0,
        output_path: Path | None = None,
        **kwargs: str,
    ) -> AudioResult: ...
