"""Provider protocols for visual generation.

All video/image generation backends implement these protocols, allowing
transparent provider swapping without changing pipeline logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable


class GenerationStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GenerationResult:
    """Result from an image or video generation call."""

    provider: str
    generation_id: str
    status: GenerationStatus
    output_path: Path | None = None
    seed: int | None = None
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    error_message: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class VideoGenerator(Protocol):
    """Generates a video clip from a prompt and optional reference inputs."""

    provider_name: str
    supports_character_reference: bool
    supports_image_to_video: bool
    max_duration_seconds: float

    async def generate_video(
        self,
        prompt: str,
        duration_seconds: float = 4.0,
        reference_images: list[Path] | None = None,
        first_frame: Path | None = None,
        last_frame: Path | None = None,
        aspect_ratio: str = "16:9",
        seed: int | None = None,
        output_dir: Path | None = None,
        **kwargs: str,
    ) -> GenerationResult: ...

    async def check_status(self, generation_id: str) -> GenerationStatus: ...


@runtime_checkable
class ImageGenerator(Protocol):
    """Generates a still image from a prompt and optional references."""

    provider_name: str
    supports_lora: bool
    supports_ip_adapter: bool

    async def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        reference_images: list[Path] | None = None,
        lora_path: Path | None = None,
        lora_trigger_word: str = "",
        width: int = 1920,
        height: int = 1080,
        seed: int | None = None,
        output_dir: Path | None = None,
        **kwargs: str,
    ) -> GenerationResult: ...
