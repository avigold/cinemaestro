"""fal.ai provider — gateway to Flux, Kling, and other models via fal.

Uses the official fal_client SDK for reliable upload, queueing, and polling.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path

import httpx

from cinemaestro.config import ProviderConfig
from cinemaestro.visual.base import GenerationResult, GenerationStatus

logger = logging.getLogger(__name__)

# Video models on fal.ai — used when generating video clips
VIDEO_MODEL = "fal-ai/kling-video/v2/master/image-to-video"
VIDEO_MODEL_TEXT = "fal-ai/kling-video/v2/master/text-to-video"

# Image models — used for first-frame generation and character refs
IMAGE_MODEL = "fal-ai/flux/dev"


class FalVideoGenerator:
    """Video generation via fal.ai's hosted models (Kling v2)."""

    provider_name = "fal"
    supports_character_reference = True
    supports_image_to_video = True
    max_duration_seconds = 10.0

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.api_key = config.api_key
        # Always use a video model — ignore shared default_model if it's an image model
        dm = config.default_model or ""
        if "video" in dm:
            self.model = dm
        else:
            self.model = VIDEO_MODEL_TEXT
        self.i2v_model = VIDEO_MODEL
        # Set env var so fal_client picks up the key
        os.environ.setdefault("FAL_KEY", self.api_key)

    async def generate_video(
        self,
        prompt: str,
        duration_seconds: float = 5.0,
        reference_images: list[Path] | None = None,
        first_frame: Path | None = None,
        last_frame: Path | None = None,
        aspect_ratio: str = "16:9",
        seed: int | None = None,
        output_dir: Path | None = None,
        **kwargs: str,
    ) -> GenerationResult:
        import fal_client

        dur = str(min(int(duration_seconds), 10))

        if first_frame and first_frame.exists():
            # Image-to-video mode
            image_url = fal_client.upload_file(first_frame)
            model = self.i2v_model
            payload: dict = {
                "prompt": prompt,
                "image_url": image_url,
                "duration": dur,
                "aspect_ratio": aspect_ratio,
            }
        else:
            # Text-to-video mode
            model = self.model
            payload = {
                "prompt": prompt,
                "duration": dur,
                "aspect_ratio": aspect_ratio,
            }

        if seed is not None:
            payload["seed"] = seed

        logger.info("Submitting video job to fal.ai: %s", model)

        # Use fal_client's queue for async generation
        result_data = await asyncio.to_thread(
            fal_client.subscribe,
            model,
            arguments=payload,
            with_logs=False,
        )

        # Extract video URL from result
        video = result_data.get("video", {})
        output_url = video.get("url", "")
        gen_id = result_data.get("request_id", str(uuid.uuid4()))

        result = GenerationResult(
            provider=self.provider_name,
            generation_id=gen_id,
            status=GenerationStatus.COMPLETED if output_url else GenerationStatus.FAILED,
            metadata={"output_url": output_url},
        )

        if result.status == GenerationStatus.COMPLETED and output_dir and output_url:
            output_path = output_dir / f"fal_{gen_id}.mp4"
            await self._download(output_url, output_path)
            result.output_path = output_path

        return result

    async def _download(self, url: str, dest: Path) -> None:
        async with httpx.AsyncClient(timeout=120) as dl_client:
            resp = await dl_client.get(url)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)

    async def check_status(self, generation_id: str) -> GenerationStatus:
        import fal_client
        status = await asyncio.to_thread(
            fal_client.status, self.model, generation_id, with_logs=False
        )
        status_str = getattr(status, "status", "PENDING") if status else "PENDING"
        return {
            "IN_QUEUE": GenerationStatus.PENDING,
            "IN_PROGRESS": GenerationStatus.PROCESSING,
            "COMPLETED": GenerationStatus.COMPLETED,
            "FAILED": GenerationStatus.FAILED,
        }.get(status_str, GenerationStatus.PENDING)


class FalImageGenerator:
    """Image generation via fal.ai (Flux, SDXL, etc.)."""

    provider_name = "fal"
    supports_lora = True
    supports_ip_adapter = True

    API_BASE = "https://fal.run"

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.api_key = config.api_key
        # Use image model — ignore shared default if it's a video model
        dm = config.default_model or ""
        if "video" in dm:
            self.model = IMAGE_MODEL
        else:
            self.model = dm or IMAGE_MODEL
        self._client: httpx.AsyncClient | None = None
        os.environ.setdefault("FAL_KEY", self.api_key)

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={"Authorization": f"Key {self.api_key}"},
                timeout=self.config.timeout_seconds,
            )
        return self._client

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
    ) -> GenerationResult:
        if lora_trigger_word and lora_trigger_word not in prompt:
            prompt = f"{lora_trigger_word}, {prompt}"

        payload: dict = {
            "prompt": prompt,
            "image_size": {"width": width, "height": height},
            "num_images": 1,
        }

        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if seed is not None:
            payload["seed"] = seed
        if lora_path:
            payload["loras"] = [{"path": str(lora_path), "scale": 1.0}]

        url = f"{self.API_BASE}/{self.model}"
        resp = await self.client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

        images = data.get("images", [])
        output_url = images[0]["url"] if images else ""
        gen_id = data.get("request_id", str(uuid.uuid4()))

        result = GenerationResult(
            provider=self.provider_name,
            generation_id=gen_id,
            status=GenerationStatus.COMPLETED if output_url else GenerationStatus.FAILED,
            seed=data.get("seed"),
            metadata={"output_url": output_url},
        )

        if result.status == GenerationStatus.COMPLETED and output_dir and output_url:
            output_path = output_dir / f"fal_{gen_id}.png"
            await self._download(output_url, output_path)
            result.output_path = output_path

        return result

    async def _download(self, url: str, dest: Path) -> None:
        async with httpx.AsyncClient() as dl_client:
            resp = await dl_client.get(url)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
