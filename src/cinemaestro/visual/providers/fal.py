"""fal.ai provider — gateway to Flux, Kling, and other models via fal."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import httpx

from cinemaestro.config import ProviderConfig
from cinemaestro.visual.base import GenerationResult, GenerationStatus


class FalVideoGenerator:
    """Video generation via fal.ai's hosted models."""

    provider_name = "fal"
    supports_character_reference = True
    supports_image_to_video = True
    max_duration_seconds = 10.0

    API_BASE = "https://queue.fal.run"

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.api_key = config.api_key
        self.model = config.default_model or "fal-ai/kling-video/v2/master/image-to-video"
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={"Authorization": f"Key {self.api_key}"},
                timeout=self.config.timeout_seconds,
            )
        return self._client

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
        payload: dict = {
            "prompt": prompt,
            "duration": str(min(int(duration_seconds), 10)),
            "aspect_ratio": aspect_ratio,
        }

        if first_frame:
            upload_url = await self._upload_image(first_frame)
            payload["image_url"] = upload_url

        if seed is not None:
            payload["seed"] = seed

        url = f"{self.API_BASE}/{self.model}"
        resp = await self.client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        request_id = data.get("request_id", str(uuid.uuid4()))

        result = await self._poll_result(request_id)

        if result.status == GenerationStatus.COMPLETED and output_dir:
            output_path = output_dir / f"fal_{request_id}.mp4"
            output_url = result.metadata.get("output_url", "")
            if output_url:
                await self._download(output_url, output_path)
                result.output_path = output_path

        return result

    async def _upload_image(self, image_path: Path) -> str:
        """Upload an image to fal's CDN and return the URL."""
        upload_url = "https://fal.run/fal-ai/workflows/upload"
        files = {"file": (image_path.name, image_path.read_bytes())}
        resp = await self.client.post(upload_url, files=files)
        resp.raise_for_status()
        return resp.json().get("url", "")

    async def _poll_result(
        self, request_id: str, poll_interval: float = 5.0
    ) -> GenerationResult:
        status_url = f"https://queue.fal.run/{self.model}/requests/{request_id}/status"
        result_url = f"https://queue.fal.run/{self.model}/requests/{request_id}"

        while True:
            resp = await self.client.get(status_url)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "")

            if status == "COMPLETED":
                resp = await self.client.get(result_url)
                resp.raise_for_status()
                result_data = resp.json()
                video = result_data.get("video", {})
                output_url = video.get("url", "")
                return GenerationResult(
                    provider=self.provider_name,
                    generation_id=request_id,
                    status=GenerationStatus.COMPLETED,
                    metadata={"output_url": output_url},
                )
            elif status in ("FAILED", "ERROR"):
                return GenerationResult(
                    provider=self.provider_name,
                    generation_id=request_id,
                    status=GenerationStatus.FAILED,
                    error_message=data.get("error", "Unknown error"),
                )

            await asyncio.sleep(poll_interval)

    async def _download(self, url: str, dest: Path) -> None:
        async with httpx.AsyncClient() as dl_client:
            resp = await dl_client.get(url)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)

    async def check_status(self, generation_id: str) -> GenerationStatus:
        url = f"https://queue.fal.run/{self.model}/requests/{generation_id}/status"
        resp = await self.client.get(url)
        resp.raise_for_status()
        status = resp.json().get("status", "")
        return {
            "IN_QUEUE": GenerationStatus.PENDING,
            "IN_PROGRESS": GenerationStatus.PROCESSING,
            "COMPLETED": GenerationStatus.COMPLETED,
            "FAILED": GenerationStatus.FAILED,
        }.get(status, GenerationStatus.PENDING)


class FalImageGenerator:
    """Image generation via fal.ai (Flux, SDXL, etc.)."""

    provider_name = "fal"
    supports_lora = True
    supports_ip_adapter = True

    API_BASE = "https://fal.run"

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.api_key = config.api_key
        self.model = config.default_model or "fal-ai/flux/dev"
        self._client: httpx.AsyncClient | None = None

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
