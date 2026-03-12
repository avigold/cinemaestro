"""Replicate provider — multi-model gateway."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import httpx

from cinemaestro.config import ProviderConfig
from cinemaestro.visual.base import GenerationResult, GenerationStatus


class ReplicateVideoGenerator:
    """Video generation via Replicate's API."""

    provider_name = "replicate"
    supports_character_reference = False
    supports_image_to_video = True
    max_duration_seconds = 10.0

    API_BASE = "https://api.replicate.com/v1"

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.api_key = config.api_key
        self.model = config.default_model or "wan-ai/wan-2.1-i2v-480p"
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.API_BASE,
                headers={"Authorization": f"Bearer {self.api_key}"},
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
        input_data: dict = {"prompt": prompt}

        if seed is not None:
            input_data["seed"] = seed

        resp = await self.client.post(
            "/predictions",
            json={
                "version": self.model,
                "input": input_data,
            },
        )
        resp.raise_for_status()
        prediction = resp.json()
        pred_id = prediction["id"]

        result = await self._poll(pred_id)

        if result.status == GenerationStatus.COMPLETED and output_dir:
            output_url = result.metadata.get("output_url", "")
            if output_url:
                output_path = output_dir / f"replicate_{pred_id}.mp4"
                await self._download(output_url, output_path)
                result.output_path = output_path

        return result

    async def _poll(self, pred_id: str, poll_interval: float = 5.0) -> GenerationResult:
        while True:
            resp = await self.client.get(f"/predictions/{pred_id}")
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "")

            if status == "succeeded":
                output = data.get("output", "")
                if isinstance(output, list):
                    output = output[0] if output else ""
                return GenerationResult(
                    provider=self.provider_name,
                    generation_id=pred_id,
                    status=GenerationStatus.COMPLETED,
                    metadata={"output_url": output},
                )
            elif status in ("failed", "canceled"):
                return GenerationResult(
                    provider=self.provider_name,
                    generation_id=pred_id,
                    status=GenerationStatus.FAILED,
                    error_message=data.get("error", ""),
                )

            await asyncio.sleep(poll_interval)

    async def _download(self, url: str, dest: Path) -> None:
        async with httpx.AsyncClient() as dl_client:
            resp = await dl_client.get(url)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)

    async def check_status(self, generation_id: str) -> GenerationStatus:
        resp = await self.client.get(f"/predictions/{generation_id}")
        resp.raise_for_status()
        status = resp.json().get("status", "")
        return {
            "starting": GenerationStatus.PENDING,
            "processing": GenerationStatus.PROCESSING,
            "succeeded": GenerationStatus.COMPLETED,
            "failed": GenerationStatus.FAILED,
        }.get(status, GenerationStatus.PENDING)


class ReplicateImageGenerator:
    """Image generation via Replicate's API."""

    provider_name = "replicate"
    supports_lora = True
    supports_ip_adapter = False

    API_BASE = "https://api.replicate.com/v1"

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.api_key = config.api_key
        self.model = config.default_model or "black-forest-labs/flux-dev"
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.API_BASE,
                headers={"Authorization": f"Bearer {self.api_key}"},
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

        input_data: dict = {
            "prompt": prompt,
            "width": width,
            "height": height,
        }
        if negative_prompt:
            input_data["negative_prompt"] = negative_prompt
        if seed is not None:
            input_data["seed"] = seed

        resp = await self.client.post(
            "/predictions",
            json={"version": self.model, "input": input_data},
        )
        resp.raise_for_status()
        pred_id = resp.json()["id"]

        # Reuse video poller
        while True:
            resp = await self.client.get(f"/predictions/{pred_id}")
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "")

            if status == "succeeded":
                output = data.get("output", "")
                if isinstance(output, list):
                    output = output[0] if output else ""
                result = GenerationResult(
                    provider=self.provider_name,
                    generation_id=pred_id,
                    status=GenerationStatus.COMPLETED,
                    metadata={"output_url": output},
                )
                if output_dir and output:
                    output_path = output_dir / f"replicate_{pred_id}.png"
                    await self._download(output, output_path)
                    result.output_path = output_path
                return result
            elif status in ("failed", "canceled"):
                return GenerationResult(
                    provider=self.provider_name,
                    generation_id=pred_id,
                    status=GenerationStatus.FAILED,
                    error_message=data.get("error", ""),
                )

            await asyncio.sleep(5.0)

    async def _download(self, url: str, dest: Path) -> None:
        async with httpx.AsyncClient() as dl_client:
            resp = await dl_client.get(url)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
