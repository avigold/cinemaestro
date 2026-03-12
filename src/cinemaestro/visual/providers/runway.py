"""Runway Gen-4 / Gen-4.5 video generation provider."""

from __future__ import annotations

import asyncio
import base64
import uuid
from pathlib import Path

import httpx

from cinemaestro.config import ProviderConfig
from cinemaestro.visual.base import GenerationResult, GenerationStatus


class RunwayVideoGenerator:
    """Adapter for Runway's Gen-4 / Gen-4.5 API."""

    provider_name = "runway"
    supports_character_reference = True
    supports_image_to_video = True
    max_duration_seconds = 10.0

    API_BASE = "https://api.dev.runwayml.com/v1"

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.api_key = config.api_key
        self.model = config.default_model or "gen4_turbo"
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.API_BASE,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "X-Runway-Version": "2024-11-06",
                },
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
            "model": self.model,
            "promptText": prompt,
            "ratio": aspect_ratio,
            "duration": min(int(duration_seconds), 10),
        }

        if first_frame:
            image_data = base64.b64encode(first_frame.read_bytes()).decode()
            mime = "image/png" if first_frame.suffix == ".png" else "image/jpeg"
            payload["promptImage"] = f"data:{mime};base64,{image_data}"

        if seed is not None:
            payload["seed"] = seed

        resp = await self.client.post("/image_to_video", json=payload)
        resp.raise_for_status()
        task_id = resp.json()["id"]

        # Poll for completion
        result = await self._poll_task(task_id)

        if result.status == GenerationStatus.COMPLETED and output_dir:
            output_path = output_dir / f"{task_id}.mp4"
            await self._download(result.metadata.get("output_url", ""), output_path)
            result.output_path = output_path

        return result

    async def _poll_task(
        self, task_id: str, poll_interval: float = 5.0
    ) -> GenerationResult:
        while True:
            resp = await self.client.get(f"/tasks/{task_id}")
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "")

            if status == "SUCCEEDED":
                output_url = ""
                output_list = data.get("output", [])
                if output_list:
                    output_url = output_list[0]
                return GenerationResult(
                    provider=self.provider_name,
                    generation_id=task_id,
                    status=GenerationStatus.COMPLETED,
                    metadata={"output_url": output_url},
                )
            elif status == "FAILED":
                return GenerationResult(
                    provider=self.provider_name,
                    generation_id=task_id,
                    status=GenerationStatus.FAILED,
                    error_message=data.get("failure", "Unknown error"),
                )

            await asyncio.sleep(poll_interval)

    async def _download(self, url: str, dest: Path) -> None:
        if not url:
            return
        async with httpx.AsyncClient() as dl_client:
            resp = await dl_client.get(url)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)

    async def check_status(self, generation_id: str) -> GenerationStatus:
        resp = await self.client.get(f"/tasks/{generation_id}")
        resp.raise_for_status()
        status = resp.json().get("status", "")
        return {
            "PENDING": GenerationStatus.PENDING,
            "RUNNING": GenerationStatus.PROCESSING,
            "SUCCEEDED": GenerationStatus.COMPLETED,
            "FAILED": GenerationStatus.FAILED,
        }.get(status, GenerationStatus.PENDING)
