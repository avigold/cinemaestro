"""Kling video generation provider with Elements support for character consistency."""

from __future__ import annotations

import asyncio
import base64
import uuid
from pathlib import Path

import httpx

from cinemaestro.config import ProviderConfig
from cinemaestro.visual.base import GenerationResult, GenerationStatus


class KlingVideoGenerator:
    """Adapter for Kling's video generation API.

    Kling's "Elements" feature allows uploading character reference images and
    referencing them as @Element1, @Element2 etc. in prompts, enabling native
    character consistency.
    """

    provider_name = "kling"
    supports_character_reference = True
    supports_image_to_video = True
    max_duration_seconds = 10.0

    API_BASE = "https://api.klingai.com/v1"

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.api_key = config.api_key
        self.model = config.default_model or "kling-v2"
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
        payload: dict = {
            "model_name": self.model,
            "prompt": prompt,
            "cfg_scale": 0.5,
            "mode": "std",
            "aspect_ratio": aspect_ratio,
            "duration": str(min(int(duration_seconds), 10)),
        }

        if first_frame:
            image_data = base64.b64encode(first_frame.read_bytes()).decode()
            payload["image"] = image_data
            endpoint = "/videos/image2video"
        else:
            endpoint = "/videos/text2video"

        # Character elements for native consistency
        if reference_images:
            elements = []
            for i, img_path in enumerate(reference_images[:4]):  # max 4 elements
                img_data = base64.b64encode(img_path.read_bytes()).decode()
                elements.append({
                    "image": img_data,
                    "element_id": f"element_{i+1}",
                })
            if elements:
                payload["elements"] = elements

        resp = await self.client.post(endpoint, json=payload)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        task_id = data.get("task_id", str(uuid.uuid4()))

        result = await self._poll_task(task_id, endpoint)

        if result.status == GenerationStatus.COMPLETED and output_dir:
            output_path = output_dir / f"{task_id}.mp4"
            output_url = result.metadata.get("output_url", "")
            if output_url:
                await self._download(output_url, output_path)
                result.output_path = output_path

        return result

    async def _poll_task(
        self, task_id: str, endpoint: str, poll_interval: float = 5.0
    ) -> GenerationResult:
        status_url = f"{endpoint}/{task_id}"
        while True:
            resp = await self.client.get(status_url)
            resp.raise_for_status()
            data = resp.json().get("data", {})
            status = data.get("task_status", "")

            if status == "succeed":
                videos = data.get("task_result", {}).get("videos", [])
                output_url = videos[0]["url"] if videos else ""
                return GenerationResult(
                    provider=self.provider_name,
                    generation_id=task_id,
                    status=GenerationStatus.COMPLETED,
                    metadata={"output_url": output_url},
                )
            elif status == "failed":
                return GenerationResult(
                    provider=self.provider_name,
                    generation_id=task_id,
                    status=GenerationStatus.FAILED,
                    error_message=data.get("task_status_msg", "Unknown error"),
                )

            await asyncio.sleep(poll_interval)

    async def _download(self, url: str, dest: Path) -> None:
        async with httpx.AsyncClient() as dl_client:
            resp = await dl_client.get(url)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)

    async def check_status(self, generation_id: str) -> GenerationStatus:
        resp = await self.client.get(f"/videos/text2video/{generation_id}")
        resp.raise_for_status()
        status = resp.json().get("data", {}).get("task_status", "")
        return {
            "submitted": GenerationStatus.PENDING,
            "processing": GenerationStatus.PROCESSING,
            "succeed": GenerationStatus.COMPLETED,
            "failed": GenerationStatus.FAILED,
        }.get(status, GenerationStatus.PENDING)
