"""Stable Audio music generation via Stability AI API."""

from __future__ import annotations

from pathlib import Path

import httpx

from cinemaestro.audio.base import AudioResult
from cinemaestro.config import ProviderConfig


class StableAudioMusic:
    """Music generation using Stability AI's Stable Audio API."""

    provider_name = "stable_audio"

    API_BASE = "https://api.stability.ai/v2beta"

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.api_key = config.api_key
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.API_BASE,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "audio/*",
                },
                timeout=120,
            )
        return self._client

    async def generate(
        self,
        prompt: str,
        duration_seconds: float = 30.0,
        output_path: Path | None = None,
        **kwargs: str,
    ) -> AudioResult:
        payload = {
            "prompt": prompt,
            "seconds_total": min(duration_seconds, 180),
            "seconds_start": 0,
        }

        resp = await self.client.post("/audio/generate", json=payload)
        resp.raise_for_status()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(resp.content)

        return AudioResult(
            provider=self.provider_name,
            output_path=output_path,
            duration_seconds=duration_seconds,
        )
