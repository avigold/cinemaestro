"""ElevenLabs TTS provider — highest quality voice synthesis."""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

from cinemaestro.audio.base import AudioResult
from cinemaestro.config import ProviderConfig

logger = logging.getLogger(__name__)


class ElevenLabsTTS:
    """Text-to-speech using ElevenLabs API."""

    provider_name = "elevenlabs"
    supports_voice_cloning = True
    supports_emotion_control = True

    API_BASE = "https://api.elevenlabs.io/v1"

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.api_key = config.api_key
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.API_BASE,
                headers={"xi-api-key": self.api_key},
                timeout=60,
            )
        return self._client

    async def synthesize(
        self,
        text: str,
        voice_id: str | None = None,
        voice_reference: Path | None = None,
        emotion: str = "neutral",
        speed: float = 1.0,
        output_path: Path | None = None,
    ) -> AudioResult:
        vid = voice_id or "21m00Tcm4TlvDq8ikWAM"  # default: Rachel

        payload: dict = {
            "text": text,
            "model_id": "eleven_flash_v2_5",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": self._emotion_to_style(emotion),
                "use_speaker_boost": True,
            },
        }

        resp = await self.client.post(
            f"/text-to-speech/{vid}",
            json=payload,
            headers={"Accept": "audio/mpeg"},
        )
        resp.raise_for_status()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(resp.content)

        return AudioResult(
            provider=self.provider_name,
            output_path=output_path,
            duration_seconds=len(resp.content) / (128 * 1000 / 8),  # rough mp3 estimate
        )

    async def clone_voice(
        self,
        reference_audio: Path,
        voice_name: str,
    ) -> str:
        """Create an instant voice clone from a reference audio file."""
        files = {
            "files": (reference_audio.name, reference_audio.read_bytes(), "audio/wav"),
        }
        data = {
            "name": voice_name,
            "description": f"Cloned voice for {voice_name}",
        }

        resp = await self.client.post("/voices/add", data=data, files=files)
        resp.raise_for_status()
        voice_id = resp.json()["voice_id"]
        logger.info("Created voice clone '%s' with ID: %s", voice_name, voice_id)
        return voice_id

    def _emotion_to_style(self, emotion: str) -> float:
        """Map emotion labels to ElevenLabs style parameter (0-1)."""
        emotion_map = {
            "neutral": 0.0,
            "calm": 0.1,
            "sad": 0.2,
            "thoughtful": 0.2,
            "serious": 0.3,
            "concerned": 0.4,
            "surprised": 0.5,
            "happy": 0.6,
            "excited": 0.7,
            "angry": 0.8,
            "passionate": 0.9,
            "dramatic": 1.0,
        }
        return emotion_map.get(emotion, 0.3)
