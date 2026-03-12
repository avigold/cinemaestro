"""Local XTTS-v2 TTS provider — free, runs on local GPU."""

from __future__ import annotations

import logging
from pathlib import Path

from cinemaestro.audio.base import AudioResult
from cinemaestro.config import ProviderConfig

logger = logging.getLogger(__name__)


class XTTSTTS:
    """Text-to-speech using Coqui XTTS-v2 locally."""

    provider_name = "xtts"
    supports_voice_cloning = True
    supports_emotion_control = False

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._model = None

    def _load_model(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            try:
                from TTS.api import TTS

                self._model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
                if hasattr(self._model, "to"):
                    self._model.to("cuda")
            except ImportError:
                raise ImportError(
                    "XTTS requires the TTS package. Install with: pip install TTS"
                )
        return self._model

    async def synthesize(
        self,
        text: str,
        voice_id: str | None = None,
        voice_reference: Path | None = None,
        emotion: str = "neutral",
        speed: float = 1.0,
        output_path: Path | None = None,
    ) -> AudioResult:
        import asyncio

        model = self._load_model()
        out_path = output_path or Path(f"/tmp/xtts_output.wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        speaker_wav = str(voice_reference) if voice_reference else None

        # Run TTS in executor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: model.tts_to_file(
                text=text,
                file_path=str(out_path),
                speaker_wav=speaker_wav,
                language="en",
                speed=speed,
            ),
        )

        return AudioResult(
            provider=self.provider_name,
            output_path=out_path,
        )

    async def clone_voice(
        self,
        reference_audio: Path,
        voice_name: str,
    ) -> str:
        # XTTS uses speaker_wav directly — no separate cloning step
        return str(reference_audio)
