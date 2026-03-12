"""Visual generation provider registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cinemaestro.config import ProviderConfig
    from cinemaestro.visual.base import ImageGenerator, VideoGenerator


def get_video_generator(name: str, config: ProviderConfig) -> VideoGenerator:
    """Factory for video generation providers."""
    if name == "runway":
        from cinemaestro.visual.providers.runway import RunwayVideoGenerator
        return RunwayVideoGenerator(config)
    elif name == "kling":
        from cinemaestro.visual.providers.kling import KlingVideoGenerator
        return KlingVideoGenerator(config)
    elif name == "comfyui":
        from cinemaestro.visual.providers.comfyui import ComfyUIVideoGenerator
        return ComfyUIVideoGenerator(config)
    elif name == "fal":
        from cinemaestro.visual.providers.fal import FalVideoGenerator
        return FalVideoGenerator(config)
    elif name == "replicate":
        from cinemaestro.visual.providers.replicate import ReplicateVideoGenerator
        return ReplicateVideoGenerator(config)
    else:
        raise ValueError(f"Unknown video provider: {name}")


def get_image_generator(name: str, config: ProviderConfig) -> ImageGenerator:
    """Factory for image generation providers."""
    if name == "comfyui":
        from cinemaestro.visual.providers.comfyui import ComfyUIImageGenerator
        return ComfyUIImageGenerator(config)
    elif name == "fal":
        from cinemaestro.visual.providers.fal import FalImageGenerator
        return FalImageGenerator(config)
    elif name == "replicate":
        from cinemaestro.visual.providers.replicate import ReplicateImageGenerator
        return ReplicateImageGenerator(config)
    else:
        raise ValueError(f"Unknown image provider: {name}")
