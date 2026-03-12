"""Configuration system using Pydantic Settings.

Supports layered configuration:
  env vars > .env file > project TOML > user TOML > defaults

API keys set at the root level (via env vars or .env) are automatically
propagated into the appropriate provider configs, so users only need to
set keys once.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings


class ProviderConfig(BaseModel):
    """Configuration for a single AI provider."""

    enabled: bool = False
    api_key: str = ""
    base_url: str = ""
    default_model: str = ""
    max_concurrent: int = 3
    timeout_seconds: int = 300
    cost_per_second: float = 0.0


class StoryConfig(BaseModel):
    """Story engine configuration."""

    llm_provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.8
    max_retries: int = 3
    max_tokens: int = 8192


class VisualConfig(BaseModel):
    """Visual generation configuration."""

    default_provider: str = "runway"
    fallback_provider: str = "comfyui"
    default_aspect_ratio: str = "16:9"
    default_resolution: tuple[int, int] = (1920, 1080)
    character_consistency_strategy: str = "first_frame"
    # "first_frame" = generate still with LoRA, then animate
    # "native" = use provider's built-in character reference
    # "reference_only" = pass reference images to provider
    default_shot_duration: float = 4.0
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)


class TTSConfig(BaseModel):
    """Text-to-speech configuration."""

    provider: str = "elevenlabs"
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)


class MusicConfig(BaseModel):
    """Music generation configuration."""

    provider: str = "stable_audio"
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)


class SFXConfig(BaseModel):
    """Sound effects configuration."""

    provider: str = "stable_audio"
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)


class AudioConfig(BaseModel):
    """Audio pipeline configuration."""

    tts: TTSConfig = Field(default_factory=TTSConfig)
    music: MusicConfig = Field(default_factory=MusicConfig)
    sfx: SFXConfig = Field(default_factory=SFXConfig)
    master_volume: float = 1.0
    dialogue_volume: float = 1.0
    music_volume: float = 0.3
    sfx_volume: float = 0.6
    sample_rate: int = 44100


class ConsistencyConfig(BaseModel):
    """Face consistency verification and correction settings."""

    enabled: bool = True
    face_similarity_threshold: float = 0.55
    sample_interval_seconds: float = 0.5
    auto_repair: bool = True
    face_enhance_after_swap: bool = True
    max_repair_attempts: int = 3


class AssemblyConfig(BaseModel):
    """Video assembly configuration."""

    fps: float = 24.0
    default_transition: str = "cut"
    default_transition_duration: float = 0.5
    burn_subtitles: bool = True
    subtitle_style: str = "modern"  # "modern", "classic", "minimal"


class ExportConfig(BaseModel):
    """Export and rendering settings."""

    preset: str = "youtube_1080p"
    codec: str = "libx264"
    audio_codec: str = "aac"
    crf: int = 18
    pixel_format: str = "yuv420p"


# Mapping from root-level API key field -> (subsystem, provider name, default model)
_KEY_TO_PROVIDER: dict[str, list[tuple[str, str, str]]] = {
    "runway_api_key": [("visual", "runway", "gen4_turbo")],
    "kling_api_key": [("visual", "kling", "kling-v2")],
    "fal_key": [
        ("visual", "fal", "fal-ai/flux/dev"),
    ],
    "replicate_api_token": [("visual", "replicate", "black-forest-labs/flux-dev")],
    "elevenlabs_api_key": [("audio.tts", "elevenlabs", "")],
    "stability_api_key": [
        ("audio.music", "stable_audio", ""),
        ("audio.sfx", "stable_audio", ""),
    ],
}


class CinemaestroConfig(BaseSettings):
    """Root configuration for the entire Cinemaestro system.

    API keys can be set via:
    - Environment variables (ANTHROPIC_API_KEY, RUNWAY_API_KEY, etc.)
    - A .env file in the project directory or working directory
    - The user config at ~/.cinemaestro/config.toml
    - A project-level cinemaestro.toml

    Keys are automatically propagated to the correct provider configs.
    """

    model_config = {
        "env_prefix": "CINEMAESTRO_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    project_dir: Path = Path("./project")
    story: StoryConfig = Field(default_factory=StoryConfig)
    visual: VisualConfig = Field(default_factory=VisualConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    consistency: ConsistencyConfig = Field(default_factory=ConsistencyConfig)
    assembly: AssemblyConfig = Field(default_factory=AssemblyConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    character_registry_dir: Path = Path("~/.cinemaestro/characters")
    comfyui_url: str = "http://127.0.0.1:8188"

    log_level: str = "INFO"
    max_budget_usd: float = 50.0

    # API keys — can be set via env vars (no prefix needed for standard names)
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    runway_api_key: str = ""
    kling_api_key: str = ""
    fal_key: str = ""
    replicate_api_token: str = ""
    elevenlabs_api_key: str = ""
    stability_api_key: str = ""

    @model_validator(mode="before")
    @classmethod
    def _load_env_and_toml(cls, data: Any) -> Any:
        """Load API keys from .env files, env vars, and TOML configs."""
        if not isinstance(data, dict):
            return data

        # Mapping from config field name -> standard env var name
        env_keys = {
            "anthropic_api_key": "ANTHROPIC_API_KEY",
            "openai_api_key": "OPENAI_API_KEY",
            "runway_api_key": "RUNWAY_API_KEY",
            "kling_api_key": "KLING_API_KEY",
            "fal_key": "FAL_KEY",
            "replicate_api_token": "REPLICATE_API_TOKEN",
            "elevenlabs_api_key": "ELEVENLABS_API_KEY",
            "stability_api_key": "STABILITY_API_KEY",
        }

        # Load .env files manually so standard env var names work
        # (pydantic-settings applies env_prefix which won't match)
        dotenv_values: dict[str, str] = {}
        env_file_paths = []

        # Check project dir .env
        project_dir = data.get("project_dir")
        if project_dir:
            p = Path(project_dir) if isinstance(project_dir, str) else project_dir
            if (p / ".env").exists():
                env_file_paths.append(p / ".env")

        # Check cwd .env
        if Path(".env").exists():
            env_file_paths.append(Path(".env"))

        # Check user-level .env
        user_env = Path("~/.cinemaestro/.env").expanduser()
        if user_env.exists():
            env_file_paths.append(user_env)

        for env_path in env_file_paths:
            dotenv_values.update(_parse_dotenv(env_path))

        # Resolve keys: explicit data > os.environ > .env file
        for field_name, env_name in env_keys.items():
            if data.get(field_name):
                continue
            val = os.environ.get(env_name, "") or dotenv_values.get(env_name, "")
            if val:
                data[field_name] = val

        # Load TOML config files (project-level overrides user-level)
        toml_data: dict[str, Any] = {}

        # User-level config
        user_config = Path("~/.cinemaestro/config.toml").expanduser()
        if user_config.exists():
            with open(user_config, "rb") as f:
                toml_data = tomllib.load(f)

        # CWD-level config
        cwd_config = Path("cinemaestro.toml")
        if cwd_config.exists():
            with open(cwd_config, "rb") as f:
                _deep_merge(toml_data, tomllib.load(f))

        # Project-level config (overrides cwd and user-level)
        project_dir = data.get("project_dir", Path("./project"))
        if isinstance(project_dir, str):
            project_dir = Path(project_dir)
        project_config = project_dir / "cinemaestro.toml"
        if project_config.exists() and project_config.resolve() != cwd_config.resolve():
            with open(project_config, "rb") as f:
                _deep_merge(toml_data, tomllib.load(f))

        # Merge TOML into data (env vars / explicit values take precedence)
        if toml_data:
            _deep_merge_defaults(data, toml_data)

        return data

    @model_validator(mode="after")
    def _propagate_api_keys(self) -> CinemaestroConfig:
        """Propagate root-level API keys into provider configs.

        If a user sets RUNWAY_API_KEY in their env, this ensures:
        - visual.providers["runway"] exists and is enabled
        - Its api_key is populated
        - A sensible default_model is set if none was specified
        """
        for key_field, targets in _KEY_TO_PROVIDER.items():
            api_key = getattr(self, key_field, "")
            if not api_key:
                continue

            for subsystem_path, provider_name, default_model in targets:
                providers_dict = self._get_providers_dict(subsystem_path)
                if providers_dict is None:
                    continue

                if provider_name not in providers_dict:
                    providers_dict[provider_name] = ProviderConfig()

                pc = providers_dict[provider_name]
                if not pc.api_key:
                    pc.api_key = api_key
                pc.enabled = True
                if not pc.default_model and default_model:
                    pc.default_model = default_model

        # ComfyUI — always available if URL is set (no API key needed)
        if self.comfyui_url:
            if "comfyui" not in self.visual.providers:
                self.visual.providers["comfyui"] = ProviderConfig()
            pc = self.visual.providers["comfyui"]
            pc.enabled = True
            pc.base_url = self.comfyui_url

        return self

    def _get_providers_dict(self, path: str) -> dict[str, ProviderConfig] | None:
        """Navigate to a nested providers dict by dotted path."""
        parts = path.split(".")
        obj: Any = self
        for part in parts:
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        if hasattr(obj, "providers"):
            return obj.providers
        return None

    def get_available_providers(self) -> dict[str, list[str]]:
        """Return a mapping of subsystem -> list of enabled provider names."""
        result: dict[str, list[str]] = {}

        # Visual
        result["visual"] = [
            name for name, pc in self.visual.providers.items() if pc.enabled
        ]

        # TTS
        result["tts"] = [
            name for name, pc in self.audio.tts.providers.items() if pc.enabled
        ]

        # Music
        result["music"] = [
            name for name, pc in self.audio.music.providers.items() if pc.enabled
        ]

        # SFX
        result["sfx"] = [
            name for name, pc in self.audio.sfx.providers.items() if pc.enabled
        ]

        # Story (LLM — check standard env vars)
        story_providers = []
        if self.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY"):
            story_providers.append("anthropic")
        if self.openai_api_key or os.environ.get("OPENAI_API_KEY"):
            story_providers.append("openai")
        result["story"] = story_providers

        return result

    def get_provider_config(self, subsystem: str, provider_name: str) -> ProviderConfig | None:
        """Get a specific provider's config by subsystem and name."""
        providers = self._get_providers_dict(subsystem)
        if providers and provider_name in providers:
            return providers[provider_name]
        return None

    def summary(self) -> dict[str, Any]:
        """Human-readable config summary (keys masked)."""
        available = self.get_available_providers()

        def mask(key: str) -> str:
            if not key:
                return "(not set)"
            if len(key) <= 8:
                return "****"
            return key[:4] + "..." + key[-4:]

        return {
            "project_dir": str(self.project_dir),
            "story": {
                "provider": self.story.llm_provider,
                "model": self.story.model,
                "api_key": mask(self.anthropic_api_key if self.story.llm_provider == "anthropic" else self.openai_api_key),
            },
            "visual": {
                "default_provider": self.visual.default_provider,
                "fallback_provider": self.visual.fallback_provider,
                "consistency_strategy": self.visual.character_consistency_strategy,
                "available": available.get("visual", []),
            },
            "audio": {
                "tts_provider": self.audio.tts.provider,
                "music_provider": self.audio.music.provider,
                "available_tts": available.get("tts", []),
                "available_music": available.get("music", []),
            },
            "consistency": {
                "enabled": self.consistency.enabled,
                "threshold": self.consistency.face_similarity_threshold,
                "auto_repair": self.consistency.auto_repair,
            },
            "api_keys": {
                "anthropic": mask(self.anthropic_api_key),
                "openai": mask(self.openai_api_key),
                "runway": mask(self.runway_api_key),
                "kling": mask(self.kling_api_key),
                "fal": mask(self.fal_key),
                "replicate": mask(self.replicate_api_token),
                "elevenlabs": mask(self.elevenlabs_api_key),
                "stability": mask(self.stability_api_key),
            },
            "budget_limit": f"${self.max_budget_usd:.2f}",
        }


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Parse a .env file into a dict. Handles comments, blank lines, and quoted values."""
    values: dict[str, str] = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            values[key] = value
    except OSError:
        pass
    return values


def _deep_merge(base: dict, override: dict) -> None:
    """Merge override into base, recursively for nested dicts."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _deep_merge_defaults(target: dict, defaults: dict) -> None:
    """Merge defaults into target only where target has no value."""
    for key, value in defaults.items():
        if key not in target or target[key] is None or target[key] == "":
            target[key] = value
        elif isinstance(target[key], dict) and isinstance(value, dict):
            _deep_merge_defaults(target[key], value)


def load_config(
    project_dir: Path | None = None,
    config_file: Path | None = None,
) -> CinemaestroConfig:
    """Load configuration with full layering.

    Priority (highest to lowest):
    1. Environment variables
    2. .env file (project dir, then cwd)
    3. Explicit config_file parameter
    4. Project-level cinemaestro.toml
    5. User-level ~/.cinemaestro/config.toml
    6. Defaults
    """
    kwargs: dict[str, Any] = {}

    if project_dir:
        kwargs["project_dir"] = project_dir

    # If an explicit config file is given, load it and pass as initial values
    if config_file and config_file.exists():
        with open(config_file, "rb") as f:
            file_data = tomllib.load(f)
        _deep_merge(kwargs, file_data)

    # Look for .env in project dir
    env_files: list[str] = []
    if project_dir and (project_dir / ".env").exists():
        env_files.append(str(project_dir / ".env"))
    if Path(".env").exists():
        env_files.append(".env")

    if env_files:
        kwargs.setdefault("_env_file", env_files[0])

    return CinemaestroConfig(**kwargs)
