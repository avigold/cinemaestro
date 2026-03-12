"""Character identity system for persistent virtual actors.

Characters are first-class entities with visual embeddings, voice profiles, and
behavioral metadata. They persist across scenes and across films via the
CharacterRegistry.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class CharacterIdentity(BaseModel):
    """Persistent identity of a virtual actor across all productions."""

    character_id: str
    display_name: str
    physical_description: str
    age_range: str = ""
    gender: str = ""
    ethnicity: str = ""

    # Visual identity artifacts (paths relative to character dir)
    reference_images: list[str] = Field(default_factory=list)
    face_embedding_path: str | None = None  # .npy file
    lora_path: str | None = None  # directory with LoRA weights
    lora_trigger_word: str = ""
    ip_adapter_image: str | None = None  # primary reference for IP-Adapter

    # Voice identity
    voice_reference_audio: str | None = None  # 6-30s reference clip
    elevenlabs_voice_id: str | None = None
    xtts_speaker_wav: str | None = None
    voice_description: str = ""  # "warm alto, slight rasp"

    # Behavioral metadata (used by Story Engine)
    personality_notes: str = ""
    speech_patterns: str = ""  # "formal, avoids contractions"
    backstory: str = ""

    def identity_file(self, base_dir: Path) -> Path:
        return base_dir / self.character_id / "identity.yaml"

    def resolve_path(self, base_dir: Path, relative: str) -> Path:
        return base_dir / self.character_id / relative

    def resolved_reference_images(self, base_dir: Path) -> list[Path]:
        return [self.resolve_path(base_dir, p) for p in self.reference_images]


class CharacterRegistry:
    """Persistent store of CharacterIdentity objects.

    Each character gets a directory under the registry root containing its
    identity.yaml and all artifact files (reference images, embeddings, etc.).
    """

    def __init__(self, registry_dir: Path) -> None:
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def get(self, character_id: str) -> CharacterIdentity | None:
        identity_file = self.registry_dir / character_id / "identity.yaml"
        if not identity_file.exists():
            return None
        data = yaml.safe_load(identity_file.read_text())
        return CharacterIdentity(**data)

    def register(self, identity: CharacterIdentity) -> None:
        char_dir = self.registry_dir / identity.character_id
        char_dir.mkdir(parents=True, exist_ok=True)
        identity_file = char_dir / "identity.yaml"
        identity_file.write_text(
            yaml.dump(identity.model_dump(), default_flow_style=False, sort_keys=False)
        )

    def update(self, identity: CharacterIdentity) -> None:
        self.register(identity)

    def delete(self, character_id: str) -> bool:
        char_dir = self.registry_dir / character_id
        if char_dir.exists():
            shutil.rmtree(char_dir)
            return True
        return False

    def list_all(self) -> list[CharacterIdentity]:
        characters = []
        for char_dir in sorted(self.registry_dir.iterdir()):
            if char_dir.is_dir():
                identity = self.get(char_dir.name)
                if identity is not None:
                    characters.append(identity)
        return characters

    def exists(self, character_id: str) -> bool:
        return (self.registry_dir / character_id / "identity.yaml").exists()

    def character_dir(self, character_id: str) -> Path:
        return self.registry_dir / character_id

    def import_reference_images(
        self, character_id: str, image_paths: list[Path]
    ) -> list[str]:
        """Copy reference images into the character's directory.

        Returns the relative paths stored in the identity.
        """
        char_dir = self.registry_dir / character_id
        char_dir.mkdir(parents=True, exist_ok=True)

        relative_paths = []
        for i, src in enumerate(image_paths):
            suffix = src.suffix
            dest_name = f"reference_{i:02d}{suffix}"
            dest = char_dir / dest_name
            shutil.copy2(src, dest)
            relative_paths.append(dest_name)

        return relative_paths
