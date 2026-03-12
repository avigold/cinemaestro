"""Tests for CharacterIdentity and CharacterRegistry."""

from __future__ import annotations

from pathlib import Path

from cinemaestro.core.character import CharacterIdentity, CharacterRegistry


class TestCharacterRegistry:
    def test_register_and_get(self, tmp_path: Path) -> None:
        registry = CharacterRegistry(tmp_path / "characters")
        identity = CharacterIdentity(
            character_id="test_char",
            display_name="Test Character",
            physical_description="a test character",
        )

        registry.register(identity)
        retrieved = registry.get("test_char")

        assert retrieved is not None
        assert retrieved.character_id == "test_char"
        assert retrieved.display_name == "Test Character"

    def test_get_nonexistent(self, tmp_path: Path) -> None:
        registry = CharacterRegistry(tmp_path / "characters")
        assert registry.get("nonexistent") is None

    def test_list_all(self, tmp_path: Path) -> None:
        registry = CharacterRegistry(tmp_path / "characters")

        for i in range(3):
            identity = CharacterIdentity(
                character_id=f"char_{i}",
                display_name=f"Character {i}",
                physical_description=f"description {i}",
            )
            registry.register(identity)

        all_chars = registry.list_all()
        assert len(all_chars) == 3

    def test_exists(self, tmp_path: Path) -> None:
        registry = CharacterRegistry(tmp_path / "characters")
        identity = CharacterIdentity(
            character_id="existing",
            display_name="Existing",
            physical_description="exists",
        )
        registry.register(identity)

        assert registry.exists("existing")
        assert not registry.exists("nonexistent")

    def test_delete(self, tmp_path: Path) -> None:
        registry = CharacterRegistry(tmp_path / "characters")
        identity = CharacterIdentity(
            character_id="to_delete",
            display_name="Delete Me",
            physical_description="temporary",
        )
        registry.register(identity)
        assert registry.exists("to_delete")

        result = registry.delete("to_delete")
        assert result is True
        assert not registry.exists("to_delete")

        result = registry.delete("nonexistent")
        assert result is False

    def test_update(self, tmp_path: Path) -> None:
        registry = CharacterRegistry(tmp_path / "characters")
        identity = CharacterIdentity(
            character_id="updatable",
            display_name="Original",
            physical_description="original desc",
        )
        registry.register(identity)

        identity.display_name = "Updated"
        registry.update(identity)

        retrieved = registry.get("updatable")
        assert retrieved is not None
        assert retrieved.display_name == "Updated"
