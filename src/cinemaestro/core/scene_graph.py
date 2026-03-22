"""The SceneGraph: central intermediate representation of a short film.

This is the single most important data structure in Cinemaestro. The Story Engine
produces it, and every downstream stage consumes it. It is designed to be:
- Serializable to/from YAML for human editing
- Validated by Pydantic for machine processing
- Rich enough to capture cinematic intent (camera, lighting, mood)
- Flat enough to iterate over shots without deep nesting
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class CameraMovement(str, Enum):
    STATIC = "static"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    TRACKING = "tracking"
    CRANE_UP = "crane_up"
    CRANE_DOWN = "crane_down"
    HANDHELD = "handheld"
    STEADICAM = "steadicam"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    ORBIT = "orbit"


class ShotType(str, Enum):
    EXTREME_WIDE = "extreme_wide"
    WIDE = "wide"
    MEDIUM_WIDE = "medium_wide"
    MEDIUM = "medium"
    MEDIUM_CLOSE = "medium_close"
    CLOSE_UP = "close_up"
    EXTREME_CLOSE_UP = "extreme_close_up"
    OVER_THE_SHOULDER = "over_the_shoulder"
    POV = "pov"
    INSERT = "insert"
    TWO_SHOT = "two_shot"
    GROUP = "group"
    AERIAL = "aerial"


class TransitionType(str, Enum):
    CUT = "cut"
    DISSOLVE = "dissolve"
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    FADE_TO_BLACK = "fade_to_black"
    WIPE = "wipe"
    SMASH_CUT = "smash_cut"
    MATCH_CUT = "match_cut"
    J_CUT = "j_cut"
    L_CUT = "l_cut"


class DialogueLine(BaseModel):
    """A single line of dialogue spoken by a character within a shot."""

    character_id: str
    text: str
    emotion: str = "neutral"
    direction: str = ""  # parenthetical acting direction, e.g. "(whispering)"
    overlap_previous: bool = False  # starts while previous line still playing


class SoundEffect(BaseModel):
    """A sound effect to be generated or sourced for a shot."""

    description: str  # e.g. "door creaking open", "rain on window"
    timestamp_hint: float = 0.0  # relative to shot start, in seconds
    duration_hint: float | None = None
    intensity: float = 0.7  # 0.0 to 1.0
    spatial: str = ""  # "left", "right", "center", "ambient"

    @field_validator("intensity", mode="before")
    @classmethod
    def _coerce_intensity(cls, v: object) -> float:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            mapping = {
                "very_low": 0.2, "very low": 0.2,
                "low": 0.3,
                "medium_low": 0.4, "medium low": 0.4,
                "medium": 0.5,
                "medium_high": 0.6, "medium high": 0.6,
                "high": 0.8,
                "very_high": 0.9, "very high": 0.9,
            }
            key = str(v).strip().lower()
            if key in mapping:
                return mapping[key]
            try:
                return float(v)
            except ValueError:
                return 0.5
        return 0.5


class Shot(BaseModel):
    """A single camera shot — the atomic unit of visual generation."""

    shot_id: str
    shot_type: ShotType
    camera_movement: CameraMovement = CameraMovement.STATIC
    duration_seconds: float = 4.0
    visual_description: str  # what the camera sees — the generation prompt core
    characters_present: list[str] = Field(default_factory=list)
    dialogue: list[DialogueLine] = Field(default_factory=list)
    sound_effects: list[SoundEffect] = Field(default_factory=list)
    mood: str = ""
    lighting: str = ""  # "low key", "golden hour", "neon noir", "overcast"
    color_palette: str = ""  # "warm earth tones", "cold blues", "desaturated"
    transition_in: TransitionType = TransitionType.CUT
    transition_out: TransitionType = TransitionType.CUT
    location_id: str = ""
    time_of_day: str = ""
    weather: str = ""
    generation_hints: dict[str, str] = Field(default_factory=dict)
    # Provider-specific overrides, e.g. {"provider": "kling", "motion_amount": "high"}

    @property
    def has_dialogue(self) -> bool:
        return len(self.dialogue) > 0

    @property
    def total_dialogue_duration_estimate(self) -> float:
        """Rough estimate: ~150 words per minute, ~5 chars per word."""
        total_chars = sum(len(line.text) for line in self.dialogue)
        return total_chars / (150 * 5 / 60)  # chars / (chars_per_second)


class Scene(BaseModel):
    """A continuous sequence at a single location — contains multiple shots."""

    scene_id: str = ""
    scene_number: int = 0
    title: str = ""
    description: str = ""  # prose description of the scene's purpose and content
    location_id: str = ""
    time_of_day: str = "day"
    mood: str = ""
    music_direction: str = ""  # "slow piano, melancholic", "tense strings building"
    shots: list[Shot] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _fill_missing(cls, data: dict) -> dict:  # type: ignore[override]
        if not isinstance(data, dict):
            return data
        # Derive scene_id from scene_number if missing
        if not data.get("scene_id") and data.get("scene_number"):
            data["scene_id"] = f"scene_{data['scene_number']}"
        # Derive scene_number from scene_id if missing
        if not data.get("scene_number") and data.get("scene_id"):
            # Try to extract a number from the scene_id
            import re
            m = re.search(r"(\d+)", str(data["scene_id"]))
            if m:
                data["scene_number"] = int(m.group(1))
        return data

    @property
    def duration_seconds(self) -> float:
        return sum(shot.duration_seconds for shot in self.shots)

    @property
    def characters_present(self) -> set[str]:
        chars: set[str] = set()
        for shot in self.shots:
            chars.update(shot.characters_present)
        return chars


class Location(BaseModel):
    """A physical location in the film's world."""

    location_id: str
    name: str
    description: str
    style_notes: str = ""  # "art deco interior", "brutalist concrete"
    reference_images: list[str] = Field(default_factory=list)


class CharacterAppearance(BaseModel):
    """Character-specific appearance notes for the SceneGraph.

    The full CharacterIdentity lives in the registry; this is the subset
    needed by the story engine and visual generators.
    """

    character_id: str
    name: str
    physical_description: str
    wardrobe: str = ""  # what they wear in this film
    distinguishing_features: str = ""


class Act(BaseModel):
    """A major section of the film — maps to traditional act structure."""

    act_number: int = 0
    title: str = ""
    scenes: list[Scene] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _fill_missing(cls, data: dict) -> dict:  # type: ignore[override]
        if not isinstance(data, dict):
            return data
        # Derive act_number from act_id if missing
        if not data.get("act_number") and data.get("act_id"):
            import re
            m = re.search(r"(\d+)", str(data["act_id"]))
            if m:
                data["act_number"] = int(m.group(1))
        return data

    @property
    def duration_seconds(self) -> float:
        return sum(scene.duration_seconds for scene in self.scenes)


class SceneGraph(BaseModel):
    """The complete intermediate representation of a short film.

    This is the 'screenplay as data' — produced by the Story Engine and consumed
    by every downstream pipeline stage. It can be serialized to YAML for human
    editing and loaded back without loss.
    """

    title: str
    logline: str = ""
    genre: str = ""
    tone: str = ""  # "gritty noir", "whimsical comedy", "psychological thriller"
    style: str = ""  # "photorealistic", "anime", "watercolor", "claymation"
    target_duration_seconds: float = 120.0
    aspect_ratio: str = "16:9"
    characters: list[CharacterAppearance] = Field(default_factory=list)
    locations: list[Location] = Field(default_factory=list)
    acts: list[Act] = Field(default_factory=list)
    music_theme: str = ""  # overall musical direction
    color_grade: str = ""  # "teal and orange", "bleach bypass", "vibrant"

    def all_shots(self) -> list[Shot]:
        """Flatten all shots across all acts and scenes, in order."""
        shots = []
        for act in self.acts:
            for scene in act.scenes:
                shots.extend(scene.shots)
        return shots

    def all_scenes(self) -> list[Scene]:
        """Flatten all scenes across all acts, in order."""
        scenes = []
        for act in self.acts:
            scenes.extend(act.scenes)
        return scenes

    def shots_for_character(self, character_id: str) -> list[Shot]:
        """All shots featuring a specific character."""
        return [
            shot
            for shot in self.all_shots()
            if character_id in shot.characters_present
        ]

    def character_ids(self) -> set[str]:
        """All unique character IDs referenced anywhere in the graph."""
        ids: set[str] = set()
        for char in self.characters:
            ids.add(char.character_id)
        for shot in self.all_shots():
            ids.update(shot.characters_present)
            for line in shot.dialogue:
                ids.add(line.character_id)
        return ids

    def get_location(self, location_id: str) -> Location | None:
        for loc in self.locations:
            if loc.location_id == location_id:
                return loc
        return None

    def get_character(self, character_id: str) -> CharacterAppearance | None:
        for char in self.characters:
            if char.character_id == character_id:
                return char
        return None

    @property
    def total_duration_seconds(self) -> float:
        return sum(act.duration_seconds for act in self.acts)

    @property
    def total_shots(self) -> int:
        return len(self.all_shots())

    @property
    def total_dialogue_lines(self) -> int:
        return sum(len(shot.dialogue) for shot in self.all_shots())
