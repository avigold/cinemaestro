"""Constructs visual generation prompts from SceneGraph shots.

Translates the structured Shot data (camera, lighting, mood, characters)
into natural-language prompts optimized for each provider's model.
"""

from __future__ import annotations

from cinemaestro.core.scene_graph import (
    CameraMovement,
    CharacterAppearance,
    Location,
    SceneGraph,
    Shot,
    ShotType,
)


SHOT_TYPE_DESCRIPTIONS: dict[ShotType, str] = {
    ShotType.EXTREME_WIDE: "extreme wide shot, tiny figures in vast landscape",
    ShotType.WIDE: "wide shot, full environment visible",
    ShotType.MEDIUM_WIDE: "medium wide shot, characters from knees up",
    ShotType.MEDIUM: "medium shot, characters from waist up",
    ShotType.MEDIUM_CLOSE: "medium close-up, characters from chest up",
    ShotType.CLOSE_UP: "close-up shot, face filling frame",
    ShotType.EXTREME_CLOSE_UP: "extreme close-up, detail shot",
    ShotType.OVER_THE_SHOULDER: "over-the-shoulder shot",
    ShotType.POV: "point-of-view shot, first person perspective",
    ShotType.INSERT: "insert shot, close detail of object",
    ShotType.TWO_SHOT: "two shot, two characters in frame",
    ShotType.GROUP: "group shot, multiple characters",
    ShotType.AERIAL: "aerial shot, bird's eye view",
}

CAMERA_MOVEMENT_DESCRIPTIONS: dict[CameraMovement, str] = {
    CameraMovement.STATIC: "",
    CameraMovement.PAN_LEFT: "camera panning left",
    CameraMovement.PAN_RIGHT: "camera panning right",
    CameraMovement.TILT_UP: "camera tilting upward",
    CameraMovement.TILT_DOWN: "camera tilting downward",
    CameraMovement.DOLLY_IN: "camera dolly moving forward",
    CameraMovement.DOLLY_OUT: "camera dolly pulling back",
    CameraMovement.TRACKING: "camera tracking alongside subject",
    CameraMovement.CRANE_UP: "crane shot rising upward",
    CameraMovement.CRANE_DOWN: "crane shot descending",
    CameraMovement.HANDHELD: "handheld camera, slight movement",
    CameraMovement.STEADICAM: "smooth steadicam movement",
    CameraMovement.ZOOM_IN: "slow zoom in",
    CameraMovement.ZOOM_OUT: "slow zoom out",
    CameraMovement.ORBIT: "camera orbiting around subject",
}


class PromptBuilder:
    """Builds generation prompts from structured shot data."""

    def __init__(self, scene_graph: SceneGraph) -> None:
        self.scene_graph = scene_graph

    def build_shot_prompt(self, shot: Shot) -> str:
        """Build a complete visual prompt for a shot."""
        parts: list[str] = []

        # Global style
        if self.scene_graph.style:
            parts.append(f"{self.scene_graph.style} style")

        # Shot type
        shot_desc = SHOT_TYPE_DESCRIPTIONS.get(shot.shot_type, "")
        if shot_desc:
            parts.append(shot_desc)

        # Camera movement
        camera_desc = CAMERA_MOVEMENT_DESCRIPTIONS.get(shot.camera_movement, "")
        if camera_desc:
            parts.append(camera_desc)

        # Main visual description
        parts.append(shot.visual_description)

        # Character descriptions (inline for providers that need text-only)
        for char_id in shot.characters_present:
            char = self.scene_graph.get_character(char_id)
            if char and char.physical_description:
                parts.append(char.physical_description)
                if char.wardrobe:
                    parts.append(f"wearing {char.wardrobe}")

        # Location context
        if shot.location_id:
            location = self.scene_graph.get_location(shot.location_id)
            if location:
                parts.append(f"setting: {location.description}")

        # Lighting and mood
        if shot.lighting:
            parts.append(f"{shot.lighting} lighting")
        if shot.mood:
            parts.append(f"{shot.mood} mood")
        if shot.color_palette:
            parts.append(f"{shot.color_palette} color palette")
        if shot.weather:
            parts.append(shot.weather)
        if shot.time_of_day:
            parts.append(shot.time_of_day)

        # Film-level color grade
        if self.scene_graph.color_grade:
            parts.append(f"{self.scene_graph.color_grade} color grading")

        # Cinematic quality tags
        parts.append("cinematic, high quality, film grain, 35mm")

        return ", ".join(p for p in parts if p)

    def build_first_frame_prompt(self, shot: Shot) -> str:
        """Build a prompt for generating a still first frame (for first-frame strategy).

        This is similar to the shot prompt but optimized for still image generation
        and includes LoRA trigger words.
        """
        parts: list[str] = []

        if self.scene_graph.style:
            parts.append(f"{self.scene_graph.style} style")

        shot_desc = SHOT_TYPE_DESCRIPTIONS.get(shot.shot_type, "")
        if shot_desc:
            parts.append(shot_desc)

        parts.append(shot.visual_description)

        # For first-frame strategy, character descriptions are critical
        for char_id in shot.characters_present:
            char = self.scene_graph.get_character(char_id)
            if char:
                parts.append(char.physical_description)
                if char.wardrobe:
                    parts.append(f"wearing {char.wardrobe}")

        if shot.location_id:
            location = self.scene_graph.get_location(shot.location_id)
            if location:
                parts.append(f"in {location.description}")

        if shot.lighting:
            parts.append(f"{shot.lighting} lighting")
        if shot.mood:
            parts.append(f"{shot.mood} atmosphere")

        parts.append("photorealistic, sharp focus, cinematic composition, 35mm film")

        return ", ".join(p for p in parts if p)
