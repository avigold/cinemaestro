"""LLM-backed story writer using Anthropic Claude or OpenAI.

Multi-step process:
1. Generate a treatment/outline from the concept
2. Expand into full scenes with dialogue
3. Generate the detailed shot list with camera directions
4. Validate and parse into SceneGraph
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from cinemaestro.config import StoryConfig
from cinemaestro.core.scene_graph import SceneGraph

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"

SYSTEM_PROMPT = """\
You are a master screenwriter and cinematographer. You create structured \
screenplays with precise visual directions for AI-generated short films.

Your output must be valid JSON matching the SceneGraph schema exactly. \
Every shot must have a unique shot_id following the pattern: \
act{N}_s{M}_shot{K} (e.g., act1_s1_shot1).

Focus on:
- Visual storytelling (show, don't tell)
- Strong character distinctiveness (each character must be visually unique)
- Varied shot types and camera movements for cinematic interest
- Realistic dialogue that reveals character
- Clear scene transitions that maintain narrative flow
- Mood and lighting that reinforce the emotional arc
"""

SCREENPLAY_PROMPT = """\
Create a complete short film screenplay from this concept:

CONCEPT: {concept}

TARGET DURATION: {target_duration_seconds} seconds (approximately \
{target_duration_seconds_div_60:.1f} minutes)
{genre_line}
{tone_line}
{style_line}
{characters_line}

Generate a complete SceneGraph as JSON with this structure:
- title: string
- logline: one-sentence summary
- genre: string
- tone: string
- style: visual style (e.g., "photorealistic", "cinematic noir")
- target_duration_seconds: {target_duration_seconds}
- aspect_ratio: "16:9"
- characters: list of {{ character_id, name, physical_description, wardrobe, \
distinguishing_features }}
- locations: list of {{ location_id, name, description, style_notes }}
- acts: list of acts, each containing scenes, each containing shots
- music_theme: overall musical direction
- color_grade: overall color grading style

Each shot must include:
- shot_id: unique identifier (act1_s1_shot1 format)
- shot_type: one of {shot_types}
- camera_movement: one of {camera_movements}
- duration_seconds: realistic duration (2-8 seconds typical)
- visual_description: detailed description of what the camera sees
- characters_present: list of character_ids in frame
- dialogue: list of {{ character_id, text, emotion, direction }}
- sound_effects: list of {{ description, intensity }}
- mood: emotional tone of the shot
- lighting: lighting description
- location_id: which location
- time_of_day: time context
- transition_in/transition_out: one of {transitions}

Make each character visually distinctive with unique physical features, \
clothing, and mannerisms. This is critical for AI generation consistency.

Respond with ONLY the JSON, no markdown formatting or explanation.
"""


class LLMStoryWriter:
    """Story writer backed by an LLM (Claude or GPT-4o)."""

    def __init__(self, config: StoryConfig) -> None:
        self.config = config
        self.provider = config.llm_provider
        self.model = config.model

    async def write(
        self,
        concept: str,
        target_duration_seconds: float = 120.0,
        genre: str = "",
        tone: str = "",
        style: str = "",
        existing_characters: list[str] | None = None,
    ) -> SceneGraph:
        from cinemaestro.core.scene_graph import (
            CameraMovement,
            ShotType,
            TransitionType,
        )

        shot_types = ", ".join(s.value for s in ShotType)
        camera_movements = ", ".join(c.value for c in CameraMovement)
        transitions = ", ".join(t.value for t in TransitionType)

        prompt = SCREENPLAY_PROMPT.format(
            concept=concept,
            target_duration_seconds=target_duration_seconds,
            target_duration_seconds_div_60=target_duration_seconds / 60,
            genre_line=f"GENRE: {genre}" if genre else "",
            tone_line=f"TONE: {tone}" if tone else "",
            style_line=f"STYLE: {style}" if style else "",
            characters_line=(
                f"EXISTING CHARACTERS (reuse these): {', '.join(existing_characters)}"
                if existing_characters
                else ""
            ),
            shot_types=shot_types,
            camera_movements=camera_movements,
            transitions=transitions,
        )

        raw_json = await self._call_llm(prompt)
        scene_graph = self._parse_response(raw_json)

        logger.info(
            "Generated SceneGraph: %d acts, %d scenes, %d shots, ~%.0fs duration",
            len(scene_graph.acts),
            len(scene_graph.all_scenes()),
            scene_graph.total_shots,
            scene_graph.total_duration_seconds,
        )

        return scene_graph

    async def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM provider."""
        if self.provider == "anthropic":
            return await self._call_anthropic(prompt)
        elif self.provider == "openai":
            return await self._call_openai(prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    async def _call_anthropic(self, prompt: str) -> str:
        import anthropic

        client = anthropic.AsyncAnthropic()
        message = await client.messages.create(
            model=self.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    async def _call_openai(self, prompt: str) -> str:
        import openai

        client = openai.AsyncOpenAI()
        response = await client.chat.completions.create(
            model=self.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    def _parse_response(self, raw: str) -> SceneGraph:
        """Parse LLM response into a validated SceneGraph.

        Handles common LLM quirks like markdown fences around JSON.
        """
        text = raw.strip()
        if text.startswith("```"):
            # Strip markdown code fences
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        data = json.loads(text)
        return SceneGraph(**data)
