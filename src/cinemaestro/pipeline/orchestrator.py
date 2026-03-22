"""Pipeline orchestrator — the brain that coordinates story-to-film.

Manages the full pipeline: story generation, character forging, visual
generation, audio synthesis, consistency checking, assembly, and export.
Supports parallel generation, checkpointing, and partial re-runs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from cinemaestro.assembly.timeline_builder import TimelineBuilder
from cinemaestro.assembly.video_editor import SubtitleGenerator, VideoEditor
from cinemaestro.audio.mixer import AudioMixer
from cinemaestro.config import CinemaestroConfig
from cinemaestro.consistency.face_analyzer import FaceAnalyzer
from cinemaestro.consistency.face_swapper import FaceSwapper
from cinemaestro.core.character import CharacterIdentity, CharacterRegistry
from cinemaestro.core.events import EventBus, EventType, PipelineEvent
from cinemaestro.core.project import Project
from cinemaestro.core.scene_graph import SceneGraph, Shot
from cinemaestro.pipeline.state import PipelineState, StageStatus
from cinemaestro.story.llm_writer import LLMStoryWriter
from cinemaestro.visual.prompt_builder import PromptBuilder
from cinemaestro.visual.providers import get_image_generator, get_video_generator
from cinemaestro.visual.strategies.routing import ShotRouter

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Coordinates the full story-to-film pipeline."""

    def __init__(
        self,
        config: CinemaestroConfig,
        project: Project,
        event_bus: EventBus | None = None,
    ) -> None:
        self.config = config
        self.project = project
        self.event_bus = event_bus or EventBus()
        self.state = PipelineState.load(project.pipeline_state_path)
        self.registry = CharacterRegistry(config.character_registry_dir.expanduser())

    async def run_full_pipeline(
        self,
        concept: str,
        scene_graph: SceneGraph | None = None,
    ) -> Path:
        """Execute the complete pipeline from concept to rendered film."""
        await self._emit(EventType.PIPELINE_START, message=f"Starting pipeline: {concept[:80]}")

        try:
            # Stage 1: Story
            if scene_graph is None and not self.state.is_stage_completed("story"):
                scene_graph = await self.run_story(concept)
            elif scene_graph is None:
                scene_graph = self.project.load_scene_graph()
                if scene_graph is None:
                    scene_graph = await self.run_story(concept)

            assert scene_graph is not None

            # Stage 2: Characters
            if not self.state.is_stage_completed("characters"):
                await self.run_characters(scene_graph)

            # Stage 3: Visual generation
            if not self.state.is_stage_completed("visual"):
                await self.run_visual(scene_graph)

            # Stage 4: Audio
            if not self.state.is_stage_completed("audio"):
                await self.run_audio(scene_graph)

            # Stage 5: Consistency
            if self.config.consistency.enabled and not self.state.is_stage_completed(
                "consistency"
            ):
                await self.run_consistency(scene_graph)

            # Stage 6-7: Assembly and export
            if not self.state.is_stage_completed("assembly"):
                output = await self.run_assembly(scene_graph)
            else:
                output = self.project.export_dir / "final.mp4"

            await self._emit(
                EventType.PIPELINE_COMPLETE,
                message=f"Pipeline complete! Output: {output}",
                data={"output": str(output), "cost": str(self.state.total_cost_usd)},
            )
            return output

        except Exception as e:
            await self._emit(
                EventType.PIPELINE_ERROR,
                message=f"Pipeline failed: {e}",
            )
            self.state.save(self.project.pipeline_state_path)
            raise

    async def run_story(self, concept: str) -> SceneGraph:
        """Stage 1: Generate the SceneGraph from a concept."""
        self.state.mark_stage_started("story")
        self.state.save(self.project.pipeline_state_path)
        await self._emit(EventType.STAGE_START, stage="story")

        writer = LLMStoryWriter(self.config.story)
        scene_graph = await writer.write(concept)

        self.project.save_scene_graph(scene_graph)
        self.state.mark_stage_completed("story")
        self.state.save(self.project.pipeline_state_path)
        await self._emit(
            EventType.STAGE_COMPLETE,
            stage="story",
            message=f"Generated {scene_graph.total_shots} shots across {len(scene_graph.all_scenes())} scenes",
        )
        return scene_graph

    async def run_characters(self, scene_graph: SceneGraph) -> None:
        """Stage 2: Create or retrieve character identities."""
        self.state.mark_stage_started("characters")
        await self._emit(EventType.STAGE_START, stage="characters")

        for char in scene_graph.characters:
            existing = self.registry.get(char.character_id)
            if existing:
                logger.info("Character '%s' found in registry", char.character_id)
                continue

            # Create new character identity
            identity = CharacterIdentity(
                character_id=char.character_id,
                display_name=char.name,
                physical_description=char.physical_description,
            )

            # Generate reference images using image generator
            await self._generate_character_references(identity, scene_graph)

            self.registry.register(identity)
            logger.info("Registered new character: %s", char.character_id)

        self.state.mark_stage_completed("characters")
        self.state.save(self.project.pipeline_state_path)
        await self._emit(EventType.STAGE_COMPLETE, stage="characters")

    async def _generate_character_references(
        self, identity: CharacterIdentity, scene_graph: SceneGraph
    ) -> None:
        """Generate reference images for a new character."""
        # Try to get an image generator
        visual_config = self.config.visual
        provider_name = visual_config.default_provider
        provider_config = visual_config.providers.get(provider_name)
        if not provider_config:
            logger.warning("No image generator configured — skipping reference generation")
            return

        try:
            gen = get_image_generator(provider_name, provider_config)
        except ValueError:
            logger.warning("Provider '%s' does not support image generation", provider_name)
            return

        char_dir = self.registry.character_dir(identity.character_id)
        char_dir.mkdir(parents=True, exist_ok=True)

        # Generate front-facing reference
        prompt = (
            f"portrait photo of {identity.physical_description}, "
            f"front facing, neutral expression, studio lighting, "
            f"clean background, high detail, photorealistic"
        )

        result = await gen.generate_image(
            prompt=prompt,
            width=1024,
            height=1024,
            output_dir=char_dir,
        )

        if result.output_path:
            identity.reference_images.append(result.output_path.name)

            # Extract face embedding
            try:
                analyzer = FaceAnalyzer(self.config.consistency)
                embedding = analyzer.extract_embedding(result.output_path)
                if embedding is not None:
                    emb_path = char_dir / "face_embedding.npy"
                    analyzer.save_embedding(embedding, emb_path)
                    identity.face_embedding_path = "face_embedding.npy"
            except Exception as e:
                logger.warning("Failed to extract face embedding: %s", e)

    async def run_visual(self, scene_graph: SceneGraph) -> None:
        """Stage 3: Generate video clips for all shots."""
        self.state.mark_stage_started("visual")
        await self._emit(EventType.STAGE_START, stage="visual")

        router = ShotRouter(self.config)
        prompt_builder = PromptBuilder(scene_graph)
        shots = scene_graph.all_shots()

        # Generate shots in parallel with concurrency control
        semaphore = asyncio.Semaphore(
            self.config.visual.providers.get(
                self.config.visual.default_provider,
                type("C", (), {"max_concurrent": 3})(),  # type: ignore[arg-type]
            ).max_concurrent
        )

        tasks = []
        for shot in shots:
            # Skip already-completed shots
            shot_state = self.state.shots.get(shot.shot_id)
            if shot_state and shot_state.status == StageStatus.COMPLETED:
                continue

            provider_name = router.route(shot)
            task = self._generate_shot(
                shot, provider_name, prompt_builder, semaphore
            )
            tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Check if any shots actually produced video files
        failed_shots = [
            (sid, ss.error)
            for sid, ss in self.state.shots.items()
            if ss.status == StageStatus.FAILED
        ]
        succeeded = sum(
            1 for ss in self.state.shots.values()
            if ss.status == StageStatus.COMPLETED
        )

        if succeeded == 0 and failed_shots:
            errors = "; ".join(f"{sid}: {err}" for sid, err in failed_shots[:5])
            self.state.mark_stage_failed("visual", errors)
            self.state.save(self.project.pipeline_state_path)
            raise RuntimeError(
                f"Visual generation failed: all {len(failed_shots)} shots failed. "
                f"First errors: {errors}"
            )

        self.state.mark_stage_completed("visual")
        self.state.save(self.project.pipeline_state_path)
        await self._emit(EventType.STAGE_COMPLETE, stage="visual")

    async def _generate_shot(
        self,
        shot: Shot,
        provider_name: str,
        prompt_builder: PromptBuilder,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Generate a single shot video clip."""
        async with semaphore:
            await self._emit(
                EventType.SHOT_GENERATION_START,
                message=f"Generating shot {shot.shot_id} via {provider_name}",
            )

            shot_dir = self.project.shot_dir(shot.shot_id)
            provider_config = self.config.visual.providers.get(provider_name)

            if not provider_config:
                logger.error("No config for provider: %s", provider_name)
                self.state.mark_shot(
                    shot.shot_id, status=StageStatus.FAILED, error="No provider config"
                )
                return

            try:
                gen = get_video_generator(provider_name, provider_config)

                # Build prompt
                prompt = prompt_builder.build_shot_prompt(shot)

                # Save prompt for reproducibility
                (shot_dir / "prompt.txt").write_text(prompt)

                # Gather character reference images
                ref_images = self._gather_character_refs(shot)

                # Check if first-frame strategy should be used
                first_frame = None
                if (
                    self.config.visual.character_consistency_strategy == "first_frame"
                    and shot.characters_present
                ):
                    first_frame = await self._generate_first_frame(
                        shot, prompt_builder, shot_dir
                    )

                result = await gen.generate_video(
                    prompt=prompt,
                    duration_seconds=shot.duration_seconds,
                    reference_images=ref_images if ref_images else None,
                    first_frame=first_frame,
                    aspect_ratio=self.config.visual.default_aspect_ratio,
                    output_dir=shot_dir,
                )

                # Rename output to standard name
                if result.output_path and result.output_path.exists():
                    raw_path = shot_dir / "raw.mp4"
                    if result.output_path != raw_path:
                        result.output_path.rename(raw_path)
                        result.output_path = raw_path

                # Save metadata
                metadata = {
                    "provider": provider_name,
                    "generation_id": result.generation_id,
                    "seed": result.seed,
                    "cost_usd": result.cost_usd,
                    "status": result.status.value,
                }
                (shot_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

                self.state.mark_shot(
                    shot.shot_id,
                    status=StageStatus.COMPLETED,
                    provider=provider_name,
                    generation_id=result.generation_id,
                    cost_usd=result.cost_usd,
                )
                self.state.add_cost(result.cost_usd)
                self.state.save(self.project.pipeline_state_path)

                await self._emit(
                    EventType.SHOT_GENERATION_COMPLETE,
                    message=f"Shot {shot.shot_id} complete",
                )

            except Exception as e:
                logger.error("Failed to generate shot %s: %s", shot.shot_id, e)
                self.state.mark_shot(
                    shot.shot_id, status=StageStatus.FAILED, error=str(e)
                )
                await self._emit(
                    EventType.SHOT_GENERATION_ERROR,
                    message=f"Shot {shot.shot_id} failed: {e}",
                )

    async def _generate_first_frame(
        self,
        shot: Shot,
        prompt_builder: PromptBuilder,
        shot_dir: Path,
    ) -> Path | None:
        """Generate a first frame using image generation with LoRA for character consistency."""
        provider_name = self.config.visual.default_provider
        provider_config = self.config.visual.providers.get(provider_name)
        if not provider_config:
            return None

        try:
            img_gen = get_image_generator(provider_name, provider_config)
        except ValueError:
            return None

        prompt = prompt_builder.build_first_frame_prompt(shot)

        # Find LoRA for the primary character
        lora_path = None
        lora_trigger = ""
        if shot.characters_present:
            identity = self.registry.get(shot.characters_present[0])
            if identity and identity.lora_path:
                char_dir = self.registry.character_dir(identity.character_id)
                lora_path = char_dir / identity.lora_path
                lora_trigger = identity.lora_trigger_word

        result = await img_gen.generate_image(
            prompt=prompt,
            lora_path=lora_path,
            lora_trigger_word=lora_trigger,
            width=1920,
            height=1080,
            output_dir=shot_dir,
        )

        if result.output_path and result.output_path.exists():
            first_frame_path = shot_dir / "first_frame.png"
            if result.output_path != first_frame_path:
                result.output_path.rename(first_frame_path)
            return first_frame_path

        return None

    def _gather_character_refs(self, shot: Shot) -> list[Path]:
        """Gather reference images for characters in a shot."""
        refs = []
        for char_id in shot.characters_present:
            identity = self.registry.get(char_id)
            if identity:
                char_dir = self.registry.character_dir(char_id)
                for img_name in identity.reference_images:
                    img_path = char_dir / img_name
                    if img_path.exists():
                        refs.append(img_path)
        return refs

    async def run_audio(self, scene_graph: SceneGraph) -> None:
        """Stage 4: Generate dialogue, music, and sound effects."""
        self.state.mark_stage_started("audio")
        await self._emit(EventType.STAGE_START, stage="audio")

        tasks = [
            self._generate_dialogue(scene_graph),
            self._generate_music(scene_graph),
            self._generate_sfx(scene_graph),
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        self.state.mark_stage_completed("audio")
        self.state.save(self.project.pipeline_state_path)
        await self._emit(EventType.STAGE_COMPLETE, stage="audio")

    async def _generate_dialogue(self, scene_graph: SceneGraph) -> None:
        """Generate TTS for all dialogue lines."""
        tts_config = self.config.audio.tts
        provider_config = tts_config.providers.get(tts_config.provider)
        if not provider_config:
            logger.warning("No TTS provider configured — skipping dialogue")
            return

        # Import appropriate TTS provider
        if tts_config.provider == "elevenlabs":
            from cinemaestro.audio.tts.elevenlabs import ElevenLabsTTS
            tts = ElevenLabsTTS(provider_config)
        elif tts_config.provider == "xtts":
            from cinemaestro.audio.tts.xtts import XTTSTTS
            tts = XTTSTTS(provider_config)
        else:
            logger.warning("Unknown TTS provider: %s", tts_config.provider)
            return

        for shot in scene_graph.all_shots():
            for i, line in enumerate(shot.dialogue):
                output_path = self.project.dialogue_dir / f"{shot.shot_id}_line{i}.wav"
                if output_path.exists():
                    continue

                # Get character voice
                voice_id = None
                voice_ref = None
                identity = self.registry.get(line.character_id)
                if identity:
                    voice_id = identity.elevenlabs_voice_id
                    if identity.voice_reference_audio:
                        char_dir = self.registry.character_dir(identity.character_id)
                        voice_ref = char_dir / identity.voice_reference_audio

                await tts.synthesize(
                    text=line.text,
                    voice_id=voice_id,
                    voice_reference=voice_ref,
                    emotion=line.emotion,
                    output_path=output_path,
                )

    async def _generate_music(self, scene_graph: SceneGraph) -> None:
        """Generate music for each scene."""
        music_config = self.config.audio.music
        provider_config = music_config.providers.get(music_config.provider)
        if not provider_config:
            logger.warning("No music provider configured — skipping music")
            return

        if music_config.provider == "stable_audio":
            from cinemaestro.audio.music.stable_audio import StableAudioMusic
            gen = StableAudioMusic(provider_config)
        else:
            logger.warning("Unknown music provider: %s", music_config.provider)
            return

        for scene in scene_graph.all_scenes():
            output_path = self.project.music_dir / f"{scene.scene_id}.wav"
            if output_path.exists():
                continue

            prompt = scene.music_direction or f"{scene.mood} {scene_graph.music_theme}"
            if not prompt.strip():
                continue

            await gen.generate(
                prompt=prompt,
                duration_seconds=scene.duration_seconds,
                output_path=output_path,
            )

    async def _generate_sfx(self, scene_graph: SceneGraph) -> None:
        """Generate sound effects for shots."""
        # SFX generation follows same pattern — skipping if no provider
        pass

    async def run_consistency(self, scene_graph: SceneGraph) -> None:
        """Stage 5: Verify and repair character consistency."""
        self.state.mark_stage_started("consistency")
        await self._emit(EventType.STAGE_START, stage="consistency")

        analyzer = FaceAnalyzer(self.config.consistency)
        swapper = FaceSwapper()

        for shot in scene_graph.all_shots():
            if not shot.characters_present:
                continue

            shot_dir = self.project.shots_dir / shot.shot_id
            video_path = shot_dir / "raw.mp4"
            if not video_path.exists():
                continue

            for char_id in shot.characters_present:
                identity = self.registry.get(char_id)
                if not identity or not identity.face_embedding_path:
                    continue

                char_dir = self.registry.character_dir(char_id)
                emb_path = char_dir / identity.face_embedding_path

                if not emb_path.exists():
                    continue

                score = await analyzer.check_shot(
                    video_path=video_path,
                    character_id=char_id,
                    reference_embedding_path=emb_path,
                )
                score.shot_id = shot.shot_id

                await self._emit(
                    EventType.CONSISTENCY_CHECK_RESULT,
                    message=(
                        f"Shot {shot.shot_id} / {char_id}: "
                        f"mean={score.mean_similarity:.3f}, "
                        f"{'PASS' if score.passed else 'FAIL'}"
                    ),
                )

                # Auto-repair if needed
                if not score.passed and self.config.consistency.auto_repair:
                    ref_images = identity.reference_images
                    if ref_images:
                        ref_path = char_dir / ref_images[0]
                        corrected_path = shot_dir / "corrected.mp4"
                        await swapper.repair_shot(
                            video_path=video_path,
                            character_id=char_id,
                            reference_image=ref_path,
                            failing_frames=score.failing_frame_indices,
                            output_path=corrected_path,
                        )
                        self.state.mark_shot(
                            shot.shot_id,
                            repaired=True,
                            consistency_score=score.mean_similarity,
                        )

        self.state.mark_stage_completed("consistency")
        self.state.save(self.project.pipeline_state_path)
        await self._emit(EventType.STAGE_COMPLETE, stage="consistency")

    async def run_assembly(self, scene_graph: SceneGraph) -> Path:
        """Stage 6-7: Build timeline, mix audio, assemble video, export."""
        self.state.mark_stage_started("assembly")
        await self._emit(EventType.STAGE_START, stage="assembly")

        # Build timeline
        builder = TimelineBuilder(self.project)
        timeline = builder.build(scene_graph)

        # Mix audio
        mixer = AudioMixer(self.config.audio)
        audio_mix_path = self.project.assembly_dir / "audio_mix.wav"
        try:
            mixer.mix(timeline, audio_mix_path)
        except Exception as e:
            logger.warning("Audio mixing failed: %s — proceeding without audio", e)
            audio_mix_path = None

        # Generate subtitles
        sub_gen = SubtitleGenerator()
        srt_path = self.project.assembly_dir / "subtitles.srt"
        sub_gen.generate_srt(timeline, srt_path)

        # Assemble video
        editor = VideoEditor(self.config.assembly, self.config.export)
        draft_path = self.project.assembly_dir / "draft.mp4"
        editor.assemble(timeline, audio_mix_path, draft_path)

        # Burn subtitles if configured
        final_path = self.project.export_dir / "final.mp4"
        if self.config.assembly.burn_subtitles and srt_path.exists():
            editor.add_subtitles(draft_path, srt_path, final_path)
        else:
            import shutil
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(draft_path, final_path)

        self.state.mark_stage_completed("assembly")
        self.state.save(self.project.pipeline_state_path)
        await self._emit(
            EventType.STAGE_COMPLETE,
            stage="assembly",
            message=f"Final output: {final_path}",
        )
        return final_path

    async def regenerate_shots(self, shot_ids: list[str]) -> None:
        """Re-generate specific shots (e.g., after manual review)."""
        scene_graph = self.project.load_scene_graph()
        if not scene_graph:
            raise ValueError("No scene graph found — run story stage first")

        router = ShotRouter(self.config)
        prompt_builder = PromptBuilder(scene_graph)
        semaphore = asyncio.Semaphore(3)

        tasks = []
        for shot_id in shot_ids:
            shot = next((s for s in scene_graph.all_shots() if s.shot_id == shot_id), None)
            if not shot:
                logger.warning("Shot not found: %s", shot_id)
                continue

            # Reset shot state
            self.state.mark_shot(shot_id, status=StageStatus.PENDING)
            provider_name = router.route(shot)
            tasks.append(
                self._generate_shot(shot, provider_name, prompt_builder, semaphore)
            )

        await asyncio.gather(*tasks)
        self.state.save(self.project.pipeline_state_path)

    async def _emit(self, event_type: EventType, **kwargs: object) -> None:
        event = PipelineEvent(event_type=event_type, **kwargs)  # type: ignore[arg-type]
        await self.event_bus.emit(event)
