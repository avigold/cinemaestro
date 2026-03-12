#!/usr/bin/env python3
"""Production runner for 'The Explorers' — a 60-second cinematic piece.

Runs each pipeline stage with explicit control over providers and parameters.
Can be re-run safely; skips completed work.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

import httpx

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cinemaestro.config import load_config
from cinemaestro.core.project import Project
from cinemaestro.core.scene_graph import SceneGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("explorers")

# ── Configuration ──────────────────────────────────────────────────────

NARRATOR_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # George - Warm, Captivating Storyteller
NARRATOR_MODEL = "eleven_v3"  # Latest, most expressive model
TITLE_CARD_SHOT_ID = "act3_s3_shot1"

# fal.ai model endpoints
FAL_IMAGE_MODEL = "fal-ai/flux/dev"
FAL_VIDEO_MODEL = "fal-ai/kling-video/v2/master/image-to-video"
FAL_TEXT_TO_VIDEO_MODEL = "fal-ai/kling-video/v2/master/text-to-video"


def load_project():
    config = load_config(project_dir=Path("projects/explorers"))
    project = Project(Path("projects/explorers"))
    sg = project.load_scene_graph()
    assert sg is not None, "No scene_graph.yaml found"
    return config, project, sg


# ── Stage 1: Generate First Frames ─────────────────────────────────────


async def generate_first_frames(config, project: Project, sg: SceneGraph):
    """Generate a still first frame for every shot using Flux via fal.ai."""
    from cinemaestro.visual.prompt_builder import PromptBuilder

    log.info("═══ STAGE 1: First Frame Generation (Flux via fal.ai) ═══")

    fal_key = config.fal_key
    if not fal_key:
        log.error("No FAL_KEY configured")
        return

    prompt_builder = PromptBuilder(sg)
    shots = sg.all_shots()

    async with httpx.AsyncClient(
        headers={"Authorization": f"Key {fal_key}"},
        timeout=120,
    ) as client:
        for shot in shots:
            shot_dir = project.shot_dir(shot.shot_id)
            first_frame = shot_dir / "first_frame.png"

            # Skip title card — we'll generate it programmatically
            if shot.shot_id == TITLE_CARD_SHOT_ID:
                log.info("  [skip] %s — title card (generated programmatically)", shot.shot_id)
                continue

            # Skip if already done
            if first_frame.exists():
                log.info("  [done] %s first_frame.png exists", shot.shot_id)
                continue

            # Build prompt — use first_frame_prompt for character shots, shot_prompt for others
            if shot.characters_present:
                prompt = prompt_builder.build_first_frame_prompt(shot)
            else:
                prompt = prompt_builder.build_shot_prompt(shot)

            log.info("  [gen]  %s — %s", shot.shot_id, prompt[:80] + "...")

            payload = {
                "prompt": prompt,
                "image_size": {"width": 1920, "height": 1080},
                "num_images": 1,
            }

            try:
                resp = await client.post(
                    f"https://fal.run/{FAL_IMAGE_MODEL}",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                images = data.get("images", [])
                if images:
                    img_url = images[0]["url"]
                    # Download
                    dl = await client.get(img_url)
                    dl.raise_for_status()
                    first_frame.write_bytes(dl.content)
                    log.info("         → saved %s (%d KB)", first_frame.name, len(dl.content) // 1024)
                else:
                    log.warning("         → no image returned for %s", shot.shot_id)

                # Save prompt for reproducibility
                (shot_dir / "first_frame_prompt.txt").write_text(prompt)

            except Exception as e:
                log.error("         → FAILED: %s", e)


# ── Stage 2: Generate Video Clips ──────────────────────────────────────


def generate_videos(config, project: Project, sg: SceneGraph):
    """Animate first frames into video clips using Kling via fal.ai.

    Uses fal_client for reliable queue management and polling.
    Submits all shots first, then collects results in parallel.
    """
    import os
    import fal_client
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from cinemaestro.visual.prompt_builder import PromptBuilder

    log.info("═══ STAGE 2: Video Generation (Kling v2 via fal.ai) ═══")

    os.environ["FAL_KEY"] = config.fal_key
    if not config.fal_key:
        log.error("No FAL_KEY configured")
        return

    prompt_builder = PromptBuilder(sg)
    shots = sg.all_shots()

    # Collect shots that need generation
    pending = []
    for shot in shots:
        shot_dir = project.shot_dir(shot.shot_id)
        raw_video = shot_dir / "raw.mp4"

        if shot.shot_id == TITLE_CARD_SHOT_ID:
            log.info("  [skip] %s — title card", shot.shot_id)
            continue
        if raw_video.exists():
            log.info("  [done] %s raw.mp4 exists", shot.shot_id)
            continue

        pending.append(shot)

    if not pending:
        log.info("  All videos already generated!")
        return

    log.info("  Generating %d video clips...", len(pending))

    def _generate_one(shot):
        """Generate a single video clip (runs in thread)."""
        shot_dir = project.shot_dir(shot.shot_id)
        raw_video = shot_dir / "raw.mp4"
        first_frame = shot_dir / "first_frame.png"
        prompt = prompt_builder.build_shot_prompt(shot)
        # Kling only accepts '5' or '10' — use '5' for all shots, trim in assembly
        duration = "5"

        log.info("  [sub]  %s — %.1fs, %s", shot.shot_id, shot.duration_seconds, shot.shot_type.value)

        try:
            if first_frame.exists():
                # Upload first frame and use image-to-video
                img_url = fal_client.upload_file(str(first_frame))
                model = FAL_VIDEO_MODEL
                arguments = {
                    "prompt": prompt,
                    "image_url": img_url,
                    "duration": duration,
                    "aspect_ratio": "16:9",
                }
            else:
                model = FAL_TEXT_TO_VIDEO_MODEL
                arguments = {
                    "prompt": prompt,
                    "duration": duration,
                    "aspect_ratio": "16:9",
                }

            result = fal_client.subscribe(
                model,
                arguments=arguments,
            )

            video = result.get("video", {})
            video_url = video.get("url", "")

            if video_url:
                dl = httpx.get(video_url, timeout=120)
                dl.raise_for_status()
                raw_video.write_bytes(dl.content)
                log.info("  [done] %s → raw.mp4 (%d KB)", shot.shot_id, len(dl.content) // 1024)

                meta = {"model": model, "duration": duration, "video_url": video_url}
                (shot_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
                return shot.shot_id, True
            else:
                log.warning("  [fail] %s — no video URL in result", shot.shot_id)
                return shot.shot_id, False

        except Exception as e:
            log.error("  [fail] %s — %s", shot.shot_id, e)
            return shot.shot_id, False

    # Run up to 3 shots in parallel
    max_parallel = 3
    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {pool.submit(_generate_one, shot): shot for shot in pending}
        for future in as_completed(futures):
            shot_id, success = future.result()
            if success:
                completed += 1
            else:
                failed += 1
            log.info("  Progress: %d/%d done, %d failed", completed, len(pending), failed)

    log.info("  Video generation complete: %d succeeded, %d failed", completed, failed)


# ── Stage 3: Title Card ────────────────────────────────────────────────


def generate_title_card(project: Project):
    """Generate the 'THINK BIGGER' title card as a video."""
    log.info("═══ STAGE 3: Title Card Generation ═══")

    shot_dir = project.shot_dir(TITLE_CARD_SHOT_ID)
    raw_video = shot_dir / "raw.mp4"

    if raw_video.exists():
        log.info("  [done] title card already exists")
        return

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        log.error("Pillow not installed — run: pip install Pillow")
        return

    try:
        from moviepy import ImageClip
    except ImportError:
        log.error("moviepy not installed")
        return

    # Create title card image
    width, height = 1920, 1080
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Try to use a nice font, fallback to default
    main_size = 96
    sub_size = 48
    try:
        main_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", main_size)
        sub_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", sub_size)
    except (OSError, IOError):
        try:
            main_font = ImageFont.truetype("/System/Library/Fonts/SFNSDisplay.ttf", main_size)
            sub_font = ImageFont.truetype("/System/Library/Fonts/SFNSDisplay.ttf", sub_size)
        except (OSError, IOError):
            main_font = ImageFont.load_default()
            sub_font = ImageFont.load_default()

    # Draw "THINK BIGGER"
    main_text = "THINK BIGGER"
    bbox = draw.textbbox((0, 0), main_text, font=main_font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (width - tw) // 2
    y = (height - th) // 2 - 40
    draw.text((x, y), main_text, fill=(255, 255, 255), font=main_font)

    # Draw "Protection Products & Services"
    sub_text = "Protection Products & Services"
    bbox2 = draw.textbbox((0, 0), sub_text, font=sub_font)
    sw = bbox2[2] - bbox2[0]
    sx = (width - sw) // 2
    sy = y + th + 50
    draw.text((sx, sy), sub_text, fill=(220, 220, 220), font=sub_font)

    # Save image
    card_img_path = shot_dir / "first_frame.png"
    img.save(str(card_img_path))
    log.info("  [gen]  title card image saved")

    # Create 4-second video from still image
    import numpy as np
    frame = np.array(img)
    clip = ImageClip(frame, duration=4.0)
    clip.write_videofile(
        str(raw_video),
        fps=24,
        codec="libx264",
        audio=False,
        logger=None,
    )
    log.info("  [gen]  title card video saved (%d KB)", raw_video.stat().st_size // 1024)


# ── Stage 4: Narration Audio ──────────────────────────────────────────


async def generate_narration(config, project: Project, sg: SceneGraph):
    """Generate narration audio for all dialogue lines via ElevenLabs.

    Uses eleven_v3 with carefully crafted text that includes natural
    pacing cues (ellipses, dashes, emphasis) so the narrator sounds
    like a human storyteller, not a text-to-speech engine.
    """
    log.info("═══ STAGE 4: Narration (ElevenLabs v3) ═══")

    api_key = config.elevenlabs_api_key
    if not api_key:
        log.error("No ELEVENLABS_API_KEY configured")
        return

    # Hand-crafted text with natural pacing and emphasis cues.
    # ElevenLabs v3 responds to punctuation, ellipses, and dashes for timing.
    DIRECTED_LINES = {
        ("act1_s1_shot1", 0): {
            "text": "They told Gutenberg... he'd made a faster pen.",
            "stability": 0.45,
            "similarity_boost": 0.75,
            "style": 0.15,
        },
        ("act1_s1_shot2", 0): {
            "text": "He'd made a world... that could read.",
            "stability": 0.35,
            "similarity_boost": 0.75,
            "style": 0.4,
        },
        ("act1_s2_shot1", 0): {
            "text": "They told Edison... he'd made a better candle.",
            "stability": 0.45,
            "similarity_boost": 0.75,
            "style": 0.2,
        },
        ("act1_s2_shot2", 0): {
            "text": "He'd made a world — that never had to sleep.",
            "stability": 0.3,
            "similarity_boost": 0.75,
            "style": 0.5,
        },
        ("act2_s1_shot1", 0): {
            "text": "Every generation gets a moment — where the tools outpace the ambition.",
            "stability": 0.4,
            "similarity_boost": 0.8,
            "style": 0.3,
        },
        ("act2_s1_shot2", 0): {
            "text": "Where the easy thing... is to move faster.",
            "stability": 0.45,
            "similarity_boost": 0.75,
            "style": 0.25,
        },
        ("act2_s1_shot3", 0): {
            "text": "And the brave thing... is to look up.",
            "stability": 0.35,
            "similarity_boost": 0.8,
            "style": 0.45,
        },
        ("act3_s1_shot1", 0): {
            "text": "Here's to the explorers. The ones who hear — 'it can't be done' — and ask... 'why has no one tried?'",
            "stability": 0.3,
            "similarity_boost": 0.8,
            "style": 0.7,
        },
        ("act3_s2_shot1", 0): {
            "text": "The tools are ready.",
            "stability": 0.5,
            "similarity_boost": 0.8,
            "style": 0.3,
        },
        ("act3_s2_shot2", 0): {
            "text": "What's worth building?",
            "stability": 0.4,
            "similarity_boost": 0.85,
            "style": 0.2,
        },
    }

    shots = sg.all_shots()

    async with httpx.AsyncClient(
        base_url="https://api.elevenlabs.io/v1",
        headers={"xi-api-key": api_key},
        timeout=60,
    ) as client:
        for shot in shots:
            for i, line in enumerate(shot.dialogue):
                output_path = project.dialogue_dir / f"{shot.shot_id}_line{i}.mp3"

                if output_path.exists():
                    log.info("  [done] %s line %d", shot.shot_id, i)
                    continue

                # Use directed version if available, otherwise fall back to raw text
                directed = DIRECTED_LINES.get((shot.shot_id, i))
                if directed:
                    text = directed["text"]
                    stability = directed["stability"]
                    similarity_boost = directed["similarity_boost"]
                    style = directed["style"]
                else:
                    text = line.text
                    stability = 0.4
                    similarity_boost = 0.75
                    style = 0.3

                log.info('  [gen]  %s line %d: "%s"', shot.shot_id, i, text[:60])

                payload = {
                    "text": text,
                    "model_id": NARRATOR_MODEL,
                    "voice_settings": {
                        "stability": stability,
                        "similarity_boost": similarity_boost,
                        "style": style,
                        "use_speaker_boost": True,
                    },
                }

                try:
                    resp = await client.post(
                        f"/text-to-speech/{NARRATOR_VOICE_ID}",
                        json=payload,
                        headers={"Accept": "audio/mpeg"},
                    )
                    resp.raise_for_status()
                    output_path.write_bytes(resp.content)
                    log.info("         → saved (%d KB)", len(resp.content) // 1024)

                except Exception as e:
                    log.error("         → FAILED: %s", e)


# ── Stage 5: Music Score ──────────────────────────────────────────────


async def generate_music(config, project: Project, sg: SceneGraph):
    """Generate the music score via Stability AI Stable Audio."""
    log.info("═══ STAGE 5: Music Score (Stable Audio) ═══")

    api_key = config.stability_api_key
    if not api_key:
        log.error("No STABILITY_API_KEY configured")
        return

    # Generate one continuous score for the whole film
    music_path = project.music_dir / "score.wav"
    if music_path.exists():
        log.info("  [done] score.wav already exists")
        return

    # Build a detailed music prompt from the scene graph
    prompt = (
        f"{sg.music_theme}. "
        f"Cinematic film score, {sg.tone} tone, {sg.genre} genre. "
        f"Starts with solo piano, builds gradually to full orchestral strings, "
        f"emotionally soaring climax, then resolves to a final sustained piano chord. "
        f"Epic, intimate, uplifting. No vocals."
    )

    log.info("  [gen]  score — %.0fs", sg.target_duration_seconds)
    log.info("         prompt: %s", prompt[:100] + "...")

    async with httpx.AsyncClient(
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "audio/*",
        },
        timeout=180,
    ) as client:
        try:
            resp = await client.post(
                "https://api.stability.ai/v2beta/audio/generate",
                json={
                    "prompt": prompt,
                    "seconds_total": min(sg.target_duration_seconds, 180),
                    "seconds_start": 0,
                },
            )
            resp.raise_for_status()
            music_path.write_bytes(resp.content)
            log.info("  [gen]  → saved score.wav (%d KB)", len(resp.content) // 1024)
        except Exception as e:
            log.error("  FAILED: %s", e)
            if hasattr(e, "response") and e.response is not None:
                log.error("  Response: %s", e.response.text[:500])


# ── Stage 6: Assembly ─────────────────────────────────────────────────


def assemble_film(config, project: Project, sg: SceneGraph):
    """Assemble all generated assets into the final film."""
    from pydub import AudioSegment
    from moviepy import (
        AudioFileClip,
        VideoFileClip,
        concatenate_videoclips,
    )
    from moviepy.video.fx import FadeIn, FadeOut

    log.info("═══ STAGE 6: Film Assembly ═══")

    final_path = project.export_dir / "the_explorers.mp4"
    if final_path.exists():
        final_path.unlink()  # always re-assemble

    TARGET_W, TARGET_H = 1920, 1080

    # ── Build video sequence ──
    log.info("  Building video sequence...")
    video_clips = []
    shots = sg.all_shots()
    missing_shots = []

    for shot in shots:
        shot_dir = project.shots_dir / shot.shot_id

        # Find best video file
        video_file = None
        for name in ["corrected.mp4", "raw.mp4"]:
            p = shot_dir / name
            if p.exists():
                video_file = p
                break

        if not video_file:
            log.warning("  [miss] %s — no video file", shot.shot_id)
            missing_shots.append(shot.shot_id)
            continue

        vc = VideoFileClip(str(video_file))

        # Resize to 1920x1080 if needed (Kling outputs 1284x716)
        if vc.size != [TARGET_W, TARGET_H]:
            vc = vc.resized((TARGET_W, TARGET_H))

        # Trim to desired duration (Kling generates 5s, many shots are shorter)
        target_dur = shot.duration_seconds
        if vc.duration > target_dur:
            vc = vc.subclipped(0, target_dur)

        # Apply fade transitions
        if shot.transition_in.value == "fade_in":
            vc = vc.with_effects([FadeIn(0.5)])
        if shot.transition_out.value in ("fade_to_black", "fade_out"):
            vc = vc.with_effects([FadeOut(0.5)])
        if shot.transition_out.value == "dissolve":
            vc = vc.with_effects([FadeOut(0.3)])
        if shot.transition_in.value == "dissolve":
            vc = vc.with_effects([FadeIn(0.3)])

        video_clips.append(vc)
        log.info("  [add]  %s — %.1fs", shot.shot_id, vc.duration)

    if missing_shots:
        log.warning("  Missing %d shots: %s", len(missing_shots), missing_shots)

    if not video_clips:
        log.error("  No video clips found — cannot assemble")
        return

    # Concatenate video — use "chain" to avoid compose-mode black borders
    final_video = concatenate_videoclips(video_clips, method="chain")
    log.info("  Video assembled: %.1fs, %dx%d", final_video.duration, final_video.w, final_video.h)

    # ── Build audio mix ──
    log.info("  Mixing audio...")

    # Load music first to determine if we need extra time for it to resolve
    # Support both wav and mp3 score files
    music_file = project.music_dir / "score.wav"
    if not music_file.exists():
        music_file = project.music_dir / "score.mp3"
    music_tail_seconds = 0.0
    music = None
    if music_file.exists():
        music = AudioSegment.from_file(str(music_file))
        music = music - 16  # sit under dialogue
        music = music.fade_in(3000)
        music = music.fade_out(5000)  # generous 5s fade out for natural resolution
        # If music extends beyond video, add extra black frames to let it breathe
        music_duration_s = len(music) / 1000
        if music_duration_s > final_video.duration:
            music_tail_seconds = min(music_duration_s - final_video.duration, 2.0)
        log.info("  Music: %.1fs (tail: %.1fs)", music_duration_s, music_tail_seconds)

    # Extend video with black frames if music needs room to resolve
    if music_tail_seconds > 0:
        import numpy as np
        black_frame = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
        from moviepy import ImageClip
        tail_clip = ImageClip(black_frame, duration=music_tail_seconds)
        final_video = concatenate_videoclips([final_video, tail_clip], method="chain")
        log.info("  Extended video by %.1fs for music resolution", music_tail_seconds)

    duration_ms = int(final_video.duration * 1000)

    # Start with stereo silence at 44100Hz
    audio_mix = AudioSegment.silent(duration=duration_ms, frame_rate=44100)

    # Layer music FIRST (it's the bed)
    if music is not None:
        # Trim to extended video duration if still longer
        if len(music) > duration_ms:
            music = music[:duration_ms]
        audio_mix = audio_mix.overlay(music, position=0)
        log.info("  Layered music bed (%.1fs, -16dB)", len(music) / 1000)

    # Layer dialogue on top (convert mono mp3 to stereo and match sample rate)
    current_time = 0.0
    dialogue_count = 0
    for shot in shots:
        if shot.shot_id in missing_shots:
            current_time += shot.duration_seconds
            continue

        for i, line in enumerate(shot.dialogue):
            for ext in [".mp3", ".wav"]:
                dialogue_file = project.dialogue_dir / f"{shot.shot_id}_line{i}{ext}"
                if dialogue_file.exists():
                    audio = AudioSegment.from_file(str(dialogue_file))
                    # Convert mono to stereo if needed
                    if audio.channels == 1:
                        audio = audio.set_channels(2)
                    # Match sample rate
                    if audio.frame_rate != 44100:
                        audio = audio.set_frame_rate(44100)
                    pos_ms = int(current_time * 1000)
                    audio_mix = audio_mix.overlay(audio, position=pos_ms)
                    dialogue_count += 1
                    log.info("    dialogue: %s @ %.1fs (%.1fs)", shot.shot_id, current_time, len(audio) / 1000)
                    break

        current_time += shot.duration_seconds

    log.info("  Layered %d dialogue clips", dialogue_count)

    # Gentle normalization — target -6 dBFS to avoid any clipping
    if audio_mix.dBFS > -6:
        change = -6.0 - audio_mix.dBFS
        audio_mix = audio_mix + change
        log.info("  Normalized to -6 dBFS (adjusted %.1f dB)", change)

    # Export audio mix as high-quality wav
    audio_mix_path = project.assembly_dir / "audio_mix.wav"
    audio_mix.export(str(audio_mix_path), format="wav", parameters=["-ar", "44100", "-ac", "2"])
    log.info("  Audio mix exported: %s", audio_mix_path)

    # ── Combine video + audio ──
    log.info("  Rendering final film...")
    audio_clip = AudioFileClip(str(audio_mix_path))
    if audio_clip.duration > final_video.duration:
        audio_clip = audio_clip.subclipped(0, final_video.duration)

    final_video = final_video.with_audio(audio_clip)

    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_video.write_videofile(
        str(final_path),
        fps=24,
        codec="libx264",
        audio_codec="aac",
        audio_bitrate="192k",
        preset="medium",
        threads=4,
        logger=None,
    )

    # Clean up
    final_video.close()
    audio_clip.close()
    for vc in video_clips:
        vc.close()

    file_size = final_path.stat().st_size
    log.info("  ════════════════════════════════════════════")
    log.info("  DONE! Final film: %s (%.1f MB)", final_path, file_size / 1024 / 1024)
    log.info("  ════════════════════════════════════════════")


# ── Main ──────────────────────────────────────────────────────────────


async def main():
    config, project, sg = load_project()

    log.info("═══════════════════════════════════════════════════════════")
    log.info("  THE EXPLORERS — Production Pipeline")
    log.info("  %d shots | %.0fs | %d dialogue lines", sg.total_shots, sg.total_duration_seconds, sg.total_dialogue_lines)
    log.info("═══════════════════════════════════════════════════════════")

    available = config.get_available_providers()
    log.info("Providers: visual=%s, tts=%s, music=%s, story=%s",
             available.get("visual"), available.get("tts"),
             available.get("music"), available.get("story"))

    # Run stages
    await generate_first_frames(config, project, sg)
    generate_videos(config, project, sg)
    generate_title_card(project)
    await generate_narration(config, project, sg)
    assemble_film(config, project, sg)

    log.info("═══════════════════════════════════════════════════════════")
    log.info("  Production complete!")
    log.info("═══════════════════════════════════════════════════════════")


if __name__ == "__main__":
    asyncio.run(main())
