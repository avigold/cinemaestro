# Cinemaestro

Automated short film production pipeline with AI-powered visual generation, audio synthesis, and character consistency.

## Quick Start

```bash
pip install -e ".[cloud]"

# Create a new film project
cinemaestro new "A lonely astronaut discovers a garden in alien wreckage" \
  --duration 60 --genre sci-fi --style "photorealistic cinematic"

# Run the full pipeline
cinemaestro run ./project
```

## Architecture

Cinemaestro transforms a story concept into a finished short film through 7 pipeline stages:

1. **Story Engine** — LLM generates a structured screenplay (SceneGraph)
2. **Character Forge** — Creates persistent virtual actor identities with reference images and voice profiles
3. **Visual Generation** — Generates video clips via Runway, Kling, fal.ai, ComfyUI, or Replicate
4. **Audio Forge** — TTS dialogue (ElevenLabs, XTTS), music generation, sound effects
5. **Consistency Pass** — Verifies and repairs character identity across all shots
6. **Assembly** — Builds timeline, mixes audio, assembles final video
7. **Export** — Renders to deliverable format

The central IR is the **SceneGraph** — a Pydantic model serializable to YAML that captures acts, scenes, shots, characters, locations, dialogue, camera movements, lighting, and transitions. The story engine produces it; every downstream stage consumes it.

## Visual Providers

| Provider | Type | Models |
|----------|------|--------|
| **fal.ai** | Cloud gateway | Flux (images), Kling v2 (video) |
| **Runway** | Cloud | Gen-4 Turbo |
| **Kling** | Cloud | Kling v2 with character Elements |
| **Replicate** | Cloud gateway | Flux, various |
| **ComfyUI** | Local | Any Stable Diffusion / AnimateDiff workflow |

## Character Consistency

Three-layer defense:
- **Prevention**: Identity-conditioned generation (LoRA, IP-Adapter, native character references, first-frame strategy)
- **Detection**: Face embedding verification via InsightFace/ArcFace
- **Repair**: Targeted face-swap correction via ReActor on failing frames + GFPGAN enhancement

## Standalone Video Generators

The `projects/lyric_video/` directory contains self-contained video generators that don't require cloud APIs:

### Kinetic Typography (`lyric_video.py`)

Gritty, aggressive lyric video generator with VHS distortion, film grain, screen corruption, particle systems, and 19 text animation effects. Se7en meets Fight Club.

```bash
python projects/lyric_video/lyric_video.py \
  --audio song.mp3 --lyrics lyrics.txt --output video.mp4
```

### 8-bit NES (`nes_video.py`)

Retro NES-style pixel art music video — renders at native 256x240 resolution, scaled to 1080p with nearest-neighbor for crisp pixels. Features hand-drawn sprite characters, 8 tile-based backgrounds, NES dialogue boxes, and scene choreography synced to song structure.

```bash
python projects/lyric_video/nes_video.py \
  --audio song.mp3 --lyrics lyrics.txt --output nes_video.mp4
```

Both use Whisper for automatic vocal region detection — lyrics only display during sung sections, with instrumental breaks (guitar solos, etc.) left text-free.

## Configuration

```bash
# Interactive setup
cinemaestro config init

# Or set keys individually
cinemaestro config set FAL_KEY your-key-here
cinemaestro config set ANTHROPIC_API_KEY your-key-here

# Verify providers
cinemaestro config check
```

API keys can be set via environment variables, `.env` files, or `~/.cinemaestro/config.toml`.

## Requirements

- Python 3.11+
- Core: pydantic, httpx, moviepy 2.x, pydub, insightface, typer, rich
- Cloud extras: anthropic, elevenlabs, openai, replicate, fal-client
- Local extras: torch, diffusers, TTS (Coqui XTTS)
- Standalone video generators: pillow, numpy, openai-whisper
