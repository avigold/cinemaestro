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
3. **Visual Generation** — Generates video clips via Runway, Kling, ComfyUI, or other providers
4. **Audio Forge** — TTS dialogue, music generation, sound effects
5. **Consistency Pass** — Verifies and repairs character identity across all shots
6. **Assembly** — Builds timeline, mixes audio, assembles final video
7. **Export** — Renders to deliverable format

## Character Consistency

Three-layer defense:
- **Prevention**: Identity-conditioned generation (LoRA, IP-Adapter, native character references)
- **Detection**: Face embedding verification via InsightFace/ArcFace
- **Repair**: Targeted face-swap correction via ReActor on failing frames
