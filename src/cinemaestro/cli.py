"""Command-line interface for Cinemaestro."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from cinemaestro import __version__

app = typer.Typer(
    name="cinemaestro",
    help="Automated short film production pipeline with AI.",
    no_args_is_help=True,
)
console = Console()

characters_app = typer.Typer(help="Manage virtual actor identities.")
shots_app = typer.Typer(help="Manage individual shots.")
config_app = typer.Typer(help="Configure API keys and providers.")
app.add_typer(characters_app, name="characters")
app.add_typer(shots_app, name="shots")
app.add_typer(config_app, name="config")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@app.command()
def new(
    concept: str = typer.Argument(..., help="Story concept or logline"),
    output: Path = typer.Option(Path("./project"), "--output", "-o", help="Project directory"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config file path"),
    duration: float = typer.Option(120.0, "--duration", "-d", help="Target duration in seconds"),
    genre: str = typer.Option("", "--genre", "-g", help="Film genre"),
    tone: str = typer.Option("", "--tone", help="Film tone"),
    style: str = typer.Option("photorealistic", "--style", "-s", help="Visual style"),
) -> None:
    """Create a new short film project from a story concept."""
    setup_logging()

    from cinemaestro.config import load_config
    from cinemaestro.core.project import Project

    config_obj = load_config(project_dir=output, config_file=config)
    project = Project(output)
    project.initialize()

    console.print(f"[bold green]Created project:[/] {output}")
    console.print(f"[dim]Concept:[/] {concept}")
    console.print(f"[dim]Duration:[/] {duration}s")

    # Run the story engine to generate the scene graph
    async def _generate() -> None:
        from cinemaestro.pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(config_obj, project)
        scene_graph = await orchestrator.run_story(concept)

        console.print(f"\n[bold]Generated screenplay:[/] {scene_graph.title}")
        console.print(f"  Scenes: {len(scene_graph.all_scenes())}")
        console.print(f"  Shots: {scene_graph.total_shots}")
        console.print(f"  Duration: ~{scene_graph.total_duration_seconds:.0f}s")
        console.print(f"\n[dim]Scene graph saved to:[/] {project.scene_graph_path}")
        console.print("[dim]Edit the YAML file to refine, then run:[/] cinemaestro run {output}")

    asyncio.run(_generate())


@app.command()
def run(
    project_dir: Path = typer.Argument(..., help="Project directory"),
    stage: str = typer.Option("all", "--stage", "-s", help="Pipeline stage to run"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume from checkpoint"),
) -> None:
    """Run the production pipeline on an existing project."""
    setup_logging()

    from cinemaestro.config import load_config
    from cinemaestro.core.events import EventBus, EventType, PipelineEvent
    from cinemaestro.core.project import Project
    from cinemaestro.pipeline.orchestrator import PipelineOrchestrator

    project = Project(project_dir)
    if not project.exists:
        console.print(f"[red]Project not found:[/] {project_dir}")
        raise typer.Exit(1)

    config = load_config(project_dir=project_dir)
    event_bus = EventBus()

    # Log events to console
    async def log_event(event: PipelineEvent) -> None:
        if event.event_type in (EventType.STAGE_START, EventType.STAGE_COMPLETE):
            console.print(f"[bold cyan][{event.stage}][/] {event.message}")
        elif event.event_type == EventType.SHOT_GENERATION_COMPLETE:
            console.print(f"  [green]✓[/] {event.message}")
        elif event.event_type in (EventType.SHOT_GENERATION_ERROR, EventType.PIPELINE_ERROR):
            console.print(f"  [red]✗[/] {event.message}")
        elif event.event_type == EventType.CONSISTENCY_CHECK_RESULT:
            console.print(f"  [yellow]⊙[/] {event.message}")

    event_bus.subscribe(log_event)

    orchestrator = PipelineOrchestrator(config, project, event_bus)

    async def _run() -> None:
        scene_graph = project.load_scene_graph()
        if scene_graph is None:
            console.print("[red]No scene graph found. Run 'cinemaestro new' first.[/]")
            raise typer.Exit(1)

        if stage == "all":
            result = await orchestrator.run_full_pipeline("", scene_graph=scene_graph)
            console.print(f"\n[bold green]Film complete![/] {result}")
        elif stage == "visual":
            await orchestrator.run_visual(scene_graph)
        elif stage == "audio":
            await orchestrator.run_audio(scene_graph)
        elif stage == "consistency":
            await orchestrator.run_consistency(scene_graph)
        elif stage == "assembly":
            result = await orchestrator.run_assembly(scene_graph)
            console.print(f"\n[bold green]Assembly complete![/] {result}")
        else:
            console.print(f"[red]Unknown stage:[/] {stage}")
            raise typer.Exit(1)

    asyncio.run(_run())


@characters_app.command("list")
def characters_list() -> None:
    """List all registered character identities."""
    setup_logging()

    from cinemaestro.config import load_config
    from cinemaestro.core.character import CharacterRegistry

    config = load_config()
    registry = CharacterRegistry(config.character_registry_dir.expanduser())
    characters = registry.list_all()

    if not characters:
        console.print("[dim]No characters registered yet.[/]")
        return

    table = Table(title="Character Registry")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Description")
    table.add_column("Refs", justify="right")
    table.add_column("Voice", justify="center")
    table.add_column("LoRA", justify="center")

    for char in characters:
        table.add_row(
            char.character_id,
            char.display_name,
            char.physical_description[:50] + "..." if len(char.physical_description) > 50 else char.physical_description,
            str(len(char.reference_images)),
            "✓" if char.elevenlabs_voice_id or char.voice_reference_audio else "✗",
            "✓" if char.lora_path else "✗",
        )

    console.print(table)


@characters_app.command("create")
def characters_create(
    character_id: str = typer.Option(..., "--id", help="Unique character identifier"),
    name: str = typer.Option(..., "--name", help="Display name"),
    description: str = typer.Option(..., "--description", "-d", help="Physical description"),
) -> None:
    """Create a new character identity."""
    setup_logging()

    from cinemaestro.config import CinemaestroConfig
    from cinemaestro.core.character import CharacterIdentity, CharacterRegistry

    config = CinemaestroConfig()
    registry = CharacterRegistry(config.character_registry_dir.expanduser())

    identity = CharacterIdentity(
        character_id=character_id,
        display_name=name,
        physical_description=description,
    )
    registry.register(identity)
    console.print(f"[green]Created character:[/] {character_id} ({name})")


@shots_app.command("list")
def shots_list(
    project_dir: Path = typer.Argument(..., help="Project directory"),
) -> None:
    """List all shots in a project."""
    setup_logging()

    from cinemaestro.core.project import Project
    from cinemaestro.pipeline.state import PipelineState

    project = Project(project_dir)
    scene_graph = project.load_scene_graph()
    state = PipelineState.load(project.pipeline_state_path)

    if not scene_graph:
        console.print("[red]No scene graph found.[/]")
        raise typer.Exit(1)

    table = Table(title=f"Shots — {scene_graph.title}")
    table.add_column("#", style="dim")
    table.add_column("Shot ID", style="cyan")
    table.add_column("Type")
    table.add_column("Duration", justify="right")
    table.add_column("Characters")
    table.add_column("Status", justify="center")

    for i, shot in enumerate(scene_graph.all_shots(), 1):
        shot_state = state.shots.get(shot.shot_id)
        status = shot_state.status.value if shot_state else "pending"
        status_style = {
            "completed": "[green]✓[/]",
            "failed": "[red]✗[/]",
            "in_progress": "[yellow]⟳[/]",
            "pending": "[dim]○[/]",
        }.get(status, status)

        table.add_row(
            str(i),
            shot.shot_id,
            shot.shot_type.value,
            f"{shot.duration_seconds:.1f}s",
            ", ".join(shot.characters_present) or "—",
            status_style,
        )

    console.print(table)
    console.print(f"\n[dim]Total: {scene_graph.total_shots} shots, ~{scene_graph.total_duration_seconds:.0f}s[/]")


@shots_app.command("regenerate")
def shots_regenerate(
    project_dir: Path = typer.Argument(..., help="Project directory"),
    shot_ids: list[str] = typer.Option(..., "--shot-id", help="Shot IDs to regenerate"),
) -> None:
    """Regenerate specific shots."""
    setup_logging()

    from cinemaestro.config import load_config
    from cinemaestro.core.project import Project
    from cinemaestro.pipeline.orchestrator import PipelineOrchestrator

    config = load_config(project_dir=project_dir)
    project = Project(project_dir)
    orchestrator = PipelineOrchestrator(config, project)

    async def _regen() -> None:
        await orchestrator.regenerate_shots(shot_ids)
        console.print(f"[green]Regenerated {len(shot_ids)} shots.[/]")

    asyncio.run(_regen())


@config_app.command("show")
def config_show(
    project_dir: Path | None = typer.Option(None, "--project", "-p", help="Project directory"),
) -> None:
    """Show current configuration and detected API keys."""
    setup_logging("WARNING")

    from cinemaestro.config import load_config

    config = load_config(project_dir=project_dir)
    summary = config.summary()

    console.print("\n[bold]Cinemaestro Configuration[/]\n")

    # API Keys
    table = Table(title="API Keys", show_header=True)
    table.add_column("Provider", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Key", style="dim")

    for provider, masked_key in summary["api_keys"].items():
        is_set = masked_key != "(not set)"
        status = "[green]OK[/]" if is_set else "[dim]--[/]"
        table.add_row(provider, status, masked_key)

    console.print(table)

    # Pipeline Configuration
    console.print(f"\n[bold]Story Engine[/]")
    console.print(f"  Provider: {summary['story']['provider']}  Model: {summary['story']['model']}")
    console.print(f"  API Key:  {summary['story']['api_key']}")

    console.print(f"\n[bold]Visual Generation[/]")
    console.print(f"  Default:  {summary['visual']['default_provider']}")
    console.print(f"  Fallback: {summary['visual']['fallback_provider']}")
    console.print(f"  Strategy: {summary['visual']['consistency_strategy']}")
    avail = summary["visual"]["available"]
    console.print(f"  Available: {', '.join(avail) if avail else '[dim]none configured[/]'}")

    console.print(f"\n[bold]Audio[/]")
    console.print(f"  TTS:   {summary['audio']['tts_provider']}")
    console.print(f"  Music: {summary['audio']['music_provider']}")
    avail_tts = summary["audio"]["available_tts"]
    avail_music = summary["audio"]["available_music"]
    console.print(f"  Available TTS:   {', '.join(avail_tts) if avail_tts else '[dim]none[/]'}")
    console.print(f"  Available Music: {', '.join(avail_music) if avail_music else '[dim]none[/]'}")

    console.print(f"\n[bold]Consistency[/]")
    console.print(f"  Enabled:    {summary['consistency']['enabled']}")
    console.print(f"  Threshold:  {summary['consistency']['threshold']}")
    console.print(f"  Auto-repair: {summary['consistency']['auto_repair']}")

    console.print(f"\n[bold]Budget Limit:[/] {summary['budget_limit']}")
    console.print()


@config_app.command("check")
def config_check(
    project_dir: Path | None = typer.Option(None, "--project", "-p", help="Project directory"),
) -> None:
    """Validate API keys by making lightweight test calls."""
    setup_logging("WARNING")

    import asyncio

    from cinemaestro.config import load_config

    config = load_config(project_dir=project_dir)

    async def _check() -> None:
        import httpx

        checks: list[tuple[str, str, bool, str]] = []

        # Anthropic
        if config.anthropic_api_key:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        "https://api.anthropic.com/v1/models",
                        headers={
                            "x-api-key": config.anthropic_api_key,
                            "anthropic-version": "2023-06-01",
                        },
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        checks.append(("Anthropic", "Story Engine", True, ""))
                    else:
                        checks.append(("Anthropic", "Story Engine", False, f"HTTP {resp.status_code}"))
            except Exception as e:
                checks.append(("Anthropic", "Story Engine", False, str(e)))
        else:
            checks.append(("Anthropic", "Story Engine", False, "No API key"))

        # OpenAI
        if config.openai_api_key:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        "https://api.openai.com/v1/models",
                        headers={"Authorization": f"Bearer {config.openai_api_key}"},
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        checks.append(("OpenAI", "Story Engine (alt)", True, ""))
                    else:
                        checks.append(("OpenAI", "Story Engine (alt)", False, f"HTTP {resp.status_code}"))
            except Exception as e:
                checks.append(("OpenAI", "Story Engine (alt)", False, str(e)))

        # Runway
        if config.runway_api_key:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        "https://api.dev.runwayml.com/v1/tasks?limit=1",
                        headers={
                            "Authorization": f"Bearer {config.runway_api_key}",
                            "X-Runway-Version": "2024-11-06",
                        },
                        timeout=10,
                    )
                    # 200 or 404 both mean auth worked
                    if resp.status_code in (200, 404):
                        checks.append(("Runway", "Visual (Gen-4)", True, ""))
                    else:
                        checks.append(("Runway", "Visual (Gen-4)", False, f"HTTP {resp.status_code}"))
            except Exception as e:
                checks.append(("Runway", "Visual (Gen-4)", False, str(e)))

        # ElevenLabs
        if config.elevenlabs_api_key:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        "https://api.elevenlabs.io/v1/voices",
                        headers={"xi-api-key": config.elevenlabs_api_key},
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        voices = resp.json().get("voices", [])
                        checks.append(("ElevenLabs", "TTS", True, f"{len(voices)} voices"))
                    else:
                        checks.append(("ElevenLabs", "TTS", False, f"HTTP {resp.status_code}"))
            except Exception as e:
                checks.append(("ElevenLabs", "TTS", False, str(e)))

        # fal.ai
        if config.fal_key:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        "https://queue.fal.run/fal-ai/flux/dev/requests",
                        headers={"Authorization": f"Key {config.fal_key}"},
                        timeout=10,
                    )
                    # 200 or 404 or 405 all mean auth worked
                    if resp.status_code in (200, 404, 405, 422):
                        checks.append(("fal.ai", "Visual (Flux/Kling)", True, ""))
                    elif resp.status_code == 401:
                        checks.append(("fal.ai", "Visual (Flux/Kling)", False, "invalid key"))
                    else:
                        checks.append(("fal.ai", "Visual (Flux/Kling)", False, f"HTTP {resp.status_code}"))
            except Exception as e:
                checks.append(("fal.ai", "Visual (Flux/Kling)", False, str(e)))

        # Replicate
        if config.replicate_api_token:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        "https://api.replicate.com/v1/account",
                        headers={"Authorization": f"Bearer {config.replicate_api_token}"},
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        username = resp.json().get("username", "")
                        checks.append(("Replicate", "Visual (multi)", True, f"user: {username}"))
                    else:
                        checks.append(("Replicate", "Visual (multi)", False, f"HTTP {resp.status_code}"))
            except Exception as e:
                checks.append(("Replicate", "Visual (multi)", False, str(e)))

        # Stability AI
        if config.stability_api_key:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        "https://api.stability.ai/v1/user/balance",
                        headers={"Authorization": f"Bearer {config.stability_api_key}"},
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        credits = resp.json().get("credits", 0)
                        checks.append(("Stability AI", "Music/SFX", True, f"{credits:.0f} credits"))
                    else:
                        checks.append(("Stability AI", "Music/SFX", False, f"HTTP {resp.status_code}"))
            except Exception as e:
                checks.append(("Stability AI", "Music/SFX", False, str(e)))

        # Kling
        if config.kling_api_key:
            checks.append(("Kling", "Visual (Elements)", True, "key present (no test endpoint)"))

        # ComfyUI (local)
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{config.comfyui_url}/system_stats",
                    timeout=5,
                )
                if resp.status_code == 200:
                    checks.append(("ComfyUI", "Visual (local)", True, "running"))
                else:
                    checks.append(("ComfyUI", "Visual (local)", False, "not responding"))
        except Exception:
            checks.append(("ComfyUI", "Visual (local)", False, "not running"))

        # Display results
        table = Table(title="Provider Health Check")
        table.add_column("Provider", style="cyan")
        table.add_column("Role")
        table.add_column("Status", justify="center")
        table.add_column("Details", style="dim")

        passed = 0
        for name, role, ok, detail in checks:
            status = "[green]PASS[/]" if ok else "[red]FAIL[/]"
            table.add_row(name, role, status, detail)
            if ok:
                passed += 1

        console.print(table)
        console.print(
            f"\n[bold]{passed}/{len(checks)}[/] providers available"
        )

        # Readiness assessment
        has_story = any(name == "Anthropic" and ok for name, _, ok, _ in checks)
        has_visual = any(role.startswith("Visual") and ok for _, role, ok, _ in checks)
        has_tts = any(role == "TTS" and ok for _, role, ok, _ in checks)

        console.print()
        if has_story and has_visual:
            console.print("[bold green]Ready to produce films![/]")
            if not has_tts:
                console.print("[yellow]Note: No TTS provider — films will be silent.[/]")
        elif has_story:
            console.print("[yellow]Can generate screenplays, but no visual provider configured.[/]")
        else:
            console.print("[red]No story provider configured. Set ANTHROPIC_API_KEY to get started.[/]")

    asyncio.run(_check())


@config_app.command("init")
def config_init(
    location: str = typer.Option(
        "user",
        "--location", "-l",
        help="Where to save config: 'user' (~/.cinemaestro/config.toml) or 'project' (./cinemaestro.toml)",
    ),
    project_dir: Path | None = typer.Option(None, "--project", "-p", help="Project directory"),
) -> None:
    """Interactive setup — configure API keys and save to config file."""
    setup_logging("WARNING")

    console.print("\n[bold]Cinemaestro Configuration Setup[/]\n")
    console.print("Enter your API keys below. Press Enter to skip any provider.\n")

    keys: dict[str, str] = {}

    prompts = [
        ("ANTHROPIC_API_KEY", "Anthropic (Claude)", "Required for story generation"),
        ("OPENAI_API_KEY", "OpenAI", "Alternative story engine / Sora"),
        ("RUNWAY_API_KEY", "Runway", "Video generation (Gen-4 / Gen-4.5)"),
        ("KLING_API_KEY", "Kling", "Video generation with character Elements"),
        ("FAL_KEY", "fal.ai", "Image/video generation gateway (Flux, Kling)"),
        ("REPLICATE_API_TOKEN", "Replicate", "Multi-model gateway"),
        ("ELEVENLABS_API_KEY", "ElevenLabs", "Text-to-speech (voice cloning)"),
        ("STABILITY_API_KEY", "Stability AI", "Music and sound effect generation"),
    ]

    for env_name, display_name, description in prompts:
        existing = os.environ.get(env_name, "")
        if existing:
            masked = existing[:4] + "..." + existing[-4:] if len(existing) > 8 else "****"
            console.print(f"  [cyan]{display_name}[/] ({description})")
            console.print(f"    [dim]Current: {masked} (from environment)[/]")
            answer = console.input(f"    Keep existing? [Y/n]: ").strip().lower()
            if answer in ("", "y", "yes"):
                keys[env_name] = existing
                continue

        console.print(f"  [cyan]{display_name}[/] ({description})")
        key = console.input(f"    API Key: ").strip()
        if key:
            keys[env_name] = key
        console.print()

    # ComfyUI
    console.print("  [cyan]ComfyUI[/] (Local image/video generation)")
    comfyui_url = console.input(f"    URL [http://127.0.0.1:8188]: ").strip()
    if not comfyui_url:
        comfyui_url = "http://127.0.0.1:8188"

    # Budget
    console.print("\n  [cyan]Budget[/]")
    budget_str = console.input(f"    Max API spend per project [$50.00]: ").strip()
    budget = 50.0
    if budget_str:
        try:
            budget = float(budget_str.replace("$", ""))
        except ValueError:
            pass

    # Determine output path
    if location == "user":
        config_dir = Path("~/.cinemaestro").expanduser()
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.toml"
        env_path = config_dir / ".env"
    else:
        config_path = (project_dir or Path(".")) / "cinemaestro.toml"
        env_path = (project_dir or Path(".")) / ".env"

    # Write .env file with API keys
    env_lines = [
        "# Cinemaestro API Keys",
        f"# Generated by 'cinemaestro config init'",
        "",
    ]
    for env_name, display_name, _ in prompts:
        if env_name in keys:
            env_lines.append(f"{env_name}={keys[env_name]}")
        else:
            env_lines.append(f"# {env_name}=")

    env_path.write_text("\n".join(env_lines) + "\n")

    # Write TOML config (non-secret settings)
    toml_lines = [
        "# Cinemaestro configuration",
        "",
        "[story]",
    ]

    if "ANTHROPIC_API_KEY" in keys:
        toml_lines.append('llm_provider = "anthropic"')
        toml_lines.append('model = "claude-sonnet-4-20250514"')
    elif "OPENAI_API_KEY" in keys:
        toml_lines.append('llm_provider = "openai"')
        toml_lines.append('model = "gpt-4o"')

    # Determine best visual provider
    toml_lines.append("")
    toml_lines.append("[visual]")

    visual_providers = []
    if "KLING_API_KEY" in keys:
        visual_providers.append("kling")
    if "RUNWAY_API_KEY" in keys:
        visual_providers.append("runway")
    if "FAL_KEY" in keys:
        visual_providers.append("fal")
    if "REPLICATE_API_TOKEN" in keys:
        visual_providers.append("replicate")

    if visual_providers:
        toml_lines.append(f'default_provider = "{visual_providers[0]}"')
        if len(visual_providers) > 1:
            toml_lines.append(f'fallback_provider = "{visual_providers[1]}"')
        else:
            toml_lines.append('fallback_provider = "comfyui"')

        # If we have native character ref providers, prefer native strategy
        if "kling" in visual_providers or "runway" in visual_providers:
            toml_lines.append('character_consistency_strategy = "native"')
        else:
            toml_lines.append('character_consistency_strategy = "first_frame"')
    else:
        toml_lines.append('default_provider = "comfyui"')
        toml_lines.append('fallback_provider = "comfyui"')
        toml_lines.append('character_consistency_strategy = "first_frame"')

    # Audio
    toml_lines.append("")
    toml_lines.append("[audio.tts]")
    if "ELEVENLABS_API_KEY" in keys:
        toml_lines.append('provider = "elevenlabs"')
    else:
        toml_lines.append('provider = "xtts"')

    toml_lines.append("")
    toml_lines.append("[audio.music]")
    if "STABILITY_API_KEY" in keys:
        toml_lines.append('provider = "stable_audio"')
    else:
        toml_lines.append('provider = "audiocraft"')

    # General
    toml_lines.append("")
    toml_lines.append(f'comfyui_url = "{comfyui_url}"')
    toml_lines.append(f"max_budget_usd = {budget}")

    config_path.write_text("\n".join(toml_lines) + "\n")

    console.print(f"\n[bold green]Configuration saved![/]")
    console.print(f"  Config: {config_path}")
    console.print(f"  API Keys: {env_path}")

    # Set up symlink for .env discovery if saved to user dir
    if location == "user":
        console.print(f"\n[dim]To use these keys, either:[/]")
        console.print(f"  1. Source the env file: [cyan]source {env_path}[/]")
        console.print(f"  2. Or symlink it: [cyan]ln -s {env_path} .env[/]")
        console.print(f"  3. Or export keys in your shell profile[/]")

    console.print(f"\n[dim]Run 'cinemaestro config check' to verify your setup.[/]")


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Config key (e.g., ANTHROPIC_API_KEY, RUNWAY_API_KEY)"),
    value: str = typer.Argument(..., help="Value to set"),
    location: str = typer.Option(
        "user", "--location", "-l",
        help="'user' or 'project'",
    ),
) -> None:
    """Set a single configuration value."""
    setup_logging("WARNING")

    if location == "user":
        env_path = Path("~/.cinemaestro/.env").expanduser()
    else:
        env_path = Path(".env")

    # Read existing
    existing_lines: list[str] = []
    if env_path.exists():
        existing_lines = env_path.read_text().splitlines()

    # Update or append
    found = False
    for i, line in enumerate(existing_lines):
        stripped = line.strip()
        if stripped.startswith(f"{key}=") or stripped.startswith(f"# {key}="):
            existing_lines[i] = f"{key}={value}"
            found = True
            break

    if not found:
        existing_lines.append(f"{key}={value}")

    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(existing_lines) + "\n")

    masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "****"
    console.print(f"[green]Set[/] {key} = {masked} in {env_path}")


@app.command()
def version() -> None:
    """Show version information."""
    console.print(f"Cinemaestro v{__version__}")


@app.callback()
def main() -> None:
    """Cinemaestro: Automated short film production pipeline."""
    pass
