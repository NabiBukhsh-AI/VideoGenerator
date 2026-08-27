"""Command-line interface.

New in this version - the original was Streamlit-only, which made it impossible to
script, batch or run on a server. Everything the UI can do is available here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from . import __version__
from .config import reload_settings
from .errors import VideoGenError
from .logging import ProgressReporter, configure, get_logger
from .models import AspectRatio, KenBurnsKind, RenderResult, TransitionKind, VideoSpec
from .pipeline import GenerationRequest, Pipeline
from .providers import KNOWN, build_image, build_llm, build_tts
from .render import ffmpeg
from .render.fonts import set_font_override
from .utils import format_duration

app = typer.Typer(
    name="videogen",
    help="Turn a PDF, article or block of text into a narrated short-form video.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()
log = get_logger("cli")


class RichProgress(ProgressReporter):
    """Progress sink backed by a Rich progress bar."""

    def __init__(self, progress: Progress) -> None:
        super().__init__(log)
        self._progress = progress
        self._task = progress.add_task("Starting", total=1.0)

    def stage(self, name: str, detail: str = "") -> None:
        self._progress.update(
            self._task, description=f"{name}{f' - {detail}' if detail else ''}", completed=0.0
        )

    def update(self, fraction: float, detail: str = "") -> None:
        self._progress.update(self._task, completed=min(1.0, max(0.0, fraction)))
        if detail:
            self._progress.update(self._task, description=self._current_description(detail))

    def _current_description(self, detail: str) -> str:
        description = self._progress.tasks[self._task].description
        base = description.split(" - ")[0]
        return f"{base} - {detail}"

    def warn(self, message: str) -> None:
        console.print(f"[yellow]![/yellow] {message}")


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"videogen {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True,
        help="Show the version and exit."
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="DEBUG, INFO, WARNING or ERROR."),
) -> None:
    configure(log_level, force=True)


@app.command()
def generate(
    source: str = typer.Argument(
        ..., help="A PDF, .txt/.md file, an http(s) URL, or raw text in quotes."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Directory for the generated project."
    ),
    aspect: AspectRatio = typer.Option(AspectRatio.VERTICAL, "--aspect", "-a",
                                       help="Canvas shape."),
    voice: str | None = typer.Option(None, "--voice", help="TTS voice name or ID."),
    music: Path | None = typer.Option(
        None, "--music", "-m", exists=True, dir_okay=False,
        help="Background music track to duck under the narration."
    ),
    music_gain: float = typer.Option(-18.0, "--music-gain", help="Music bed level in dB."),
    tone: str | None = typer.Option(None, "--tone", help="e.g. 'urgent', 'wry', 'warm'."),
    language: str | None = typer.Option(None, "--language", help="Narration language."),
    focus: str | None = typer.Option(None, "--focus", help="Angle to emphasise."),
    scenes: int | None = typer.Option(None, "--scenes", "-n", min=1, max=30,
                                         help="Maximum number of scenes."),
    fps: int = typer.Option(30, "--fps", min=1, max=120),
    transition: TransitionKind = typer.Option(TransitionKind.CROSSFADE, "--transition"),
    motion: KenBurnsKind = typer.Option(KenBurnsKind.AUTO, "--motion",
                                        help="Ken Burns movement style."),
    no_captions: bool = typer.Option(False, "--no-captions", help="Skip burned-in captions."),
    caption_color: str = typer.Option("#FFFFFF", "--caption-color"),
    highlight_color: str = typer.Option("#FFD400", "--highlight-color"),
    font: Path | None = typer.Option(None, "--font", exists=True, dir_okay=False,
                                        help="TTF to use for captions."),
    llm: str | None = typer.Option(None, "--llm", help=f"One of: {', '.join(KNOWN['llm'])}."),
    image: str | None = typer.Option(None, "--image", help=f"One of: {', '.join(KNOWN['image'])}."),
    tts: str | None = typer.Option(None, "--tts", help=f"One of: {', '.join(KNOWN['tts'])}."),
    crf: int = typer.Option(20, "--crf", min=0, max=51, help="H.264 quality; lower is better."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Use offline providers only - no API calls, no cost."
    ),
    script_only: bool = typer.Option(
        False, "--script-only", help="Write the script and stop before generating assets."
    ),
    resume: Path | None = typer.Option(
        None, "--resume", help="Resume an existing project directory."
    ),
) -> None:
    """Generate a video from SOURCE."""
    settings = reload_settings()
    if scenes is not None:
        settings = settings.model_copy(
            update={"max_scenes": scenes, "min_scenes": min(settings.min_scenes, scenes)}
        )
    if font is not None:
        set_font_override(font)

    spec = VideoSpec(
        aspect=aspect,
        fps=fps,
        transition=transition,
        ken_burns=motion,
        captions_enabled=not no_captions,
        caption_color=caption_color,
        caption_highlight_color=highlight_color,
        music_enabled=music is not None,
        music_gain_db=music_gain,
        crf=crf,
    )

    request = GenerationRequest(
        source=source,
        spec=spec,
        voice=voice,
        music_path=music,
        tone=tone,
        language=language,
        focus=focus,
        llm_override=llm,
        image_override=image,
        tts_override=tts,
        output_dir=output,
        project_dir=resume,
        dry_run=dry_run,
        script_only=script_only,
    )

    if dry_run:
        console.print("[cyan]Dry run:[/cyan] offline providers, no API calls, no spend.\n")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress_bar:
            reporter = RichProgress(progress_bar)
            result = Pipeline(settings, reporter).run(request)
    except VideoGenError as exc:
        _fail(exc)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow] Re-run with --resume to continue.")
        raise typer.Exit(130) from None

    _report(result, script_only=script_only)


@app.command()
def doctor() -> None:
    """Check credentials, dependencies and ffmpeg."""
    settings = reload_settings()
    console.print(Panel.fit(f"videogen {__version__}", style="bold cyan"))

    table = Table("Component", "Status", "Detail", show_lines=False)

    ffmpeg_path = ffmpeg.find_ffmpeg(settings)
    table.add_row(
        "ffmpeg",
        "[green]ok[/green]" if ffmpeg_path else "[red]missing[/red]",
        ffmpeg.version(settings) if ffmpeg_path else "Required to render video",
    )

    for module, extra in [
        ("openai", "openai"), ("anthropic", "anthropic"), ("elevenlabs", "elevenlabs"),
        ("fitz", "pdf"), ("bs4", "web"), ("streamlit", "ui"),
    ]:
        try:
            __import__(module)
            table.add_row(module, "[green]ok[/green]", "")
        except ImportError:
            table.add_row(module, "[yellow]absent[/yellow]", f"pip install 'videogen[{extra}]'")

    for key, label in [
        ("openai_api_key", "OPENAI_API_KEY"),
        ("anthropic_api_key", "ANTHROPIC_API_KEY"),
        ("elevenlabs_api_key", "ELEVENLABS_API_KEY"),
    ]:
        present = bool(settings.secret(key))
        table.add_row(
            label,
            "[green]set[/green]" if present else "[dim]unset[/dim]",
            "" if present else "Add it to .env to enable the matching provider",
        )

    console.print(table)

    available = settings.available_providers()
    console.print("\n[bold]Usable providers[/bold]")
    for kind, names in available.items():
        console.print(f"  {kind:<6} {', '.join(names)}")

    if not ffmpeg_path:
        console.print(
            "\n[yellow]No ffmpeg.[/yellow] You can still use --script-only; rendering needs it."
        )
    console.print("\nTry: [cyan]videogen generate --dry-run \"your text here\"[/cyan]")


@app.command()
def providers() -> None:
    """List providers and whether each one can run right now."""
    settings = reload_settings()
    table = Table("Kind", "Provider", "Status", "Note")

    builders = {"llm": build_llm, "image": build_image, "tts": build_tts}
    for kind, names in KNOWN.items():
        for name in names:
            try:
                provider = builders[kind](settings, name)
                provider.health_check()
                status, note = "[green]ready[/green]", "offline" if provider.offline else ""
            except VideoGenError as exc:
                status, note = "[yellow]unavailable[/yellow]", exc.message
            except Exception as exc:
                status, note = "[red]error[/red]", str(exc)[:60]
            table.add_row(kind, name, status, note)

    console.print(table)


@app.command()
def ui(
    port: int = typer.Option(8501, "--port", "-p"),
    host: str = typer.Option("localhost", "--host"),
) -> None:
    """Launch the Streamlit interface."""
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError:
        console.print("[red]Streamlit is not installed.[/red] Run: pip install 'videogen[ui]'")
        raise typer.Exit(1) from None

    app_path = Path(__file__).parent / "ui" / "streamlit_app.py"
    sys.argv = [
        "streamlit", "run", str(app_path),
        "--server.port", str(port),
        "--server.address", host,
    ]
    console.print(f"Starting the UI at [cyan]http://{host}:{port}[/cyan]")
    sys.exit(streamlit_cli.main())


@app.command()
def init() -> None:
    """Write a starter .env file."""
    from .config import dotenv_path

    target = dotenv_path()
    if target.exists():
        console.print(f"[yellow]{target} already exists.[/yellow] Leaving it alone.")
        raise typer.Exit(1)

    example = Path(__file__).parent.parent / ".env.example"
    if example.exists():
        target.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        target.write_text(
            "OPENAI_API_KEY=\nANTHROPIC_API_KEY=\nELEVENLABS_API_KEY=\n", encoding="utf-8"
        )

    console.print(f"[green]Wrote {target}[/green] - add your keys, then run: videogen doctor")
    console.print("[dim]It is gitignored. Never commit real keys.[/dim]")


def _report(result: RenderResult, *, script_only: bool) -> None:
    """Print a summary of what the run produced."""
    script = result.script
    console.print()
    console.print(Panel.fit(script.title, style="bold green"))

    table = Table(show_header=False, box=None, padding=(0, 2))
    if script_only:
        table.add_row("Project", str(result.project_dir))
        table.add_row("Estimated runtime", format_duration(result.duration_ms))
    else:
        table.add_row("Video", str(result.video_path))
        table.add_row("Duration", format_duration(result.duration_ms))
        if result.subtitle_path:
            table.add_row("Subtitles", str(result.subtitle_path))
        if result.thumbnail_path:
            table.add_row("Thumbnail", str(result.thumbnail_path))
    if result.metadata_path:
        table.add_row("Metadata", str(result.metadata_path))
    table.add_row("Scenes", str(len(script.scenes)))
    if script.hashtags:
        table.add_row("Hashtags", " ".join(f"#{tag}" for tag in script.hashtags))

    usage = result.usage
    if usage.estimated_cost_usd > 0:
        table.add_row(
            "Estimated cost",
            f"${usage.estimated_cost_usd:.3f} "
            f"({usage.images_generated} images, {usage.tts_characters} TTS chars)",
        )
    console.print(table)
    console.print()


def _fail(exc: VideoGenError) -> None:
    console.print(f"\n[red]Error:[/red] {exc.message}")
    if exc.hint:
        console.print(f"[yellow]Hint:[/yellow] {exc.hint}")
    raise typer.Exit(1)


if __name__ == "__main__":  # pragma: no cover
    app()
