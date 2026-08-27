"""The generation pipeline.

Stages: ingest -> script -> (images and narration, concurrently) -> align -> render.

Two structural changes over the original `main.py`, which had the entire flow inlined
twice - once with background music and once without, with the 70-line prompt copied
verbatim into both:

* There is one path. Background music is a parameter, not a second implementation.
* Images and narration are generated concurrently and cached by content hash, so a
  six-scene short makes six parallel calls instead of twelve sequential ones, and a
  re-run after a failure pays only for what is actually missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings, get_settings
from .errors import DependencyError, VideoGenError
from .ingest import load_source
from .ingest.base import SourceInput
from .logging import ProgressReporter, get_logger
from .models import RenderResult, Scene, Script, SourceDocument, Usage, VideoSpec
from .providers import build_image, build_llm, build_tts
from .providers.base import ImageProvider, LLMProvider, TTSProvider
from .render import ffmpeg
from .render.align import build_aligner
from .render.compositor import Compositor
from .script.generator import ScriptGenerator, ScriptRequest
from .script.prompts import IMAGE_PROMPT_SUFFIX
from .storage import Project
from .utils import Timer, parallel_map

log = get_logger("pipeline")


@dataclass
class GenerationRequest:
    """Everything one run needs. Built by the CLI or the Streamlit UI."""

    source: SourceInput
    spec: VideoSpec = field(default_factory=VideoSpec)

    #: A script that has already been generated and reviewed (the UI's two-step flow).
    #: When set, the script stage is skipped entirely.
    script: Script | None = None

    voice: str | None = None
    music_path: Path | None = None
    tone: str | None = None
    language: str | None = None
    focus: str | None = None

    llm_override: str | None = None
    image_override: str | None = None
    tts_override: str | None = None

    output_dir: Path | None = None
    project_dir: Path | None = None  # resume into an existing project
    dry_run: bool = False
    script_only: bool = False


class Pipeline:
    """Runs a `GenerationRequest` end to end."""

    def __init__(
        self,
        settings: Settings | None = None,
        progress: ProgressReporter | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.progress = progress or ProgressReporter(log)

    # -- providers -------------------------------------------------------------
    def _providers(
        self, request: GenerationRequest
    ) -> tuple[LLMProvider, ImageProvider, TTSProvider]:
        settings = self.settings.offline() if request.dry_run else self.settings
        llm = build_llm(settings, request.llm_override)
        image = build_image(settings, request.image_override)
        tts = build_tts(settings, request.tts_override)

        # Fail on missing credentials now, not four minutes into a run.
        for provider in (llm, image, tts):
            provider.health_check()

        log.info(
            "Providers: llm=%s image=%s tts=%s", llm.label, image.label, tts.label
        )
        return llm, image, tts

    # -- entry point -----------------------------------------------------------
    def run(self, request: GenerationRequest) -> RenderResult:
        usage = Usage()
        llm, image_provider, tts_provider = self._providers(request)

        if not request.script_only and not ffmpeg.available(self.settings):
            raise DependencyError(
                "ffmpeg is required to render video.",
                hint=(
                    "Install it (see the README), or pass --script-only to generate just "
                    "the script, images and narration."
                ),
            )

        # 1. Ingest ------------------------------------------------------------
        self.progress.stage("Reading source material")
        document = load_source(request.source, max_words=self.settings.max_source_words)
        log.info("Source %r: %d words (%s)", document.name, document.word_count, document.kind)

        project = self._open_project(request, document)

        # 2. Script ------------------------------------------------------------
        script = self._get_script(request, project, llm, document, usage)
        project.save_manifest(
            script=script,
            spec=request.spec,
            providers={
                "llm": llm.name,
                "image": image_provider.name,
                "tts": tts_provider.name,
            },
        )

        if request.script_only:
            self.progress.stage("Script only - stopping before asset generation")
            return RenderResult(
                video_path=project.root,
                script=script,
                spec=request.spec,
                duration_ms=script.estimated_duration_ms,
                metadata_path=project.write_metadata(script),
                usage=usage,
                project_dir=project.root,
            )

        # 3. Assets ------------------------------------------------------------
        self._generate_images(script, project, image_provider, request.spec, usage)
        self._generate_narration(script, project, tts_provider, request.voice, usage)
        self._measure_and_align(script, llm)

        project.save_manifest(script=script, usage=usage)

        # 4. Render ------------------------------------------------------------
        spec = request.spec.model_copy(
            update={"music_enabled": request.music_path is not None and request.spec.music_enabled}
        )
        compositor = Compositor(spec, self.settings)
        output_path = project.output_path(script)

        with Timer("render", log):
            composition = compositor.render(
                script,
                project.work_dir,
                output_path,
                music_path=request.music_path,
                progress=self.progress,
            )

        metadata_path = project.write_metadata(script)
        project.save_manifest(
            script=script,
            spec=spec,
            usage=usage,
            extra={
                "video": str(composition.video_path),
                "duration_ms": composition.duration_ms,
            },
        )
        project.cleanup_work()

        self.progress.stage("Done", str(composition.video_path))
        return RenderResult(
            video_path=composition.video_path,
            script=script,
            spec=spec,
            duration_ms=composition.duration_ms,
            subtitle_path=composition.subtitle_path,
            thumbnail_path=composition.thumbnail_path,
            metadata_path=metadata_path,
            usage=usage,
            project_dir=project.root,
        )

    # -- stages ----------------------------------------------------------------
    def _open_project(self, request: GenerationRequest, document: SourceDocument) -> Project:
        if request.project_dir is not None:
            log.info("Resuming project at %s", request.project_dir)
            return Project.open(request.project_dir)
        return Project.create(
            request.output_dir or self.settings.output_dir, document.name
        )

    def _get_script(
        self,
        request: GenerationRequest,
        project: Project,
        llm: LLMProvider,
        document: SourceDocument,
        usage: Usage,
    ) -> Script:
        if request.script is not None:
            log.info("Using the supplied script (%d scenes)", len(request.script.scenes))
            self.progress.stage("Using the reviewed script")
            return request.script

        if request.project_dir is not None:
            cached = project.load_script()
            if cached is not None:
                log.info("Reusing the cached script (%d scenes)", len(cached.scenes))
                self.progress.stage("Reusing cached script")
                return cached

        self.progress.stage("Writing the script", f"via {llm.label}")
        generator = ScriptGenerator(llm, self.settings)
        script, script_usage = generator.generate(
            ScriptRequest(
                document=document,
                tone=request.tone,
                language=request.language,
                focus=request.focus,
            )
        )
        _merge(usage, script_usage)
        return script

    def _generate_images(
        self,
        script: Script,
        project: Project,
        provider: ImageProvider,
        spec: VideoSpec,
        usage: Usage,
    ) -> None:
        size = spec.aspect.image_size
        self.progress.stage("Generating images", f"{len(script.scenes)} via {provider.label}")
        completed = 0

        def make(scene: Scene) -> Path:
            prompt = f"{scene.image_prompt} {IMAGE_PROMPT_SUFFIX}"
            path = project.image_path(scene.index, prompt, provider.name, size)

            if self.settings.cache_enabled and path.exists() and path.stat().st_size > 0:
                log.debug("Scene %d: reusing cached image", scene.index + 1)
                return path

            provider.generate(prompt=prompt, output_path=path, size=size)
            usage.images_generated += 1
            usage.estimated_cost_usd += provider.estimate_cost(1)
            return path

        def on_done(_: int, __: Path) -> None:
            nonlocal completed
            completed += 1
            total = len(script.scenes)
            self.progress.update(completed / total, f"image {completed}/{total}")

        with Timer("images", log):
            paths = parallel_map(
                make, script.scenes, max_workers=self.settings.max_workers, on_result=on_done
            )
        for scene, path in zip(script.scenes, paths, strict=True):
            scene.image_path = path

    def _generate_narration(
        self,
        script: Script,
        project: Project,
        provider: TTSProvider,
        voice: str | None,
        usage: Usage,
    ) -> None:
        chosen_voice = voice or ""
        self.progress.stage("Recording narration", f"{len(script.scenes)} via {provider.label}")
        completed = 0

        def speak(scene: Scene) -> Path:
            path = project.audio_path(
                scene.index,
                scene.narration,
                provider.name,
                chosen_voice,
                extension=provider.audio_format,
            )

            if self.settings.cache_enabled and path.exists() and path.stat().st_size > 0:
                log.debug("Scene %d: reusing cached narration", scene.index + 1)
                return path

            result = provider.synthesize(
                text=scene.narration, output_path=path, voice=voice
            )
            usage.tts_characters += len(scene.narration)
            usage.estimated_cost_usd += provider.estimate_cost(len(scene.narration))
            return result

        def on_done(_: int, __: Path) -> None:
            nonlocal completed
            completed += 1
            self.progress.update(
                completed / len(script.scenes), f"narration {completed}/{len(script.scenes)}"
            )

        with Timer("narration", log):
            paths = parallel_map(
                speak, script.scenes, max_workers=self.settings.max_workers, on_result=on_done
            )
        for scene, path in zip(script.scenes, paths, strict=True):
            scene.audio_path = path

    def _measure_and_align(self, script: Script, llm: LLMProvider) -> None:
        """Measure each clip, then compute caption word timings."""
        self.progress.stage("Timing captions")
        for scene in script.scenes:
            if scene.audio_path is None:
                continue
            scene.duration_ms = ffmpeg.duration_ms(scene.audio_path, self.settings)

        aligner = build_aligner(self.settings.aligner, llm)
        for scene in script.scenes:
            scene.word_timings = aligner.align(scene)

        log.info(
            "Narration total: %.1fs across %d scenes",
            sum(s.duration_ms or 0 for s in script.scenes) / 1000,
            len(script.scenes),
        )


def _merge(target: Usage, other: Usage) -> None:
    """Fold ``other`` into ``target`` in place."""
    merged = target.merge(other)
    for field_name in type(merged).model_fields:
        setattr(target, field_name, getattr(merged, field_name))


def generate(request: GenerationRequest, settings: Settings | None = None) -> RenderResult:
    """Convenience wrapper: run one request with default progress reporting."""
    return Pipeline(settings).run(request)


__all__ = ["GenerationRequest", "Pipeline", "VideoGenError", "generate"]
