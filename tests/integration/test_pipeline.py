"""End-to-end pipeline runs.

Everything here uses offline providers. Tests that produce an actual MP4 are marked
``ffmpeg`` and skip automatically when the binary is absent.
"""

from __future__ import annotations

import json

import pytest

from videogen.errors import DependencyError
from videogen.models import AspectRatio, KenBurnsKind, VideoSpec
from videogen.pipeline import GenerationRequest, Pipeline
from videogen.render import ffmpeg
from videogen.storage import Project


class TestScriptOnly:
    """Runs that stop before asset generation - these need no ffmpeg."""

    def test_produces_a_script_and_a_project(self, settings, sample_text, tmp_path):
        result = Pipeline(settings).run(
            GenerationRequest(
                source=sample_text, output_dir=tmp_path, dry_run=True, script_only=True
            )
        )
        assert result.script.title
        assert settings.min_scenes <= len(result.script.scenes) <= settings.max_scenes
        assert result.project_dir.exists()
        assert result.metadata_path.exists()

    def test_writes_a_manifest_recording_the_providers(self, settings, sample_text, tmp_path):
        result = Pipeline(settings).run(
            GenerationRequest(
                source=sample_text, output_dir=tmp_path, dry_run=True, script_only=True
            )
        )
        manifest = json.loads((result.project_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["providers"] == {
            "llm": "echo", "image": "placeholder", "tts": "silent"
        }
        assert manifest["script"]["title"] == result.script.title

    def test_a_supplied_script_skips_generation(self, settings, sample_text, tmp_path, script):
        result = Pipeline(settings).run(
            GenerationRequest(
                source=sample_text, script=script, output_dir=tmp_path,
                dry_run=True, script_only=True,
            )
        )
        assert result.script.title == script.title
        assert result.usage.llm_calls == 0

    def test_scene_count_honours_settings(self, sample_text, tmp_path):
        from videogen.config import Settings

        settings = Settings(
            llm_provider="echo", image_provider="placeholder", tts_provider="silent",
            min_scenes=3, max_scenes=3,
        )
        result = Pipeline(settings).run(
            GenerationRequest(
                source=sample_text, output_dir=tmp_path, dry_run=True, script_only=True
            )
        )
        assert len(result.script.scenes) == 3


class TestAssetGeneration:
    """Image and narration generation, without rendering."""

    def test_generates_and_caches_assets(self, settings, sample_text, tmp_path, monkeypatch):
        from videogen.ingest import load_source
        from videogen.models import Usage
        from videogen.providers.image.placeholder_image import PlaceholderImage
        from videogen.providers.registry import build_llm
        from videogen.script.generator import ScriptGenerator, ScriptRequest

        pipeline = Pipeline(settings)
        document = load_source(sample_text)
        script, _ = ScriptGenerator(build_llm(settings), settings).generate(
            ScriptRequest(document=document)
        )
        project = Project.create(tmp_path, "test")

        calls = {"n": 0}
        real_generate = PlaceholderImage.generate

        def counting(self, **kwargs):
            calls["n"] += 1
            return real_generate(self, **kwargs)

        monkeypatch.setattr(PlaceholderImage, "generate", counting)

        usage = Usage()
        provider = PlaceholderImage()
        spec = VideoSpec()
        pipeline._generate_images(script, project, provider, spec, usage)
        first_pass = calls["n"]
        assert first_pass == len(script.scenes)
        assert all(s.image_path.exists() for s in script.scenes)

        # Second pass must hit the cache and make no new calls.
        pipeline._generate_images(script, project, provider, spec, usage)
        assert calls["n"] == first_pass

    def test_narration_durations_are_measured(self, settings, sample_text, tmp_path):
        from videogen.ingest import load_source
        from videogen.models import Usage
        from videogen.providers.registry import build_llm, build_tts
        from videogen.script.generator import ScriptGenerator, ScriptRequest

        pipeline = Pipeline(settings)
        script, _ = ScriptGenerator(build_llm(settings), settings).generate(
            ScriptRequest(document=load_source(sample_text))
        )
        project = Project.create(tmp_path, "test")

        pipeline._generate_narration(script, project, build_tts(settings), None, Usage())
        pipeline._measure_and_align(script, build_llm(settings))

        for scene in script.scenes:
            assert scene.audio_path.exists()
            assert scene.duration_ms > 0
            assert len(scene.word_timings) == scene.word_count
            assert scene.word_timings[-1].end_ms == pytest.approx(scene.duration_ms)


class TestFullRender:
    @pytest.mark.ffmpeg
    def test_produces_a_playable_mp4(self, settings, sample_text, tmp_path):
        spec = VideoSpec(aspect=AspectRatio.VERTICAL, fps=12, ken_burns=KenBurnsKind.NONE)
        result = Pipeline(settings).run(
            GenerationRequest(
                source=sample_text, spec=spec, output_dir=tmp_path, dry_run=True
            )
        )

        assert result.video_path.exists()
        assert result.video_path.suffix == ".mp4"
        assert result.video_path.stat().st_size > 1000

        info = ffmpeg.probe(result.video_path)
        streams = {s["codec_type"]: s for s in info["streams"]}
        assert streams["video"]["codec_name"] == "h264"
        assert streams["video"]["pix_fmt"] == "yuv420p"
        assert (streams["video"]["width"], streams["video"]["height"]) == (1080, 1920)
        assert "audio" in streams

    @pytest.mark.ffmpeg
    def test_picture_and_audio_stay_in_sync(self, settings, sample_text, tmp_path):
        spec = VideoSpec(fps=12, ken_burns=KenBurnsKind.NONE)
        result = Pipeline(settings).run(
            GenerationRequest(
                source=sample_text, spec=spec, output_dir=tmp_path, dry_run=True
            )
        )
        info = ffmpeg.probe(result.video_path)
        streams = {s["codec_type"]: s for s in info["streams"]}
        video_s = float(streams["video"]["duration"])
        audio_s = float(streams["audio"]["duration"])
        assert abs(video_s - audio_s) < 0.5

    @pytest.mark.ffmpeg
    def test_writes_subtitles_and_a_thumbnail(self, settings, sample_text, tmp_path):
        result = Pipeline(settings).run(
            GenerationRequest(
                source=sample_text,
                spec=VideoSpec(fps=12, ken_burns=KenBurnsKind.NONE),
                output_dir=tmp_path,
                dry_run=True,
            )
        )
        assert result.subtitle_path.exists()
        assert result.subtitle_path.read_text(encoding="utf-8").strip()
        assert result.subtitle_path.with_suffix(".vtt").exists()
        assert result.thumbnail_path.exists()

    @pytest.mark.ffmpeg
    @pytest.mark.parametrize("aspect", list(AspectRatio))
    def test_every_aspect_ratio_renders(self, settings, sample_text, tmp_path, aspect):
        result = Pipeline(settings).run(
            GenerationRequest(
                source=sample_text,
                spec=VideoSpec(aspect=aspect, fps=8, ken_burns=KenBurnsKind.NONE),
                output_dir=tmp_path / aspect.name,
                dry_run=True,
            )
        )
        info = ffmpeg.probe(result.video_path)
        video = next(s for s in info["streams"] if s["codec_type"] == "video")
        assert (video["width"], video["height"]) == aspect.dimensions

    @pytest.mark.ffmpeg
    def test_work_directory_is_cleaned_up(self, settings, sample_text, tmp_path):
        result = Pipeline(settings).run(
            GenerationRequest(
                source=sample_text,
                spec=VideoSpec(fps=8, ken_burns=KenBurnsKind.NONE),
                output_dir=tmp_path,
                dry_run=True,
            )
        )
        assert not (result.project_dir / "work").exists()


class TestFFmpegAbsent:
    def test_rendering_without_ffmpeg_fails_clearly(
        self, settings, sample_text, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(ffmpeg, "available", lambda *a, **k: False)
        with pytest.raises(DependencyError, match="ffmpeg"):
            Pipeline(settings).run(
                GenerationRequest(source=sample_text, output_dir=tmp_path, dry_run=True)
            )

    def test_script_only_works_without_ffmpeg(
        self, settings, sample_text, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(ffmpeg, "available", lambda *a, **k: False)
        result = Pipeline(settings).run(
            GenerationRequest(
                source=sample_text, output_dir=tmp_path, dry_run=True, script_only=True
            )
        )
        assert result.script.scenes
