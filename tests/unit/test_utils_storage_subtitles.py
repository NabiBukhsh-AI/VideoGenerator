"""Utilities, project storage and subtitle export."""

from __future__ import annotations

import json
import time

import pytest

from videogen.errors import ProviderError
from videogen.models import Usage, VideoSpec
from videogen.providers.tts.silent_tts import estimate_speech_ms
from videogen.render.subtitles import _timestamp, build_cues, write_srt, write_vtt
from videogen.storage import Project
from videogen.utils import (
    format_duration,
    normalize_text,
    parallel_map,
    retry,
    slugify,
    stable_hash,
)


class TestNormalizeText:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("it’s", "it's"),
            ("“quoted”", '"quoted"'),
            ("wait…", "wait..."),
            ("a  b\t\tc", "a b c"),
            ("  padded  ", "padded"),
        ],
    )
    def test_cleans_punctuation_and_whitespace(self, raw, expected):
        assert normalize_text(raw) == expected

    def test_returns_the_value(self):
        """The original called str.replace and discarded the result."""
        assert normalize_text("it’s") != "it’s"


class TestSlugify:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Hello World", "hello-world"),
            ("A/B\\C:D", "abcd"),
            ("Ünïcodé Tëxt", "unicode-text"),
            ("", "untitled"),
            ("!!!", "untitled"),
        ],
    )
    def test_produces_safe_slugs(self, raw, expected):
        assert slugify(raw) == expected

    def test_respects_max_length(self):
        assert len(slugify("word " * 100, max_length=20)) <= 20


class TestStableHash:
    def test_is_deterministic(self):
        assert stable_hash("a", 1, {"k": "v"}) == stable_hash("a", 1, {"k": "v"})

    def test_differs_on_different_input(self):
        assert stable_hash("a") != stable_hash("b")

    def test_is_insensitive_to_dict_ordering(self):
        assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


class TestRetry:
    def test_retries_retryable_provider_errors(self):
        calls = []

        @retry(attempts=3, base_delay=0.001, jitter=False)
        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise ProviderError("transient", retryable=True)
            return "ok"

        assert flaky() == "ok"
        assert len(calls) == 3

    def test_does_not_retry_non_retryable_errors(self):
        calls = []

        @retry(attempts=5, base_delay=0.001, jitter=False)
        def hard_fail():
            calls.append(1)
            raise ProviderError("permanent", retryable=False)

        with pytest.raises(ProviderError):
            hard_fail()
        assert len(calls) == 1

    def test_reraises_after_exhausting_attempts(self):
        @retry(attempts=2, base_delay=0.001, jitter=False)
        def always():
            raise ProviderError("nope", retryable=True)

        with pytest.raises(ProviderError):
            always()


class TestParallelMap:
    def test_preserves_input_order(self):
        assert parallel_map(lambda x: x * 2, [1, 2, 3, 4, 5], max_workers=3) == [2, 4, 6, 8, 10]

    def test_handles_an_empty_list(self):
        assert parallel_map(lambda x: x, [], max_workers=4) == []

    def test_actually_runs_concurrently(self):
        def slow(_):
            time.sleep(0.15)
            return 1

        started = time.perf_counter()
        parallel_map(slow, list(range(4)), max_workers=4)
        # Sequential would be ~0.6s.
        assert time.perf_counter() - started < 0.45

    def test_propagates_the_first_error(self):
        def boom(x):
            if x == 2:
                raise ValueError("boom")
            return x

        with pytest.raises(ValueError, match="boom"):
            parallel_map(boom, [1, 2, 3], max_workers=2)

    def test_reports_each_result(self):
        seen = []
        parallel_map(lambda x: x, [1, 2, 3], max_workers=1, on_result=lambda i, v: seen.append(v))
        assert sorted(seen) == [1, 2, 3]


class TestFormatDuration:
    @pytest.mark.parametrize(
        "ms,expected", [(0, "0:00.000"), (1500, "0:01.500"), (65_250, "1:05.250")]
    )
    def test_formats(self, ms, expected):
        assert format_duration(ms) == expected


class TestEstimateSpeech:
    def test_scales_with_length(self):
        assert estimate_speech_ms("one two three four five six") > estimate_speech_ms("one two")

    def test_has_a_floor(self):
        assert estimate_speech_ms("hi") >= 900

    def test_punctuation_adds_pause(self):
        assert estimate_speech_ms("one, two, three.") > estimate_speech_ms("one two three")


class TestProject:
    def test_create_makes_the_directory_tree(self, tmp_path):
        project = Project.create(tmp_path, "My Source Document")
        assert project.root.exists()
        assert project.images_dir.exists()
        assert project.audio_dir.exists()
        assert "my-source-document" in project.root.name

    def test_does_not_collide_with_an_existing_run(self, tmp_path):
        first = Project.create(tmp_path, "same", timestamp=False)
        (first.root / "marker.txt").write_text("x", encoding="utf-8")
        second = Project.create(tmp_path, "same", timestamp=False)
        assert first.root != second.root

    def test_image_paths_are_content_addressed(self, tmp_path):
        project = Project.create(tmp_path, "test")
        same = project.image_path(0, "a prompt", "openai", "1024x1536")
        again = project.image_path(0, "a prompt", "openai", "1024x1536")
        different = project.image_path(0, "another prompt", "openai", "1024x1536")
        assert same == again
        assert same != different

    def test_manifest_round_trips_the_script(self, tmp_path, script):
        project = Project.create(tmp_path, "test")
        project.save_manifest(script=script, spec=VideoSpec(), usage=Usage(llm_calls=2))

        reopened = Project.open(project.root)
        restored = reopened.load_script()
        assert restored is not None
        assert restored.title == script.title
        assert len(restored.scenes) == len(script.scenes)

    def test_survives_a_corrupt_manifest(self, tmp_path):
        project = Project.create(tmp_path, "test")
        project.manifest_path.write_text("{ not json", encoding="utf-8")
        assert project.load_manifest() == {}
        assert project.load_script() is None

    def test_manifest_is_valid_json(self, tmp_path, script):
        project = Project.create(tmp_path, "test")
        path = project.save_manifest(script=script, spec=VideoSpec())
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["version"] == 2
        assert data["script"]["title"] == script.title

    def test_writes_publishing_metadata(self, tmp_path, script):
        project = Project.create(tmp_path, "test")
        content = project.write_metadata(script).read_text(encoding="utf-8")
        assert script.title in content
        assert "#ocean" in content
        assert script.scenes[0].narration in content

    def test_cleanup_removes_only_the_work_dir(self, tmp_path):
        project = Project.create(tmp_path, "test")
        (project.work_dir / "temp.txt").write_text("x", encoding="utf-8")
        (project.images_dir / "keep.png").write_text("x", encoding="utf-8")
        project.cleanup_work()
        assert not project.work_dir.exists()
        assert (project.images_dir / "keep.png").exists()


class TestSubtitles:
    @pytest.mark.parametrize(
        "ms,sep,expected",
        [
            (0, ",", "00:00:00,000"),
            (1234, ",", "00:00:01,234"),
            (3_661_500, ".", "01:01:01.500"),
        ],
    )
    def test_timestamp_format(self, ms, sep, expected):
        assert _timestamp(ms, sep) == expected

    def test_cues_advance_along_the_timeline(self, rendered_script):
        cues = build_cues(rendered_script.scenes)
        assert len(cues) == len(rendered_script.scenes)
        assert cues[0].start_ms == 0
        for earlier, later in zip(cues, cues[1:]):
            assert later.start_ms >= earlier.end_ms
            assert later.index == earlier.index + 1

    def test_gap_offsets_subsequent_cues(self, rendered_script):
        without = build_cues(rendered_script.scenes, gap_ms=0)
        with_gap = build_cues(rendered_script.scenes, gap_ms=500)
        assert with_gap[1].start_ms == pytest.approx(without[1].start_ms + 500)

    def test_long_narration_is_split(self, rendered_script):
        cues = build_cues(rendered_script.scenes, max_chars=20)
        assert len(cues) > len(rendered_script.scenes)

    def test_srt_output_is_well_formed(self, rendered_script, tmp_path):
        cues = build_cues(rendered_script.scenes)
        content = write_srt(cues, tmp_path / "out.srt").read_text(encoding="utf-8")
        blocks = content.strip().split("\n\n")
        assert len(blocks) == len(cues)
        assert blocks[0].splitlines()[0] == "1"
        assert " --> " in blocks[0].splitlines()[1]

    def test_vtt_has_the_required_header(self, rendered_script, tmp_path):
        cues = build_cues(rendered_script.scenes)
        content = write_vtt(cues, tmp_path / "out.vtt").read_text(encoding="utf-8")
        assert content.startswith("WEBVTT")
        assert "." in content.split("-->")[0].splitlines()[-1]
