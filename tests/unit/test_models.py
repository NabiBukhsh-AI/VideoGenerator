"""Domain model validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from videogen.models import (
    AspectRatio,
    Scene,
    Script,
    SourceDocument,
    Usage,
    VideoSpec,
    WordTiming,
)


class TestScene:
    def test_normalises_smart_punctuation(self):
        scene = Scene(index=0, narration="It’s a “test”…", image_prompt="x")
        assert scene.narration == 'It\'s a "test"...'

    def test_rejects_empty_narration(self):
        with pytest.raises(ValidationError):
            Scene(index=0, narration="", image_prompt="x")

    def test_is_rendered_requires_all_artifacts(self, tmp_path):
        scene = Scene(index=0, narration="hello there", image_prompt="x")
        assert not scene.is_rendered
        scene.image_path = tmp_path / "a.png"
        scene.audio_path = tmp_path / "a.mp3"
        assert not scene.is_rendered
        scene.duration_ms = 1000
        assert scene.is_rendered

    def test_word_count(self):
        scene = Scene(index=0, narration="one two three four", image_prompt="x")
        assert scene.word_count == 4


class TestWordTiming:
    def test_rejects_reversed_span(self):
        with pytest.raises(ValidationError):
            WordTiming(word="x", start_ms=500, end_ms=100)

    def test_duration(self):
        assert WordTiming(word="x", start_ms=100, end_ms=350).duration_ms == 250


class TestScript:
    def test_requires_at_least_one_scene(self):
        with pytest.raises(ValidationError):
            Script(title="t", scenes=[])

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (["#Ocean", "science"], ["Ocean", "science"]),
            ("#a, #b  c", ["a", "b", "c"]),
            (["dup", "DUP"], ["dup"]),
            (["with space!"], ["withspace"]),
        ],
    )
    def test_hashtag_cleaning(self, raw, expected):
        script = Script(
            title="t",
            scenes=[Scene(index=0, narration="hello world", image_prompt="x")],
            hashtags=raw,
        )
        assert script.hashtags == expected

    def test_renumber_makes_indices_contiguous(self):
        script = Script(
            title="t",
            scenes=[
                Scene(index=7, narration="a b c", image_prompt="x"),
                Scene(index=3, narration="d e f", image_prompt="y"),
            ],
        )
        assert [s.index for s in script.renumber().scenes] == [0, 1]

    def test_estimated_duration_uses_measured_audio_when_present(self, script):
        for scene in script.scenes:
            scene.duration_ms = 2000
        assert script.estimated_duration_ms == 6000

    def test_estimated_duration_falls_back_to_word_rate(self, script):
        assert script.estimated_duration_ms > 0

    def test_slug(self, script):
        assert script.slug == "test-short"


class TestVideoSpec:
    @pytest.mark.parametrize(
        "aspect,size",
        [
            (AspectRatio.VERTICAL, (1080, 1920)),
            (AspectRatio.SQUARE, (1080, 1080)),
            (AspectRatio.LANDSCAPE, (1920, 1080)),
        ],
    )
    def test_dimensions(self, aspect, size):
        assert VideoSpec(aspect=aspect).size == size

    def test_font_size_derived_from_height(self):
        spec = VideoSpec(aspect=AspectRatio.VERTICAL, caption_font_size=0)
        assert spec.resolved_font_size() == round(1920 * 0.042)

    def test_explicit_font_size_wins(self):
        assert VideoSpec(caption_font_size=64).resolved_font_size() == 64

    def test_rejects_out_of_range_fps(self):
        with pytest.raises(ValidationError):
            VideoSpec(fps=0)

    def test_frame_ms(self):
        assert VideoSpec(fps=25).frame_ms == 40.0


class TestSourceDocument:
    def test_truncate_records_original_length(self):
        doc = SourceDocument(text=" ".join(["word"] * 100), name="d")
        short = doc.truncate(10)
        assert short.word_count == 10
        assert short.metadata["truncated_from_words"] == 100

    def test_truncate_is_noop_when_short_enough(self):
        doc = SourceDocument(text="a b c", name="d")
        assert doc.truncate(100) is doc


class TestUsage:
    def test_merge_sums_every_field(self):
        a = Usage(llm_calls=1, images_generated=2, estimated_cost_usd=0.5)
        b = Usage(llm_calls=3, images_generated=1, estimated_cost_usd=0.25)
        merged = a.merge(b)
        assert merged.llm_calls == 4
        assert merged.images_generated == 3
        assert merged.estimated_cost_usd == 0.75
