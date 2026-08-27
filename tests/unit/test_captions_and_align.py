"""Caption layout and word alignment.

These cover the two things the original renderer got wrong: word timing was split
evenly regardless of word length, and line layout drifted because the cursor advance
did not match the measured width used for centring.
"""

from __future__ import annotations

import pytest

from videogen.models import Scene, VideoSpec
from videogen.render.align import HeuristicAligner, _weight
from videogen.render.captions import CaptionRenderer, _parse_color


class TestHeuristicAligner:
    def test_covers_the_whole_clip(self):
        scene = Scene(index=0, narration="one two three four five", image_prompt="x")
        scene.duration_ms = 5000
        timings = HeuristicAligner().align(scene)

        assert len(timings) == 5
        assert timings[0].start_ms == 0
        # Must end exactly on the clip duration, with no gaps between words.
        assert timings[-1].end_ms == pytest.approx(5000)
        for earlier, later in zip(timings, timings[1:]):
            assert later.start_ms == pytest.approx(earlier.end_ms)

    def test_longer_words_get_more_time(self):
        scene = Scene(index=0, narration="a extraordinarily", image_prompt="x")
        scene.duration_ms = 2000
        short, long = HeuristicAligner().align(scene)
        assert long.duration_ms > short.duration_ms * 2

    def test_trailing_punctuation_adds_a_pause(self):
        assert _weight("word.") > _weight("word")
        assert _weight("word?") > _weight("word,")

    def test_digits_cost_extra_time(self):
        assert _weight("2024") > _weight("abcd")

    def test_returns_nothing_without_a_duration(self):
        scene = Scene(index=0, narration="hello world", image_prompt="x")
        assert HeuristicAligner().align(scene) == []


class TestCaptionRenderer:
    @pytest.fixture
    def renderer(self):
        return CaptionRenderer(VideoSpec(caption_font_size=40, caption_words_per_line=3))

    def test_groups_words_into_lines(self, renderer):
        scene = Scene(index=0, narration="one two three four five six seven", image_prompt="x")
        scene.duration_ms = 4000
        scene.word_timings = HeuristicAligner().align(scene)

        plan = renderer.plan(scene)
        assert len(plan.lines) == 3
        assert plan.lines[0].words == ["one", "two", "three"]
        assert plan.lines[-1].words == ["seven"]

    def test_lines_cover_the_clip_without_gaps(self, renderer):
        scene = Scene(index=0, narration="alpha bravo charlie delta echo", image_prompt="x")
        scene.duration_ms = 3000
        scene.word_timings = HeuristicAligner().align(scene)

        plan = renderer.plan(scene)
        assert plan.lines[0].start_ms == 0
        assert plan.lines[-1].end_ms == pytest.approx(3000)

    def test_line_at_holds_the_last_line_through_trailing_silence(self, renderer):
        scene = Scene(index=0, narration="one two", image_prompt="x")
        scene.duration_ms = 1000
        scene.word_timings = HeuristicAligner().align(scene)

        plan = renderer.plan(scene)
        assert plan.line_at(999_999) is plan.lines[-1]

    def test_active_word_advances_with_time(self, renderer):
        scene = Scene(index=0, narration="one two three", image_prompt="x")
        scene.duration_ms = 3000
        scene.word_timings = HeuristicAligner().align(scene)

        line = renderer.plan(scene).lines[0]
        assert line.active_word(0) == 0
        assert line.active_word(2999) == 2

    def test_overlay_is_cropped_and_inside_the_canvas(self, renderer):
        scene = Scene(index=0, narration="hello world", image_prompt="x")
        scene.duration_ms = 1000
        scene.word_timings = HeuristicAligner().align(scene)
        plan = renderer.plan(scene)

        overlay = renderer.overlay_at(plan, 100)
        assert overlay is not None
        height, width = overlay.rgb.shape[:2]
        # Cropped to the text, not the full frame.
        assert 0 < width < renderer.width
        assert 0 < height < renderer.height
        # And it lands entirely on screen.
        assert overlay.x >= 0 and overlay.y >= 0
        assert overlay.x + width <= renderer.width
        assert overlay.y + height <= renderer.height

    def test_overlay_is_horizontally_centred(self, renderer):
        """The original drifted right because advance != measured width."""
        scene = Scene(index=0, narration="alpha bravo charlie", image_prompt="x")
        scene.duration_ms = 1000
        scene.word_timings = HeuristicAligner().align(scene)
        overlay = renderer.overlay_at(renderer.plan(scene), 100)

        left_margin = overlay.x
        right_margin = renderer.width - (overlay.x + overlay.rgb.shape[1])
        assert abs(left_margin - right_margin) <= 4

    def test_overlays_are_cached(self, renderer):
        scene = Scene(index=0, narration="one two three", image_prompt="x")
        scene.duration_ms = 3000
        scene.word_timings = HeuristicAligner().align(scene)
        plan = renderer.plan(scene)
        assert renderer.overlay_at(plan, 10) is renderer.overlay_at(plan, 12)

    def test_composite_does_not_mutate_the_input_frame(self, renderer):
        import numpy as np

        scene = Scene(index=0, narration="hello world", image_prompt="x")
        scene.duration_ms = 1000
        scene.word_timings = HeuristicAligner().align(scene)
        overlay = renderer.overlay_at(renderer.plan(scene), 100)

        frame = np.zeros((renderer.height, renderer.width, 3), dtype=np.uint8)
        result = overlay.composite(frame)
        assert frame.sum() == 0          # untouched
        assert result.sum() > 0          # caption drawn

    def test_empty_scene_produces_no_lines(self, renderer):
        scene = Scene(index=0, narration="hello", image_prompt="x")
        assert renderer.plan(scene).lines == []


class TestColorParsing:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("#FFFFFF", (255, 255, 255, 255)),
            ("#000", (0, 0, 0, 255)),
            ("#FF000080", (255, 0, 0, 128)),
            ("red", (255, 0, 0, 255)),
        ],
    )
    def test_parses_common_formats(self, value, expected):
        assert _parse_color(value) == expected

    def test_falls_back_to_white_on_garbage(self):
        assert _parse_color("not-a-color") == (255, 255, 255, 255)
