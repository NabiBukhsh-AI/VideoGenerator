"""Frame generation, framing and timeline behaviour."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from videogen.errors import RenderError
from videogen.models import KenBurnsKind, TransitionKind, VideoSpec
from videogen.render.captions import CaptionRenderer
from videogen.render.frames import SceneClip, Timeline, _blend, build_clips, load_cover


@pytest.fixture
def wide_image(tmp_path):
    """A landscape image, which a vertical canvas must crop rather than letterbox."""
    path = tmp_path / "wide.png"
    Image.new("RGB", (1600, 900), (200, 50, 50)).save(path)
    return path


class TestLoadCover:
    def test_fills_the_canvas_exactly(self, wide_image):
        image = load_cover(wide_image, (1080, 1920))
        assert image.size == (1080, 1920)

    def test_headroom_oversizes_for_motion(self, wide_image):
        image = load_cover(wide_image, (1080, 1920), headroom=0.2)
        assert image.size == (round(1080 * 1.2), round(1920 * 1.2))

    def test_crops_centrally(self, tmp_path):
        """A centred marker must survive the crop."""
        source = Image.new("RGB", (2000, 1000), (0, 0, 0))
        for x in range(950, 1050):
            for y in range(450, 550):
                source.putpixel((x, y), (255, 255, 255))
        path = tmp_path / "marked.png"
        source.save(path)

        result = np.asarray(load_cover(path, (500, 500)))
        centre = result[240:260, 240:260]
        assert centre.mean() > 200

    def test_raises_on_unreadable_file(self, tmp_path):
        bad = tmp_path / "not-an-image.png"
        bad.write_text("nope", encoding="utf-8")
        with pytest.raises(RenderError):
            load_cover(bad, (100, 100))


class TestSceneClip:
    def _clip(self, motion, size=(200, 400), intensity=0.15):
        headroom = 0.0 if motion is KenBurnsKind.NONE else intensity
        image = Image.new("RGB", (round(size[0] * (1 + headroom)),
                                 round(size[1] * (1 + headroom))), (100, 120, 140))
        return SceneClip(image=image, duration_ms=2000, motion=motion,
                         intensity=intensity, canvas=size)

    @pytest.mark.parametrize("motion", list(KenBurnsKind))
    def test_renders_canvas_sized_rgb_frames(self, motion):
        if motion is KenBurnsKind.AUTO:
            pytest.skip("AUTO is resolved to a concrete move by build_clips")
        clip = self._clip(motion)
        frame = clip.render(0.5)
        assert frame.shape == (400, 200, 3)
        assert frame.dtype == np.uint8

    @pytest.mark.parametrize("progress", [-1.0, 0.0, 0.5, 1.0, 2.0])
    def test_progress_is_clamped(self, progress):
        clip = self._clip(KenBurnsKind.ZOOM_IN)
        assert clip.render(progress).shape == (400, 200, 3)

    def test_static_frames_are_cached(self):
        clip = self._clip(KenBurnsKind.NONE)
        assert clip.render(0.0) is clip.render(1.0)

    def test_motion_actually_changes_the_frame(self, tmp_path):
        source = Image.new("RGB", (240, 480))
        for y in range(480):  # vertical gradient so panning is detectable
            for x in range(240):
                source.putpixel((x, y), (y % 256, x % 256, 128))
        path = tmp_path / "grad.png"
        source.save(path)

        clip = SceneClip(image=load_cover(path, (200, 400), headroom=0.15),
                         duration_ms=2000, motion=KenBurnsKind.ZOOM_IN,
                         intensity=0.15, canvas=(200, 400))
        assert not np.array_equal(clip.render(0.0), clip.render(1.0))


class TestTimeline:
    def _clips(self, count=3, duration=1000.0):
        return [
            SceneClip(image=Image.new("RGB", (100, 200), (i * 40, 0, 0)),
                      duration_ms=duration, motion=KenBurnsKind.NONE,
                      intensity=0.0, canvas=(100, 200))
            for i in range(count)
        ]

    def test_total_duration_is_the_sum_of_scenes(self):
        timeline = Timeline(self._clips(3, 1500), VideoSpec(fps=10))
        assert timeline.total_ms == 4500

    def test_frame_count_matches_duration_and_fps(self):
        timeline = Timeline(self._clips(2, 1000), VideoSpec(fps=30))
        assert timeline.frame_count == 60

    def test_scene_boundaries_select_the_right_clip(self):
        timeline = Timeline(self._clips(3, 1000), VideoSpec(fps=10, transition_ms=0))
        assert timeline._index_at(0) == 0
        assert timeline._index_at(999) == 0
        assert timeline._index_at(1000) == 1
        assert timeline._index_at(2500) == 2

    def test_yields_exactly_frame_count_frames(self):
        timeline = Timeline(self._clips(2, 500), VideoSpec(fps=10, transition_ms=0))
        assert len(list(timeline.frames())) == timeline.frame_count

    def test_crossfade_blends_toward_the_next_scene(self):
        spec = VideoSpec(fps=30, transition=TransitionKind.CROSSFADE, transition_ms=400)
        timeline = Timeline(self._clips(2, 1000), spec)

        mid_scene = timeline.frame_at(300)[0, 0, 0]
        deep_in_fade = timeline.frame_at(950)[0, 0, 0]
        # Scene 0 is red 0, scene 1 is red 40: the blend must move toward 40.
        assert mid_scene == 0
        assert deep_in_fade > mid_scene

    def test_cut_transition_does_not_blend(self):
        spec = VideoSpec(fps=30, transition=TransitionKind.CUT)
        timeline = Timeline(self._clips(2, 1000), spec)
        assert timeline.frame_at(950)[0, 0, 0] == 0

    def test_final_scene_fades_out(self):
        spec = VideoSpec(fps=30, transition=TransitionKind.CROSSFADE, transition_ms=400)
        clips = self._clips(1, 1000)
        clips[0].image = Image.new("RGB", (100, 200), (255, 255, 255))
        timeline = Timeline(clips, spec)
        assert timeline.frame_at(999).mean() < timeline.frame_at(100).mean()

    def test_rejects_an_empty_timeline(self):
        with pytest.raises(RenderError):
            Timeline([], VideoSpec())


class TestBuildClips:
    def test_auto_motion_alternates_between_scenes(self, rendered_script, spec):
        clips = build_clips(rendered_script.scenes, spec)
        motions = [clip.motion for clip in clips]
        assert KenBurnsKind.AUTO not in motions
        assert len(set(motions)) > 1

    def test_durations_come_from_the_audio(self, rendered_script, spec):
        clips = build_clips(rendered_script.scenes, spec)
        for clip, scene in zip(clips, rendered_script.scenes):
            assert clip.duration_ms == scene.duration_ms

    def test_requires_an_image(self, script, spec):
        script.scenes[0].duration_ms = 1000
        with pytest.raises(RenderError, match="no image"):
            build_clips(script.scenes, spec)

    def test_requires_a_duration(self, script, spec, tmp_path):
        from videogen.providers.image.placeholder_image import PlaceholderImage

        for scene in script.scenes:
            scene.image_path = PlaceholderImage().generate(
                prompt="x", output_path=tmp_path / f"{scene.index}.png", size="64x128"
            )
        with pytest.raises(RenderError, match="duration"):
            build_clips(script.scenes, spec)

    def test_captions_are_planned_per_scene(self, rendered_script, spec):
        renderer = CaptionRenderer(spec)
        clips = build_clips(rendered_script.scenes, spec, renderer)
        assert all(clip.plan is not None and clip.plan.lines for clip in clips)


class TestBlend:
    def test_endpoints_return_the_originals(self):
        a = np.zeros((2, 2, 3), dtype=np.uint8)
        b = np.full((2, 2, 3), 255, dtype=np.uint8)
        assert np.array_equal(_blend(a, b, 0.0), a)
        assert np.array_equal(_blend(a, b, 1.0), b)

    def test_midpoint_is_between(self):
        a = np.zeros((2, 2, 3), dtype=np.uint8)
        b = np.full((2, 2, 3), 200, dtype=np.uint8)
        assert 90 < _blend(a, b, 0.5).mean() < 110

    def test_does_not_overflow(self):
        a = np.full((4, 4, 3), 255, dtype=np.uint8)
        b = np.full((4, 4, 3), 255, dtype=np.uint8)
        assert _blend(a, b, 0.5).max() == 255
