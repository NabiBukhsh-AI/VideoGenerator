"""Shared fixtures.

Everything here runs offline. Tests never touch a paid API, and only tests marked
``ffmpeg`` require the binary.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from videogen.config import Settings
from videogen.models import AspectRatio, Scene, Script, VideoSpec

SAMPLE_TEXT = (
    "The deep ocean remains the least explored region on Earth. "
    "Sunlight fades entirely below one thousand metres, leaving a permanent night. "
    "Creatures there produce their own light through a process called bioluminescence. "
    "Pressure at the seafloor exceeds one thousand atmospheres. "
    "Despite this, thriving ecosystems cluster around hydrothermal vents. "
    "Scientists estimate that most deep sea species have never been catalogued."
)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Offline settings pointed at a temp output directory."""
    return Settings(
        llm_provider="echo",
        image_provider="placeholder",
        tts_provider="silent",
        output_dir=tmp_path / "output",
        max_scenes=4,
        min_scenes=3,
        max_workers=2,
    )


@pytest.fixture
def sample_text() -> str:
    return SAMPLE_TEXT


@pytest.fixture
def spec() -> VideoSpec:
    """A small canvas so frame tests stay fast."""
    return VideoSpec(aspect=AspectRatio.VERTICAL, fps=12, caption_font_size=40)


@pytest.fixture
def script() -> Script:
    return Script(
        title="Test Short",
        scenes=[
            Scene(index=0, narration="The deep ocean is mostly unexplored.",
                  image_prompt="dark ocean water, cinematic"),
            Scene(index=1, narration="Strange creatures glow in the permanent night.",
                  image_prompt="bioluminescent jellyfish, cinematic"),
            Scene(index=2, narration="Ecosystems thrive around hydrothermal vents.",
                  image_prompt="hydrothermal vent, cinematic"),
        ],
        description="A short about the deep sea.",
        hashtags=["ocean", "science"],
    )


@pytest.fixture
def rendered_script(script: Script, tmp_path: Path, spec: VideoSpec) -> Script:
    """A script with images, narration and timings already generated offline."""
    from videogen.providers.image.placeholder_image import PlaceholderImage
    from videogen.providers.tts.silent_tts import SilentTTS
    from videogen.render.align import HeuristicAligner
    from videogen.render.ffmpeg import duration_ms

    images = PlaceholderImage()
    tts = SilentTTS()
    aligner = HeuristicAligner()

    for scene in script.scenes:
        scene.image_path = images.generate(
            prompt=scene.image_prompt,
            output_path=tmp_path / f"image_{scene.index}.png",
            size="512x768",
        )
        scene.audio_path = tts.synthesize(
            text=scene.narration, output_path=tmp_path / f"audio_{scene.index}.wav"
        )
        scene.duration_ms = duration_ms(scene.audio_path)
        scene.word_timings = aligner.align(scene)
    return script


@pytest.fixture
def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def pytest_collection_modifyitems(config, items):
    """Skip ffmpeg-marked tests when the binary is absent."""
    if shutil.which("ffmpeg") is not None:
        return
    skip = pytest.mark.skip(reason="ffmpeg is not installed")
    for item in items:
        if "ffmpeg" in item.keywords:
            item.add_marker(skip)
