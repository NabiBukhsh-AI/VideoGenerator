"""Domain models.

The original pipeline passed around a list of untyped dicts (``{"type": "image", ...}``)
parsed out of free-form LLM prose. Everything downstream had to re-derive meaning from
that shape, and a malformed line silently produced a video with missing scenes.

Here a `Script` is a validated object: scene count, narration text and image prompts are
checked once at the boundary, and every later stage works against a real type.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .utils import normalize_text, slugify


class AspectRatio(str, Enum):
    """Target canvas shape. Presets cover the platforms people actually publish to."""

    VERTICAL = "9:16"
    SQUARE = "1:1"
    LANDSCAPE = "16:9"

    @property
    def dimensions(self) -> tuple[int, int]:
        return {
            AspectRatio.VERTICAL: (1080, 1920),
            AspectRatio.SQUARE: (1080, 1080),
            AspectRatio.LANDSCAPE: (1920, 1080),
        }[self]

    @property
    def image_size(self) -> str:
        """Closest size the OpenAI image API supports for this aspect."""
        return {
            AspectRatio.VERTICAL: "1024x1536",
            AspectRatio.SQUARE: "1024x1024",
            AspectRatio.LANDSCAPE: "1536x1024",
        }[self]


class TransitionKind(str, Enum):
    CROSSFADE = "crossfade"
    CUT = "cut"
    FADE_TO_BLACK = "fade_to_black"


class KenBurnsKind(str, Enum):
    """Slow camera move applied to each still so the frame is never static."""

    NONE = "none"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    AUTO = "auto"  # alternate per scene


class WordTiming(BaseModel):
    """One word with its position inside a scene's narration audio."""

    word: str
    start_ms: float = Field(ge=0)
    end_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def _check_order(self) -> WordTiming:
        if self.end_ms < self.start_ms:
            raise ValueError(f"word {self.word!r}: end_ms {self.end_ms} < start_ms {self.start_ms}")
        return self

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


class Scene(BaseModel):
    """A single narrated beat: one image, one line of narration, one audio clip."""

    model_config = ConfigDict(validate_assignment=True)

    index: int = Field(ge=0)
    narration: str = Field(min_length=1)
    image_prompt: str = Field(min_length=1)

    # Artifacts filled in as the pipeline runs.
    image_path: Path | None = None
    audio_path: Path | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    word_timings: list[WordTiming] = Field(default_factory=list)

    @field_validator("narration", "image_prompt", mode="before")
    @classmethod
    def _clean(cls, value: Any) -> Any:
        return normalize_text(value) if isinstance(value, str) else value

    @property
    def word_count(self) -> int:
        return len(self.narration.split())

    @property
    def is_rendered(self) -> bool:
        return (
            self.image_path is not None
            and self.audio_path is not None
            and self.duration_ms is not None
        )


class Script(BaseModel):
    """A complete, validated short-video script plus publishing metadata."""

    model_config = ConfigDict(validate_assignment=True)

    title: str = Field(min_length=1, max_length=120)
    scenes: list[Scene] = Field(min_length=1)
    description: str = ""
    hashtags: list[str] = Field(default_factory=list)
    source_name: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("hashtags", mode="before")
    @classmethod
    def _clean_hashtags(cls, value: Any) -> Any:
        """Accept a list of tags or one space/comma separated string; emit bare tags."""
        if isinstance(value, str):
            value = [part for part in value.replace(",", " ").split() if part]
        if not isinstance(value, list):
            return value
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in value:
            tag = str(raw).strip().lstrip("#").strip()
            tag = "".join(ch for ch in tag if ch.isalnum() or ch == "_")
            if tag and tag.lower() not in seen:
                seen.add(tag.lower())
                cleaned.append(tag)
        return cleaned

    @property
    def full_narration(self) -> str:
        return " ".join(scene.narration for scene in self.scenes)

    @property
    def total_words(self) -> int:
        return sum(scene.word_count for scene in self.scenes)

    @property
    def estimated_duration_ms(self) -> float:
        """Measured duration when audio exists, else ~150 wpm as a planning estimate."""
        if all(scene.duration_ms is not None for scene in self.scenes):
            return sum(scene.duration_ms or 0.0 for scene in self.scenes)
        return self.total_words / 150 * 60_000

    @property
    def slug(self) -> str:
        return slugify(self.title)

    def renumber(self) -> Script:
        """Reassign scene indices to be contiguous from zero."""
        for position, scene in enumerate(self.scenes):
            scene.index = position
        return self


class VideoSpec(BaseModel):
    """Everything the renderer needs that is not the script itself."""

    aspect: AspectRatio = AspectRatio.VERTICAL
    fps: int = Field(default=30, ge=1, le=120)
    transition: TransitionKind = TransitionKind.CROSSFADE
    transition_ms: float = Field(default=700, ge=0, le=5000)
    ken_burns: KenBurnsKind = KenBurnsKind.AUTO
    ken_burns_intensity: float = Field(default=0.12, ge=0.0, le=0.6)

    captions_enabled: bool = True
    caption_words_per_line: int = Field(default=4, ge=1, le=12)
    caption_font_size: int = Field(default=0, ge=0)  # 0 = derive from canvas height
    caption_color: str = "#FFFFFF"
    caption_highlight_color: str = "#FFD400"
    caption_stroke_color: str = "#000000"
    caption_stroke_width: int = Field(default=0, ge=0)  # 0 = derive from font size
    caption_bottom_margin: float = Field(default=0.22, ge=0.0, le=0.9)

    music_enabled: bool = False
    music_gain_db: float = Field(default=-18.0, ge=-60.0, le=0.0)
    music_duck_db: float = Field(default=-8.0, ge=-40.0, le=0.0)
    music_fade_ms: float = Field(default=1500, ge=0)

    crf: int = Field(default=20, ge=0, le=51)
    preset: str = "medium"
    audio_bitrate: str = "192k"

    @property
    def size(self) -> tuple[int, int]:
        return self.aspect.dimensions

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]

    @property
    def frame_ms(self) -> float:
        return 1000.0 / self.fps

    def resolved_font_size(self) -> int:
        return self.caption_font_size or max(28, round(self.height * 0.042))

    def resolved_stroke_width(self) -> int:
        return self.caption_stroke_width or max(2, round(self.resolved_font_size() * 0.09))


class SourceDocument(BaseModel):
    """Normalised text extracted from whatever the user supplied."""

    text: str = Field(min_length=1)
    name: str = "source"
    kind: str = "text"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def truncate(self, max_words: int) -> SourceDocument:
        """Cap length so a 300-page PDF does not blow the model context window."""
        words = self.text.split()
        if len(words) <= max_words:
            return self
        return self.model_copy(
            update={
                "text": " ".join(words[:max_words]),
                "metadata": {**self.metadata, "truncated_from_words": len(words)},
            }
        )


class Usage(BaseModel):
    """Accumulated API usage, so a run can report what it actually cost."""

    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    images_generated: int = 0
    tts_characters: int = 0
    estimated_cost_usd: float = 0.0

    def merge(self, other: Usage) -> Usage:
        return Usage(
            llm_calls=self.llm_calls + other.llm_calls,
            llm_input_tokens=self.llm_input_tokens + other.llm_input_tokens,
            llm_output_tokens=self.llm_output_tokens + other.llm_output_tokens,
            images_generated=self.images_generated + other.images_generated,
            tts_characters=self.tts_characters + other.tts_characters,
            estimated_cost_usd=self.estimated_cost_usd + other.estimated_cost_usd,
        )


class RenderResult(BaseModel):
    """What a completed run produced."""

    video_path: Path
    script: Script
    spec: VideoSpec
    duration_ms: float
    subtitle_path: Path | None = None
    thumbnail_path: Path | None = None
    metadata_path: Path | None = None
    usage: Usage = Field(default_factory=Usage)
    project_dir: Path | None = None
