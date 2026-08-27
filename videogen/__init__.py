"""videogen - turn source material into a narrated, captioned short-form video.

    from videogen import GenerationRequest, generate
    from videogen.models import AspectRatio, VideoSpec

    result = generate(
        GenerationRequest(
            source="paper.pdf",
            spec=VideoSpec(aspect=AspectRatio.VERTICAL),
            voice="Rachel",
        )
    )
    print(result.video_path)

Every stage is swappable through `videogen.providers`; see the README for the
architecture and `videogen doctor` for what your machine can currently run.
"""

from __future__ import annotations

__version__ = "2.0.0"

from .config import Settings, get_settings
from .errors import (
    ConfigError,
    DependencyError,
    IngestError,
    ProviderError,
    RenderError,
    ScriptError,
    VideoGenError,
)
from .models import (
    AspectRatio,
    KenBurnsKind,
    RenderResult,
    Scene,
    Script,
    SourceDocument,
    TransitionKind,
    Usage,
    VideoSpec,
)
from .pipeline import GenerationRequest, Pipeline, generate

__all__ = [
    "AspectRatio",
    "ConfigError",
    "DependencyError",
    "GenerationRequest",
    "IngestError",
    "KenBurnsKind",
    "Pipeline",
    "ProviderError",
    "RenderError",
    "RenderResult",
    "Scene",
    "Script",
    "ScriptError",
    "Settings",
    "SourceDocument",
    "TransitionKind",
    "Usage",
    "VideoGenError",
    "VideoSpec",
    "__version__",
    "generate",
    "get_settings",
]
