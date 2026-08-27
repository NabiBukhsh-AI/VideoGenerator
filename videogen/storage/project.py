"""Project directory and run manifest.

The original wrote everything into ``shorts/<unix timestamp>/`` with no record of what
produced it, so a failed run left an unlabelled directory of orphaned files and a
re-run regenerated every paid asset from scratch.

A `Project` here owns a named directory, records a manifest of what was generated, and
content-addresses images and narration by prompt hash. Re-running after a crash - or
after tweaking only the caption colour - reuses everything already paid for.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..logging import get_logger
from ..models import Script, Usage, VideoSpec
from ..utils import slugify, stable_hash

log = get_logger("storage")

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 2


class Project:
    """A single generation run on disk."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.audio_dir = self.root / "audio"
        self.work_dir = self.root / "work"
        self._manifest: dict[str, Any] = {}

    # -- construction ----------------------------------------------------------
    @classmethod
    def create(cls, output_dir: Path, name: str, *, timestamp: bool = True) -> Project:
        """Create a fresh project directory named after the source."""
        slug = slugify(name) or "short"
        if timestamp:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            slug = f"{stamp}-{slug}"

        root = Path(output_dir) / slug
        # Never silently merge into an unrelated existing run.
        suffix = 2
        while root.exists() and any(root.iterdir()):
            root = Path(output_dir) / f"{slug}-{suffix}"
            suffix += 1

        project = cls(root)
        project.ensure_dirs()
        log.debug("Project directory: %s", root)
        return project

    @classmethod
    def open(cls, root: Path) -> Project:
        """Reopen an existing project, loading its manifest."""
        project = cls(Path(root))
        if not project.root.exists():
            raise FileNotFoundError(f"No project at {root}")
        project.ensure_dirs()
        project.load_manifest()
        return project

    def ensure_dirs(self) -> None:
        for directory in (self.root, self.images_dir, self.audio_dir, self.work_dir):
            directory.mkdir(parents=True, exist_ok=True)

    # -- asset paths -----------------------------------------------------------
    def image_path(self, scene_index: int, prompt: str, provider: str, size: str) -> Path:
        """Content-addressed image path.

        Keying on the prompt means an unchanged scene reuses its image across runs, and
        a reworded one gets a new file instead of overwriting the old.
        """
        key = stable_hash(prompt, provider, size)
        return self.images_dir / f"scene_{scene_index + 1:02d}_{key}.png"

    def audio_path(self, scene_index: int, text: str, provider: str, voice: str,
                   extension: str = "mp3") -> Path:
        key = stable_hash(text, provider, voice)
        return self.audio_dir / f"scene_{scene_index + 1:02d}_{key}.{extension}"

    def output_path(self, script: Script, extension: str = "mp4") -> Path:
        return self.root / f"{slugify(script.title) or 'short'}.{extension}"

    # -- manifest --------------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    def load_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            try:
                self._manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Ignoring unreadable manifest at %s (%s)", self.manifest_path, exc)
                self._manifest = {}
        return self._manifest

    def save_manifest(
        self,
        *,
        script: Script | None = None,
        spec: VideoSpec | None = None,
        usage: Usage | None = None,
        providers: dict[str, str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Write the run manifest. Called after each stage so a crash leaves a record."""
        payload: dict[str, Any] = {
            **self._manifest,
            "version": MANIFEST_VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        payload.setdefault("created_at", payload["updated_at"])

        if script is not None:
            payload["script"] = json.loads(script.model_dump_json())
        if spec is not None:
            payload["spec"] = json.loads(spec.model_dump_json())
        if usage is not None:
            payload["usage"] = json.loads(usage.model_dump_json())
        if providers is not None:
            payload["providers"] = providers
        if extra:
            payload.update(extra)

        self._manifest = payload
        self.manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return self.manifest_path

    def load_script(self) -> Script | None:
        """Restore the script from a previous run, if one completed."""
        data = self.load_manifest().get("script")
        if not data:
            return None
        try:
            return Script.model_validate(data)
        except Exception as exc:
            log.warning("Could not restore the cached script: %s", exc)
            return None

    def write_metadata(self, script: Script) -> Path:
        """Human-readable publishing metadata next to the video."""
        path = self.root / "metadata.md"
        hashtags = " ".join(f"#{tag}" for tag in script.hashtags)
        lines = [
            f"# {script.title}",
            "",
            script.description or "",
            "",
            f"**Hashtags:** {hashtags}" if hashtags else "",
            f"**Source:** {script.source_name}" if script.source_name else "",
            f"**Scenes:** {len(script.scenes)}",
            "",
            "## Narration",
            "",
        ]
        lines += [f"{i + 1}. {scene.narration}" for i, scene in enumerate(script.scenes)]
        path.write_text("\n".join(line for line in lines if line is not None) + "\n",
                        encoding="utf-8")
        return path

    # -- housekeeping ----------------------------------------------------------
    def cleanup_work(self) -> None:
        """Remove intermediates, keeping the video and its sidecars."""
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Project({self.root})"
