"""OpenAI image generation (gpt-image-1 / DALL-E 3)."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from ...config import Settings
from ...errors import DependencyError, ProviderError
from ...logging import get_logger
from ...utils import retry
from ..base import ImageProvider

log = get_logger("providers.image.openai")

# USD per image at the sizes this project uses.
_PRICING = {"gpt-image-1": 0.042, "dall-e-3": 0.080}
_DEFAULT_PRICE = 0.05

# dall-e-3 accepts a different size vocabulary than gpt-image-1.
_DALLE3_SIZES = {
    "1024x1536": "1024x1792",
    "1536x1024": "1792x1024",
    "1024x1024": "1024x1024",
}


class OpenAIImage(ImageProvider):
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.image_model
        self._client: Any = None

    def health_check(self) -> None:
        self.settings.require("openai_api_key", provider="OpenAI images")
        self._ensure_client()

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise DependencyError(
                "The openai package is not installed.",
                hint="pip install 'videogen[openai]'",
            ) from exc
        self._client = OpenAI(
            api_key=self.settings.require("openai_api_key", provider="OpenAI images"),
            timeout=self.settings.request_timeout_s,
            max_retries=0,
        )
        return self._client

    def generate(self, *, prompt: str, output_path: Path, size: str = "1024x1536") -> Path:
        return self._generate(prompt=prompt, output_path=output_path, size=size)

    @retry(attempts=4)
    def _generate(self, *, prompt: str, output_path: Path, size: str) -> Path:
        client = self._ensure_client()
        if self.model == "dall-e-3":
            size = _DALLE3_SIZES.get(size, "1024x1792")

        kwargs: dict[str, Any] = {"model": self.model, "prompt": prompt, "size": size, "n": 1}
        if self.model == "dall-e-3":
            # gpt-image-1 always returns b64; dall-e-3 needs to be asked.
            kwargs["response_format"] = "b64_json"
            kwargs["quality"] = "standard"

        try:
            response = client.images.generate(**kwargs)
        except Exception as exc:
            raise _wrap(exc) from exc

        payload = response.data[0]
        encoded = getattr(payload, "b64_json", None)
        if not encoded:
            raise ProviderError(
                "OpenAI returned no image data.", provider=self.name, retryable=True
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(base64.b64decode(encoded))
        log.debug("wrote %s (%d bytes)", output_path.name, output_path.stat().st_size)
        return output_path

    def estimate_cost(self, count: int) -> float:
        return count * _PRICING.get(self.model, _DEFAULT_PRICE)


def _wrap(exc: Exception) -> ProviderError:
    kind = type(exc).__name__
    message = str(exc)
    transient = kind in {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
    }
    hint = None
    # The content filter is the single most common hard failure for this pipeline,
    # because image prompts are model-written. Say so plainly instead of retrying.
    if "content_policy" in message or "safety system" in message:
        hint = (
            "The image prompt tripped the content filter. The script generator is "
            "instructed to avoid named people and sensitive imagery; re-running usually "
            "produces a different prompt."
        )
        transient = False
    elif kind == "AuthenticationError":
        hint = "Check OPENAI_API_KEY. If this key was ever committed to git, rotate it."
    return ProviderError(
        f"OpenAI images {kind}: {exc}", provider="openai", retryable=transient, hint=hint
    )
