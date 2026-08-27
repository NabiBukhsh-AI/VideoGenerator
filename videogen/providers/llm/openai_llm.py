"""OpenAI chat provider with native structured output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...config import Settings
from ...errors import DependencyError, ProviderError
from ...logging import get_logger
from ...models import Usage
from ...utils import retry
from ..base import LLMProvider, LLMResponse

log = get_logger("providers.llm.openai")

# Rough USD per 1M tokens, used only for the run's cost estimate. Update freely -
# being slightly stale is fine, silently reporting zero is not.
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}
_DEFAULT_PRICING = (2.00, 8.00)


class OpenAILLM(LLMProvider):
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.openai_model
        self._client: Any = None

    # -- lifecycle -------------------------------------------------------------
    def health_check(self) -> None:
        self.settings.require("openai_api_key", provider="OpenAI")
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
            api_key=self.settings.require("openai_api_key", provider="OpenAI"),
            timeout=self.settings.request_timeout_s,
            max_retries=0,  # we own retries so backoff is uniform across providers
        )
        return self._client

    # -- LLMProvider -----------------------------------------------------------
    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        temperature: float = 0.7,
    ) -> LLMResponse:
        return self._complete_json(
            system=system, user=user, schema=schema, temperature=temperature
        )

    @retry(attempts=4)
    def _complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        temperature: float,
    ) -> LLMResponse:
        client = self._ensure_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "short_video_script",
                        "strict": True,
                        "schema": schema,
                    },
                },
            )
        except Exception as exc:
            raise _wrap(exc) from exc

        message = response.choices[0].message
        if getattr(message, "refusal", None):
            raise ProviderError(
                f"OpenAI refused the request: {message.refusal}",
                provider=self.name,
                hint="Rephrase the source material or lower the amount of sensitive content.",
            )

        text = message.content or ""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"OpenAI returned invalid JSON: {exc}", provider=self.name, retryable=True
            ) from exc

        return LLMResponse(data=data, raw_text=text, usage=self._usage(response))

    def _usage(self, response: Any) -> Usage:
        raw = getattr(response, "usage", None)
        if raw is None:
            return Usage(llm_calls=1)
        prompt_in = getattr(raw, "prompt_tokens", 0) or 0
        prompt_out = getattr(raw, "completion_tokens", 0) or 0
        in_rate, out_rate = _PRICING.get(self.model, _DEFAULT_PRICING)
        return Usage(
            llm_calls=1,
            llm_input_tokens=prompt_in,
            llm_output_tokens=prompt_out,
            estimated_cost_usd=(prompt_in * in_rate + prompt_out * out_rate) / 1_000_000,
        )

    # -- optional capability ---------------------------------------------------
    @retry(attempts=3)
    def transcribe_words(self, audio_path: Path) -> list[dict[str, Any]]:
        """Word-level timestamps via Whisper, used by the caption aligner."""
        client = self._ensure_client()
        try:
            with audio_path.open("rb") as handle:
                result = client.audio.transcriptions.create(
                    model=self.settings.whisper_model,
                    file=handle,
                    response_format="verbose_json",
                    timestamp_granularities=["word"],
                )
        except Exception as exc:
            log.warning("Whisper alignment failed for %s: %s", audio_path.name, exc)
            return []

        words = getattr(result, "words", None) or []
        return [
            {
                "word": getattr(w, "word", "") or "",
                "start": float(getattr(w, "start", 0.0) or 0.0),
                "end": float(getattr(w, "end", 0.0) or 0.0),
            }
            for w in words
        ]


def _wrap(exc: Exception) -> ProviderError:
    """Map vendor exceptions onto ProviderError, marking transient ones retryable."""
    kind = type(exc).__name__
    transient = kind in {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
    }
    hint = None
    if kind == "AuthenticationError":
        hint = "Check OPENAI_API_KEY. If this key was ever committed to git, rotate it."
    elif kind == "RateLimitError":
        hint = "You are being rate limited or are out of quota; the run will back off and retry."
    return ProviderError(f"OpenAI {kind}: {exc}", provider="openai", retryable=transient, hint=hint)
