"""Anthropic Claude provider.

Structured output is obtained by forcing a single tool call whose ``input_schema`` is
the requested schema, which is Claude's reliable path to a typed object.
"""

from __future__ import annotations

from typing import Any

from ...config import Settings
from ...errors import DependencyError, ProviderError
from ...logging import get_logger
from ...models import Usage
from ...utils import retry
from ..base import LLMProvider, LLMResponse

log = get_logger("providers.llm.anthropic")

# USD per 1M tokens (input, output).
_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-opus-4-1": (15.00, 75.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
_DEFAULT_PRICING = (3.00, 15.00)

_TOOL_NAME = "emit_script"


class AnthropicLLM(LLMProvider):
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.anthropic_model
        self._client: Any = None

    def health_check(self) -> None:
        self.settings.require("anthropic_api_key", provider="Anthropic")
        self._ensure_client()

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:
            raise DependencyError(
                "The anthropic package is not installed.",
                hint="pip install 'videogen[anthropic]'",
            ) from exc
        self._client = anthropic.Anthropic(
            api_key=self.settings.require("anthropic_api_key", provider="Anthropic"),
            timeout=self.settings.request_timeout_s,
            max_retries=0,
        )
        return self._client

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
            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
                tools=[
                    {
                        "name": _TOOL_NAME,
                        "description": "Return the finished short-video script.",
                        "input_schema": schema,
                    }
                ],
                tool_choice={"type": "tool", "name": _TOOL_NAME},
            )
        except Exception as exc:
            raise _wrap(exc) from exc

        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == _TOOL_NAME:
                return LLMResponse(
                    data=dict(block.input),
                    raw_text=str(block.input),
                    usage=self._usage(response),
                )

        raise ProviderError(
            "Claude did not return the expected tool call.",
            provider=self.name,
            retryable=True,
        )

    def _usage(self, response: Any) -> Usage:
        raw = getattr(response, "usage", None)
        if raw is None:
            return Usage(llm_calls=1)
        tokens_in = getattr(raw, "input_tokens", 0) or 0
        tokens_out = getattr(raw, "output_tokens", 0) or 0
        in_rate, out_rate = _PRICING.get(self.model, _DEFAULT_PRICING)
        return Usage(
            llm_calls=1,
            llm_input_tokens=tokens_in,
            llm_output_tokens=tokens_out,
            estimated_cost_usd=(tokens_in * in_rate + tokens_out * out_rate) / 1_000_000,
        )


def _wrap(exc: Exception) -> ProviderError:
    kind = type(exc).__name__
    transient = kind in {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "OverloadedError",
    }
    hint = "Check ANTHROPIC_API_KEY." if kind == "AuthenticationError" else None
    return ProviderError(
        f"Anthropic {kind}: {exc}", provider="anthropic", retryable=transient, hint=hint
    )
