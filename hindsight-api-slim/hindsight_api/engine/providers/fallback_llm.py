"""Gemini -> Claude -> GPT fallback chain with a Gemini-specific cool-down."""

import logging
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any

from ..llm_interface import LLMInterface
from ..response_models import LLMToolCallResult
from .anthropic_llm import AnthropicLLM
from .gemini_llm import GeminiLLM
from .openai_compatible_llm import OpenAICompatibleLLM

logger = logging.getLogger(__name__)


class FallbackLLM(LLMInterface):
    """Hard-coded Gemini -> Claude -> GPT failover chain.

    Invariants enforced at __init__:
      - Gemini is always present (GEMINI_API_KEY required).
      - At least one fallback (Claude or GPT) is always present.

    This matches the whole point of using FallbackLLM: if there is no fallback,
    use GeminiLLM directly instead. Enforcing these invariants up front removes
    a pile of "what if" branches from the call path.

    Each delegate is tried with max_retries=0 (one attempt). The first success
    wins; if all fail, the last exception is re-raised.

    Gemini cool-down: after Gemini fails, it is skipped for GEMINI_COOLDOWN_SECONDS.
    This avoids paying Gemini's 90s wait_for timeout on every call during bad
    patches (gemini-3.1-pro-preview routinely hangs in multi-minute clusters).
    Claude and GPT have no cool-down; their failures are expected to be one-off,
    not clustered.

    Delegates are constructed from environment variables at __init__ time:
        GEMINI_API_KEY      -> gemini-3.1-pro-preview    (required)
        ANTHROPIC_API_KEY   -> claude-sonnet-4-6         (at least one of these)
        OPENAI_API_KEY      -> gpt-5.4                   (at least one of these)
    """

    # 5 minutes: long enough to absorb a multi-call bad patch without
    # repeatedly paying Gemini's timeout cost, short enough to re-probe
    # and resume using Gemini (free credits) once it recovers.
    GEMINI_COOLDOWN_SECONDS = 300.0

    def __init__(self, provider: str, reasoning_effort: str = "low"):
        gemini: GeminiLLM | None = None
        claude: AnthropicLLM | None = None
        gpt: OpenAICompatibleLLM | None = None

        if key := os.getenv("GEMINI_API_KEY"):
            gemini = GeminiLLM(
                provider="gemini",
                api_key=key,
                base_url="",
                model="gemini-3.1-pro-preview",
                reasoning_effort=reasoning_effort,
            )
        if key := os.getenv("ANTHROPIC_API_KEY"):
            claude = AnthropicLLM(
                provider="anthropic",
                api_key=key,
                base_url="",
                model="claude-sonnet-4-6",
                reasoning_effort=reasoning_effort,
            )
        if key := os.getenv("OPENAI_API_KEY"):
            gpt = OpenAICompatibleLLM(
                provider="openai",
                api_key=key,
                base_url="",
                model="gpt-5.4",
                reasoning_effort=reasoning_effort,
            )

        if gemini is None:
            raise ValueError(
                "FallbackLLM requires GEMINI_API_KEY. "
                "If you don't want Gemini, use the target provider directly "
                "instead of the fallback provider."
            )
        if claude is None and gpt is None:
            raise ValueError(
                "FallbackLLM requires at least one fallback: ANTHROPIC_API_KEY "
                "or OPENAI_API_KEY. Without a fallback, use GeminiLLM directly."
            )

        # Composite model label like "gemini:gemini-3.1-pro-preview+anthropic:claude-sonnet-4-6"
        parts = [f"gemini:{gemini.model}"]
        if claude is not None:
            parts.append(f"anthropic:{claude.model}")
        if gpt is not None:
            parts.append(f"openai:{gpt.model}")

        super().__init__(
            provider=provider,
            api_key="",
            base_url="",
            model="+".join(parts),
        )

        self.gemini: GeminiLLM = gemini
        self.claude: AnthropicLLM | None = claude
        self.gpt: OpenAICompatibleLLM | None = gpt

        # None = not in cool-down (initial state, or cleared after a successful
        # probe). A future monotonic timestamp means cool-down is active until
        # that time.
        self._gemini_cooldown_until: float | None = None

    @property
    def _gemini_cooldown_remaining(self) -> float | None:
        """Seconds until Gemini cool-down expires, or None if not in cool-down."""
        if self._gemini_cooldown_until is None:
            return None
        remaining = self._gemini_cooldown_until - time.monotonic()
        return remaining if remaining > 0 else None

    async def verify_connection(self) -> None:
        return

    async def _call_with_fallback(
        self,
        coro_factory: Callable[[LLMInterface], Awaitable[Any]],
    ) -> Any:
        last_exc: Exception | None = None
        fallbacks: list[LLMInterface] = [d for d in (self.claude, self.gpt) if d is not None]

        # Gemini: skip when in cool-down.
        cooldown_remaining = self._gemini_cooldown_remaining
        if cooldown_remaining is not None:
            logger.debug(
                "FallbackLLM: gemini in cool-down for %.0fs more, skipping",
                cooldown_remaining,
            )
        else:
            try:
                result = await coro_factory(self.gemini)
                # Successful call: clear any stale cool-down timestamp.
                self._gemini_cooldown_until = None
                return result
            except Exception as e:
                self._gemini_cooldown_until = time.monotonic() + self.GEMINI_COOLDOWN_SECONDS
                logger.warning(
                    "FallbackLLM: gemini/%s failed (%s: %s), cool-down %.0fs, trying fallback",
                    self.gemini.model,
                    type(e).__name__,
                    e,
                    self.GEMINI_COOLDOWN_SECONDS,
                )
                last_exc = e

        # Claude then GPT: plain loop, no cool-down.
        for d in fallbacks:
            try:
                return await coro_factory(d)
            except Exception as e:
                logger.warning(
                    "FallbackLLM: fallback %s/%s failed (%s: %s), trying next",
                    d.provider,
                    d.model,
                    type(e).__name__,
                    e,
                )
                last_exc = e

        raise last_exc  # type: ignore[misc]

    async def call(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        kwargs["max_retries"] = 0
        return await self._call_with_fallback(lambda d: d.call(messages, **kwargs))

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMToolCallResult:
        kwargs["max_retries"] = 0
        return await self._call_with_fallback(lambda d: d.call_with_tools(messages, tools, **kwargs))

    async def cleanup(self) -> None:
        await self.gemini.cleanup()
        if self.claude is not None:
            await self.claude.cleanup()
        if self.gpt is not None:
            await self.gpt.cleanup()
