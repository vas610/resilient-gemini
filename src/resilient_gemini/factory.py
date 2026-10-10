"""One-call factory that teams use: ``model=resilient_model()``."""

from __future__ import annotations

import logging
from functools import cached_property
from typing import Optional

from google.adk.models.google_llm import Gemini
from google.genai import Client, types

from .config import (
    PRIORITY_COST_NOTE,
    PRIORITY_HEADERS,
    PRIORITY_PAYGO_ENV,
    PriorityMode,
    ResilienceConfig,
)
from .llm import RetryThenFallbackLlm

logger = logging.getLogger("resilient_gemini")


class LocatedGemini(Gemini):
    """ADK ``Gemini`` pinned to a Vertex AI location instead of GOOGLE_CLOUD_LOCATION.

    This is the subclass-and-override-``api_client`` pattern from ADK's own
    ``Gemini`` docs. Only ``location`` is passed; the SDK still picks the endpoint
    (no custom base_url), so ``"us"`` still resolves to the US multi-region.
    """

    location: str

    @cached_property
    def api_client(self) -> Client:
        return Client(
            location=self.location,
            http_options=types.HttpOptions(
                headers=self._tracking_headers(),
                retry_options=self.retry_options,
            ),
        )


def _gemini(model: str, location: Optional[str], retry_options: types.HttpRetryOptions) -> Gemini:
    if location is None:
        return Gemini(model=model, retry_options=retry_options)
    return LocatedGemini(model=model, location=location, retry_options=retry_options)


def resilient_model(config: ResilienceConfig | None = None, **overrides) -> RetryThenFallbackLlm:
    """Build an ADK model with linear retries on the primary and a logged fallback.

    Settings come from, in order of precedence:
      1. keyword overrides, e.g. ``resilient_model(primary_thinking="low")``
      2. an explicit ``config``
      3. RESILIENT_GEMINI_* environment variables
      4. built-in defaults (gemini-3.5-flash -> gemini-3.6-flash, 5 tries, 60s step)

    Priority PayGo is the one exception: it can only be switched on with the
    RESILIENT_GEMINI_PRIORITY_PAYGO environment variable.

    Project and Vertex mode come from the standard Google env vars
    (GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_CLOUD_PROJECT). Location comes from
    GOOGLE_CLOUD_LOCATION (e.g. ``us``) unless primary_location / backup_location
    is set.
    """
    if config is None:
        config = ResilienceConfig.from_env(**overrides)
    else:
        config = config.with_overrides(**overrides)

    codes = sorted(config.retryable_codes)

    primary = _gemini(
        config.primary_model,
        config.primary_location,
        # SDK retries off: the wrapper owns the 60s/120s/180s/... schedule.
        types.HttpRetryOptions(attempts=1),
    )
    backup = _gemini(
        config.backup_model,
        config.backup_location,
        types.HttpRetryOptions(
            attempts=config.backup_attempts,
            initial_delay=config.backup_initial_delay,
            exp_base=config.backup_exp_base,
            max_delay=config.backup_max_delay,
            jitter=config.backup_jitter,
            http_status_codes=codes,
        ),
    )

    if config.priority_paygo is not PriorityMode.OFF:
        logger.warning(
            "PRIORITY PAYGO ENABLED via %s=%s for %s and %s. Every request is %s. "
            "Headers: %s",
            PRIORITY_PAYGO_ENV, config.priority_paygo.value,
            config.primary_model, config.backup_model, PRIORITY_COST_NOTE,
            PRIORITY_HEADERS[config.priority_paygo],
        )

    return RetryThenFallbackLlm(
        model=config.primary_model,
        primary=primary,
        backup=backup,
        primary_thinking=config.primary_thinking,
        backup_thinking=config.backup_thinking,
        max_attempts=config.max_attempts,
        step_seconds=config.step_seconds,
        step_jitter=config.step_jitter,
        retryable_codes=config.retryable_codes,
        priority_paygo=config.priority_paygo,
    )


__all__ = ["LocatedGemini", "resilient_model"]
