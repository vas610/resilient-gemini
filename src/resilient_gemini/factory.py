"""One-call factory that teams use: ``model=resilient_model()``."""

from __future__ import annotations

import logging

from google.adk.models.google_llm import Gemini
from google.genai import types

from .config import (
    PRIORITY_COST_NOTE,
    PRIORITY_HEADERS,
    PRIORITY_PAYGO_ENV,
    PriorityMode,
    ResilienceConfig,
)
from .llm import RetryThenFallbackLlm

logger = logging.getLogger("resilient_gemini")


def resilient_model(config: ResilienceConfig | None = None, **overrides) -> RetryThenFallbackLlm:
    """Build an ADK model with linear retries on the primary and a logged fallback.

    Settings come from, in order of precedence:
      1. keyword overrides, e.g. ``resilient_model(primary_thinking="low")``
      2. an explicit ``config``
      3. RESILIENT_GEMINI_* environment variables
      4. built-in defaults (gemini-3.5-flash -> gemini-3.6-flash, 5 tries, 60s step)

    Priority PayGo is the one exception: it can only be switched on with the
    RESILIENT_GEMINI_PRIORITY_PAYGO environment variable.

    Location, project and Vertex mode come from the standard Google env vars
    (GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION=us).
    """
    if config is None:
        config = ResilienceConfig.from_env(**overrides)
    else:
        config = config.with_overrides(**overrides)

    codes = sorted(config.retryable_codes)

    primary = Gemini(
        model=config.primary_model,
        # SDK retries off: the wrapper owns the 60s/120s/180s/... schedule.
        retry_options=types.HttpRetryOptions(attempts=1),
    )
    backup = Gemini(
        model=config.backup_model,
        retry_options=types.HttpRetryOptions(
            attempts=config.backup_attempts,
            initial_delay=config.backup_initial_delay,
            exp_base=config.backup_exp_base,
            max_delay=config.backup_max_delay,
            jitter=1,
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
        retryable_codes=config.retryable_codes,
        priority_paygo=config.priority_paygo,
    )


__all__ = ["resilient_model"]
