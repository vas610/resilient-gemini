"""ADK model wrapper: linear-backoff retries on a primary model, then fallback."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncGenerator

import httpx
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import errors, types

from .config import (
    DEFAULT_RETRYABLE_CODES,
    PRIORITY_COST_NOTE,
    PRIORITY_HEADERS,
    PriorityMode,
)

logger = logging.getLogger("resilient_gemini")

# Indirection so tests can replace sleeping without patching asyncio globally.
_sleep = asyncio.sleep

_TRANSIENT_NETWORK_ERRORS = (httpx.TimeoutException, httpx.ConnectError, asyncio.TimeoutError)


def is_retryable(exc: BaseException, codes: frozenset[int] = DEFAULT_RETRYABLE_CODES) -> bool:
    """True for transient API status codes and transient network failures."""
    if isinstance(exc, errors.APIError):
        return getattr(exc, "code", None) in codes
    return isinstance(exc, _TRANSIENT_NETWORK_ERRORS)


class RetryThenFallbackLlm(BaseLlm):
    """Calls ``primary`` up to ``max_attempts`` times, waiting ``step_seconds * n``
    plus up to ``step_jitter`` random seconds between tries (60s, 120s, 180s, ...
    by default). If every try fails with a transient error, the same request is
    sent once to ``backup``.

    Non-transient errors (400, 403, 404, ...) are raised immediately: retrying
    them would only waste time. A streamed response that has already produced
    output is never retried, so callers never see duplicated text.
    """

    primary: BaseLlm
    backup: BaseLlm
    primary_thinking: types.ThinkingLevel | None = None
    backup_thinking: types.ThinkingLevel | None = None
    max_attempts: int = 5
    step_seconds: float = 60.0
    step_jitter: float = 0.0
    retryable_codes: frozenset[int] = DEFAULT_RETRYABLE_CODES
    priority_paygo: PriorityMode = PriorityMode.OFF

    # ------------------------------------------------------------------ helpers
    def _prepare(self, req: LlmRequest, model: str, level: types.ThinkingLevel | None) -> None:
        req.model = model
        if req.config is None:
            req.config = types.GenerateContentConfig()
        if level is not None:
            req.config.thinking_config = types.ThinkingConfig(thinking_level=level)
        if self.priority_paygo is not PriorityMode.OFF:
            self._add_priority_headers(req.config)
            logger.info(
                "PRIORITY PAYGO (%s): request to %s is %s",
                self.priority_paygo.value,
                model,
                PRIORITY_COST_NOTE,
            )

    def _add_priority_headers(self, config: types.GenerateContentConfig) -> None:
        headers = PRIORITY_HEADERS[self.priority_paygo]
        existing = config.http_options
        if existing is None:
            config.http_options = types.HttpOptions(headers=dict(headers))
        else:
            merged = dict(existing.headers or {})
            merged.update(headers)
            config.http_options = existing.model_copy(update={"headers": merged})

    def _log_traffic_type(self, model: str, resp: LlmResponse) -> None:
        """Report whether Google actually served the call at priority."""
        if self.priority_paygo is PriorityMode.OFF or resp.usage_metadata is None:
            return
        traffic = getattr(resp.usage_metadata, "traffic_type", None)
        if traffic is None:
            return
        name = getattr(traffic, "value", str(traffic))
        if "PRIORITY" in name:
            logger.info(
                "PRIORITY PAYGO: %s served at priority (traffic_type=%s), %s", model, name, PRIORITY_COST_NOTE
            )
        elif name == "ON_DEMAND":
            logger.warning(
                "PRIORITY PAYGO: %s was downgraded to Standard PayGo "
                "(traffic_type=%s), billed at the standard rate",
                model,
                name,
            )
        else:
            logger.info("PRIORITY PAYGO: %s traffic_type=%s", model, name)

    async def _call(self, model: BaseLlm, req: LlmRequest, stream: bool, state: dict):
        last = None
        async for resp in model.generate_content_async(req, stream):
            state["yielded"] = True
            if resp.usage_metadata is not None:
                last = resp
            yield resp
        if last is not None:
            self._log_traffic_type(model.model, last)

    # --------------------------------------------------------------- main flow
    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        last_exc: BaseException | None = None

        for attempt in range(1, self.max_attempts + 1):
            state = {"yielded": False}
            try:
                self._prepare(llm_request, self.primary.model, self.primary_thinking)
                async for resp in self._call(self.primary, llm_request, stream, state):
                    yield resp
                if attempt > 1:
                    logger.info(
                        "Primary model %s succeeded on attempt %d/%d",
                        self.primary.model,
                        attempt,
                        self.max_attempts,
                    )
                return
            except Exception as exc:
                if state["yielded"] or not is_retryable(exc, self.retryable_codes):
                    raise
                last_exc = exc
                if attempt == self.max_attempts:
                    break
                wait = self.step_seconds * attempt
                if self.step_jitter:
                    wait += random.uniform(0, self.step_jitter)
                logger.warning(
                    "Primary model %s attempt %d/%d failed (%s). Retrying in %.0fs",
                    self.primary.model,
                    attempt,
                    self.max_attempts,
                    exc,
                    wait,
                )
                await _sleep(wait)

        logger.error(
            "FALLBACK: primary model %s failed %d times (last error: %s). "
            "Switching to backup model %s (thinking=%s)",
            self.primary.model,
            self.max_attempts,
            last_exc,
            self.backup.model,
            self.backup_thinking.name if self.backup_thinking else "default",
            extra={
                "resilient_gemini_event": "fallback",
                "primary_model": self.primary.model,
                "backup_model": self.backup.model,
            },
        )
        self._prepare(llm_request, self.backup.model, self.backup_thinking)
        try:
            async for resp in self._call(self.backup, llm_request, stream, {"yielded": False}):
                yield resp
        except Exception as exc:
            logger.exception("FALLBACK: backup model %s also failed: %s", self.backup.model, exc)  # noqa: TRY401
            raise
        logger.info("FALLBACK: backup model %s responded successfully", self.backup.model)
