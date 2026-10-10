"""Behaviour tests. No network, no credentials: models are replaced by fakes and
sleeping is recorded instead of waited.

Run:  pip install -e ".[test]" && pytest -q
"""

import asyncio
from typing import AsyncGenerator, List, Optional

import httpx
import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import errors, types
from pydantic import Field

import resilient_gemini.llm as llm_mod
from resilient_gemini import (
    PriorityMode,
    ResilienceConfig,
    RetryThenFallbackLlm,
    parse_priority_mode,
    parse_thinking_level,
)


class FakeLlm(BaseLlm):
    """Fails the first ``fail_times`` calls with ``error_code``, then answers."""

    fail_times: int = 0
    error_code: int = 503
    calls: int = 0
    seen_thinking: List[Optional[types.ThinkingLevel]] = []
    seen_models: List[str] = []
    seen_headers: List[dict] = []
    traffic_type: Optional[types.TrafficType] = None

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.calls += 1
        cfg = llm_request.config
        tc = cfg.thinking_config if cfg else None
        self.seen_thinking.append(tc.thinking_level if tc else None)
        self.seen_models.append(llm_request.model)
        ho = cfg.http_options if cfg else None
        self.seen_headers.append(dict(ho.headers or {}) if ho else {})
        if self.calls <= self.fail_times:
            cls = errors.ServerError if self.error_code >= 500 else errors.ClientError
            raise cls(self.error_code, {"error": {"code": self.error_code, "message": "boom"}})
        usage = (
            types.GenerateContentResponseUsageMetadata(traffic_type=self.traffic_type)
            if self.traffic_type else None
        )
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=self.model)]),
            usage_metadata=usage,
        )


@pytest.fixture
def waits(monkeypatch):
    recorded: List[float] = []

    async def fake_sleep(seconds):
        recorded.append(seconds)

    monkeypatch.setattr(llm_mod, "_sleep", fake_sleep)
    return recorded


def run(model: RetryThenFallbackLlm) -> List[str]:
    async def go():
        req = LlmRequest(
            contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
            config=types.GenerateContentConfig(),
        )
        return [r.content.parts[0].text async for r in model.generate_content_async(req)]

    return asyncio.run(go())


def make(primary_fails=0, backup_fails=0, code=503, **kw) -> RetryThenFallbackLlm:
    return RetryThenFallbackLlm(
        model="primary",
        primary=FakeLlm(model="primary", fail_times=primary_fails, error_code=code),
        backup=FakeLlm(model="backup", fail_times=backup_fails, error_code=code),
        **kw,
    )


def test_success_first_try(waits):
    m = make()
    assert run(m) == ["primary"]
    assert m.primary.calls == 1 and m.backup.calls == 0
    assert waits == []


def test_linear_backoff_then_success(waits):
    m = make(primary_fails=2)
    assert run(m) == ["primary"]
    assert m.primary.calls == 3
    assert waits == [60, 120]


def test_five_failures_fall_back_with_1_2_3_4_minute_gaps(waits, caplog):
    m = make(primary_fails=99)
    with caplog.at_level("INFO", logger="resilient_gemini"):
        assert run(m) == ["backup"]
    assert m.primary.calls == 5
    assert m.backup.calls == 1
    assert waits == [60, 120, 180, 240]
    assert any("FALLBACK" in r.message and r.levelname == "ERROR" for r in caplog.records)
    assert m.backup.seen_models == ["backup"]


def test_non_retryable_error_raises_immediately(waits):
    m = make(primary_fails=1, code=400)
    with pytest.raises(errors.ClientError):
        run(m)
    assert m.primary.calls == 1 and m.backup.calls == 0
    assert waits == []


def test_backup_failure_is_raised(waits):
    m = make(primary_fails=99, backup_fails=99)
    with pytest.raises(errors.ServerError):
        run(m)


def test_429_is_retried(waits):
    m = make(primary_fails=1, code=429)
    assert run(m) == ["primary"]
    assert waits == [60]


def test_thinking_level_per_model(waits):
    m = make(
        primary_fails=99,
        primary_thinking=types.ThinkingLevel.LOW,
        backup_thinking=types.ThinkingLevel.HIGH,
    )
    run(m)
    assert set(m.primary.seen_thinking) == {types.ThinkingLevel.LOW}
    assert m.backup.seen_thinking == [types.ThinkingLevel.HIGH]


def test_custom_schedule(waits):
    m = make(primary_fails=99, max_attempts=3, step_seconds=10)
    run(m)
    assert waits == [10, 20]


def test_parse_thinking_level():
    assert parse_thinking_level("low") is types.ThinkingLevel.LOW
    assert parse_thinking_level("HIGH") is types.ThinkingLevel.HIGH
    assert parse_thinking_level(None) is None
    assert parse_thinking_level("default") is None
    with pytest.raises(ValueError):
        parse_thinking_level("turbo")


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("RESILIENT_GEMINI_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("RESILIENT_GEMINI_PRIMARY_THINKING", "minimal")
    cfg = ResilienceConfig.from_env(backup_thinking="high")
    assert cfg.max_attempts == 3
    assert cfg.primary_thinking is types.ThinkingLevel.MINIMAL
    assert cfg.backup_thinking is types.ThinkingLevel.HIGH
    assert cfg.primary_model == "gemini-3.5-flash"
    assert cfg.backup_model == "gemini-3.6-flash"


# ----------------------------------------------------------- Priority PayGo


def test_priority_off_by_default_sends_no_headers(waits, monkeypatch):
    monkeypatch.delenv("RESILIENT_GEMINI_PRIORITY_PAYGO", raising=False)
    assert ResilienceConfig.from_env().priority_paygo is PriorityMode.OFF
    m = make()
    run(m)
    assert m.primary.seen_headers == [{}]


def test_priority_mode_headers_on_primary_and_backup(waits, caplog):
    m = make(primary_fails=99, priority_paygo=PriorityMode.PRIORITY)
    with caplog.at_level("INFO", logger="resilient_gemini"):
        run(m)
    expected = {
        "X-Vertex-AI-LLM-Request-Type": "shared",
        "X-Vertex-AI-LLM-Shared-Request-Type": "priority",
    }
    assert all(h == expected for h in m.primary.seen_headers)
    assert m.backup.seen_headers == [expected]
    assert any("PRIORITY PAYGO" in r.message and "2x" in r.message for r in caplog.records)


def test_spillover_mode_sends_only_shared_header(waits):
    m = make(priority_paygo=PriorityMode.SPILLOVER)
    run(m)
    assert m.primary.seen_headers == [{"X-Vertex-AI-LLM-Shared-Request-Type": "priority"}]


def test_downgrade_to_standard_is_logged(waits, caplog):
    m = make(priority_paygo=PriorityMode.PRIORITY)
    m.primary.traffic_type = types.TrafficType.ON_DEMAND
    with caplog.at_level("INFO", logger="resilient_gemini"):
        run(m)
    assert any("downgraded" in r.message and r.levelname == "WARNING" for r in caplog.records)


def test_priority_parser_is_strict():
    assert parse_priority_mode(None) is PriorityMode.OFF
    assert parse_priority_mode("off") is PriorityMode.OFF
    assert parse_priority_mode("Priority") is PriorityMode.PRIORITY
    assert parse_priority_mode("spillover") is PriorityMode.SPILLOVER
    for bad in ("true", "yes", "1", "prio"):
        with pytest.raises(ValueError):
            parse_priority_mode(bad)


def test_priority_cannot_be_enabled_from_code(monkeypatch):
    from resilient_gemini import resilient_model

    monkeypatch.delenv("RESILIENT_GEMINI_PRIORITY_PAYGO", raising=False)
    with pytest.raises(ValueError):
        resilient_model(priority_paygo="priority")


def test_priority_from_env(monkeypatch, caplog):
    from resilient_gemini import resilient_model

    monkeypatch.setenv("RESILIENT_GEMINI_PRIORITY_PAYGO", "priority")
    with caplog.at_level("WARNING", logger="resilient_gemini"):
        m = resilient_model()
    assert m.priority_paygo is PriorityMode.PRIORITY
    assert any("PRIORITY PAYGO ENABLED" in r.message for r in caplog.records)


def test_factory_builds_without_network(monkeypatch):
    # Building the model must not create a client or call the network.
    from resilient_gemini import resilient_model

    m = resilient_model(primary_thinking="low", max_attempts=4)
    assert m.primary.model == "gemini-3.5-flash"
    assert m.backup.model == "gemini-3.6-flash"
    assert m.max_attempts == 4
    assert m.primary.retry_options.attempts == 1
    assert 499 in m.backup.retry_options.http_status_codes


def test_missing_google_adk_gives_clear_import_error(monkeypatch):
    # google-adk is a peer dependency; simulate it being absent.
    import importlib
    import sys

    for name in list(sys.modules):
        if name == "resilient_gemini" or name.startswith("resilient_gemini."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "google.adk", None)
    with pytest.raises(ImportError, match="needs google-adk"):
        importlib.import_module("resilient_gemini")


# --------------------------------------------- retry edge cases (more fakes)


class ScriptedLlm(BaseLlm):
    """Raises ``errors_to_raise`` one per call, then answers. If
    ``yield_then_fail`` is set, it yields one chunk and then raises."""

    errors_to_raise: list = Field(default_factory=list)
    yield_then_fail: Optional[Exception] = None
    calls: int = 0

    async def generate_content_async(self, llm_request, stream=False):
        self.calls += 1
        if self.calls <= len(self.errors_to_raise):
            raise self.errors_to_raise[self.calls - 1]
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=self.model)]))
        if self.yield_then_fail is not None:
            raise self.yield_then_fail


def api_error(code: int) -> errors.APIError:
    cls = errors.ServerError if code >= 500 else errors.ClientError
    return cls(code, {"error": {"code": code, "message": "boom"}})


def make_scripted(primary_errors=(), yield_then_fail=None, **kw) -> RetryThenFallbackLlm:
    return RetryThenFallbackLlm(
        model="primary",
        primary=ScriptedLlm(model="primary", errors_to_raise=list(primary_errors),
                            yield_then_fail=yield_then_fail),
        backup=ScriptedLlm(model="backup"),
        **kw,
    )


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectTimeout("connect timed out"),
        httpx.ReadTimeout("read timed out"),
        httpx.ConnectError("connection refused"),
        asyncio.TimeoutError(),
    ],
    ids=["connect-timeout", "read-timeout", "connect-error", "asyncio-timeout"],
)
def test_network_errors_are_retried(waits, exc):
    m = make_scripted(primary_errors=[exc, exc])
    assert run(m) == ["primary"]
    assert m.primary.calls == 3 and m.backup.calls == 0
    assert waits == [60, 120]


@pytest.mark.parametrize("code", [408, 429, 499, 500, 502, 503, 504])
def test_every_retryable_code_is_retried(waits, code):
    m = make_scripted(primary_errors=[api_error(code)])
    assert run(m) == ["primary"]
    assert waits == [60]


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_client_errors_never_retry_or_fall_back(waits, code):
    m = make_scripted(primary_errors=[api_error(code)])
    with pytest.raises(errors.ClientError):
        run(m)
    assert m.primary.calls == 1 and m.backup.calls == 0
    assert waits == []


def test_stream_that_already_yielded_is_not_retried(waits):
    m = make_scripted(yield_then_fail=api_error(503))
    seen: List[str] = []

    async def go():
        req = LlmRequest(contents=[], config=types.GenerateContentConfig())
        async for r in m.generate_content_async(req, stream=True):
            seen.append(r.content.parts[0].text)

    with pytest.raises(errors.ServerError):
        asyncio.run(go())
    assert seen == ["primary"]  # one chunk, no duplicate from a retry
    assert m.primary.calls == 1 and m.backup.calls == 0
    assert waits == []


def test_fallback_log_record_has_alerting_fields(waits, caplog):
    m = make(primary_fails=99)
    with caplog.at_level("ERROR", logger="resilient_gemini"):
        run(m)
    rec = next(r for r in caplog.records if r.message.startswith("FALLBACK:"))
    assert rec.name == "resilient_gemini"
    assert rec.resilient_gemini_event == "fallback"
    assert rec.primary_model == "primary"
    assert rec.backup_model == "backup"


def test_retry_then_fallback_full_story(waits, caplog):
    """Readable end-to-end demo: 5 transient failures, 1-2-3-4 minute gaps, then backup."""
    m = make_scripted(primary_errors=[api_error(503), api_error(429), httpx.ReadTimeout("slow"),
                                      api_error(500), api_error(504)])
    with caplog.at_level("INFO", logger="resilient_gemini"):
        assert run(m) == ["backup"]
    retries = [r.message for r in caplog.records if "Retrying in" in r.message]
    assert [msg.rsplit(" ", 1)[-1] for msg in retries] == ["60s", "120s", "180s", "240s"]
    assert waits == [60, 120, 180, 240]
    assert sum(r.message.startswith("FALLBACK:") for r in caplog.records) == 2  # switching + success


# ------------------------------------------------------- jitter and location


def test_step_jitter_adds_bounded_random_delay(waits):
    m = make(primary_fails=99, step_jitter=5)
    run(m)
    assert len(waits) == 4
    for base, w in zip([60, 120, 180, 240], waits):
        assert base <= w <= base + 5


def test_no_jitter_by_default_keeps_exact_schedule(waits):
    m = make(primary_fails=99)
    run(m)
    assert waits == [60, 120, 180, 240]


def test_negative_jitter_is_rejected():
    with pytest.raises(ValueError):
        ResilienceConfig(step_jitter=-1)
    with pytest.raises(ValueError):
        ResilienceConfig(backup_jitter=-1)


def test_new_settings_from_env(monkeypatch):
    monkeypatch.setenv("RESILIENT_GEMINI_PRIMARY_LOCATION", "us")
    monkeypatch.setenv("RESILIENT_GEMINI_BACKUP_LOCATION", "global")
    monkeypatch.setenv("RESILIENT_GEMINI_STEP_JITTER", "7.5")
    monkeypatch.setenv("RESILIENT_GEMINI_BACKUP_JITTER", "0")
    cfg = ResilienceConfig.from_env()
    assert cfg.primary_location == "us"
    assert cfg.backup_location == "global"
    assert cfg.step_jitter == 7.5
    assert cfg.backup_jitter == 0


def test_locations_default_to_none(monkeypatch):
    monkeypatch.delenv("RESILIENT_GEMINI_PRIMARY_LOCATION", raising=False)
    monkeypatch.setenv("RESILIENT_GEMINI_BACKUP_LOCATION", "  ")
    cfg = ResilienceConfig.from_env()
    assert cfg.primary_location is None and cfg.backup_location is None


def test_factory_without_location_uses_plain_gemini(monkeypatch):
    from google.adk.models.google_llm import Gemini

    from resilient_gemini import LocatedGemini, resilient_model

    for var in ("PRIMARY_LOCATION", "BACKUP_LOCATION"):
        monkeypatch.delenv("RESILIENT_GEMINI_" + var, raising=False)
    m = resilient_model(backup_jitter=0.5, step_jitter=3)
    assert type(m.primary) is Gemini and type(m.backup) is Gemini
    assert not isinstance(m.primary, LocatedGemini)
    assert m.backup.retry_options.jitter == 0.5
    assert m.step_jitter == 3


def test_factory_with_location_pins_client_location(monkeypatch):
    import resilient_gemini.factory as factory_mod
    from resilient_gemini import LocatedGemini, resilient_model

    created: List[dict] = []
    monkeypatch.setattr(factory_mod, "Client", lambda **kw: created.append(kw) or kw)

    m = resilient_model(primary_location="us", backup_location="global")
    assert isinstance(m.primary, LocatedGemini) and isinstance(m.backup, LocatedGemini)

    _ = m.primary.api_client, m.backup.api_client  # build both clients
    primary_kw, backup_kw = created
    assert primary_kw["location"] == "us"
    assert backup_kw["location"] == "global"
    # SDK retries stay off on the primary; backup keeps its exponential retry.
    assert primary_kw["http_options"].retry_options.attempts == 1
    assert backup_kw["http_options"].retry_options.attempts == 5
    # Never a custom base_url: the SDK picks the endpoint from the location.
    assert primary_kw["http_options"].base_url is None
    assert "base_url" not in primary_kw
