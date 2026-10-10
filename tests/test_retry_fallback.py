"""Behaviour tests. No network, no credentials: models are replaced by fakes and
sleeping is recorded instead of waited.

Run:  pip install -e ".[test]" && pytest -q
"""

import asyncio
from typing import AsyncGenerator, List, Optional

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import errors, types

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
