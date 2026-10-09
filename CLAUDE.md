# CLAUDE.md

Guidance for Claude (and other AI coding assistants) working in this repository.

## What this project is

`resilient-gemini` is a small Python library for Google ADK agents. It wraps two
Gemini models in one ADK `BaseLlm`:

1. **Primary** (`gemini-3.5-flash`): retried with **linear** backoff, waits of
   `step_seconds * attempt` (60, 120, 180, 240 s by default, 5 total calls).
2. **Backup** (`gemini-3.6-flash`): called once if the primary keeps failing with
   transient errors. It uses the google-genai SDK's own exponential retry.

It also sets a thinking level per model and can opt in to Priority PayGo.
Teams use it with one line: `LlmAgent(model=resilient_model(), ...)`.

## Layout

- `src/resilient_gemini/config.py`: `ResilienceConfig`, env var parsing, thinking
  level parsing, `PriorityMode` and the Priority PayGo headers.
- `src/resilient_gemini/llm.py`: `RetryThenFallbackLlm`, the retry/fallback loop,
  `is_retryable`, and all retry/fallback/priority logging.
- `src/resilient_gemini/factory.py`: `resilient_model()`, which builds the two
  `Gemini` instances and the wrapper.
- `src/resilient_gemini/__init__.py`: public API and `__version__`.
- `examples/`: `run_once.py` (single prompt) and `my_agent/` (an `adk web` agent).
- `tests/test_retry_fallback.py`: behaviour tests with fake models.

## Commands

Always use uv and the project virtual env.

```bash
uv sync                 # create/refresh .venv, install package (editable) + dev deps
uv run pytest -q        # run tests
uv run ruff check .     # lint
uv run ruff format .    # format
uv build                # build wheel + sdist into dist/
```

Run `uv run pytest -q` and `uv run ruff check .` before saying a change is done.

## Rules that must not be broken

- **Priority PayGo is opt-in and env-var only.** It is enabled solely by
  `RESILIENT_GEMINI_PRIORITY_PAYGO=priority|spillover`. Do not add a keyword
  argument, default, or config path that turns it on. The parser must stay strict:
  ambiguous values such as `true` or `yes` raise `ValueError`. It costs ~2x
  Standard PayGo, so every enabled path must log that.
- **Tests never touch the network or need credentials.** Use the `FakeLlm` pattern.
  Never call real Gemini in `tests/`.
- **Never sleep for real in tests.** The loop calls `llm._sleep`; tests
  monkeypatch that. Do not patch `asyncio.sleep` globally, and keep the `_sleep`
  indirection when editing `llm.py`.
- **Only transient errors are retried**: status codes in `retryable_codes`
  (408, 429, 499, 500, 502, 503, 504) plus httpx timeouts/connection errors.
  Client errors (400, 401, 403, 404) must raise immediately with no fallback.
- **Never retry a stream that has already yielded output**, or users get
  duplicated text.
- **The primary's SDK retries stay off** (`HttpRetryOptions(attempts=1)`) so the
  wrapper alone controls the 1, 2, 3, 4 minute schedule.
- **Do not set a custom `base_url`.** The US multi-region endpoint comes from
  `GOOGLE_CLOUD_LOCATION=us` and a current `google-genai`.
- **Keep log message prefixes stable**: `FALLBACK:` and `PRIORITY PAYGO`. Teams
  build Cloud Logging alerts on them and on the fallback record's `extra` fields
  (`resilient_gemini_event`, `primary_model`, `backup_model`). Logger name is
  `resilient_gemini`.

## Conventions

- Python 3.10+, type hints everywhere, `from __future__ import annotations`.
- New settings go in `ResilienceConfig` with a `RESILIENT_GEMINI_*` env var, a
  keyword override, a row in the README settings table, and a test.
- Model IDs and defaults live only in `config.py`.
- Public names must be exported from `__init__.py`.
- Behaviour changes need a test in `tests/` and a `CHANGELOG.md` entry.

## Releasing

1. Bump `version` in `pyproject.toml` **and** `__version__` in
   `src/resilient_gemini/__init__.py` (keep them equal).
2. Add a `CHANGELOG.md` entry.
3. `uv run pytest -q && uv build`
4. Commit, tag `vX.Y.Z`, push tags. Publishing steps are in the README.

## Facts worth re-checking before changing related code

Google's model IDs, Priority PayGo headers and pricing, and multi-region
behaviour change over time. Check the current Google Cloud docs rather than
relying on memory:

- Priority PayGo: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/priority-paygo
- Endpoints / locations: https://docs.cloud.google.com/gemini-enterprise-agent-platform/resources/locations
- Thinking levels: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/thinking
