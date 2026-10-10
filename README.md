# 🛡️ resilient-gemini

Drop-in retry, fallback and opt-in Priority PayGo for Google ADK agents on Gemini.

- 🔁 **Primary** `gemini-3.5-flash`: 5 tries, waiting **1, 2, 3, 4 minutes** between them.
- 🛟 **Fallback** `gemini-3.6-flash`: used if all 5 fail, logged as an `ERROR` line starting `FALLBACK:`.
- 🧠 **Thinking level** per model (`minimal`, `low`, `medium`, `high`).
- 🌎 **US multi-region** (`GOOGLE_CLOUD_LOCATION=us`), no custom endpoint needed.
- 💰 **Priority PayGo**: off unless switched on with an env var. **Costs ~2x Standard PayGo.**
- ⚡ Only transient failures are retried (408, 429, 499, 5xx, timeouts, connection errors). Errors like 400 or 403 fail fast.

```python
from google.adk.agents import LlmAgent
from resilient_gemini import resilient_model

root_agent = LlmAgent(name="my_agent", model=resilient_model(), instruction="You are helpful.")
```

---

## 🗂️ Project layout

```
resilient-gemini/
├── pyproject.toml            # uv / hatchling project, deps, dev group
├── .python-version           # 3.12 (uv picks this up)
├── .gitignore
├── README.md
├── CHANGELOG.md
├── CLAUDE.md                 # guidance for AI coding assistants working on this repo
├── src/resilient_gemini/
│   ├── __init__.py           # public API: resilient_model, ResilienceConfig, ...
│   ├── config.py             # settings, env vars, Priority PayGo modes
│   ├── llm.py                # RetryThenFallbackLlm (the retry/fallback logic)
│   └── factory.py            # resilient_model() one-liner
├── examples/
│   ├── run_once.py           # send one prompt, print the answer
│   ├── live_retry_demo.py    # fake errors in front of real Gemini, watch retry/fallback
│   └── my_agent/             # an `adk web` / `adk run` agent folder
│       ├── __init__.py
│       ├── agent.py
│       └── .env.example
└── tests/
    └── test_retry_fallback.py  # no network or credentials needed
```

---

## 🚀 Part 1 — Using the package in your project

### 1. 🐍 Create and activate a virtual environment

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
cd your-project
uv venv                      # creates .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

Without uv:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. 📦 Install

From your team's Git repo (replace URL and tag):

```bash
uv pip install "resilient-gemini @ git+https://github.com/YOUR_ORG/resilient-gemini.git@v0.3.0"
# or, if your project is a uv project:
uv add "resilient-gemini @ git+https://github.com/YOUR_ORG/resilient-gemini.git@v0.3.0"
```

From a private Artifact Registry repo (see "Publishing"):

```bash
uv pip install resilient-gemini==0.3.0 \
  --extra-index-url https://us-python.pkg.dev/YOUR_PROJECT/python-libs/simple/
```

Plain pip works the same way: replace `uv pip` with `pip`.

`google-adk` is a peer dependency: this package does not install it. Your agent
project must already have it (`google-adk>=1.39.1,<2`), e.g. `uv add "google-adk>=1.39.1,<2"`,
or install the `adk` extra to get a supported version with it: `resilient-gemini[adk]`.
Without it, `import resilient_gemini` raises an `ImportError` saying so.

### 3. 🔑 Authenticate and set the environment

```bash
gcloud auth application-default login

export GOOGLE_GENAI_USE_VERTEXAI=TRUE      # or GOOGLE_GENAI_USE_ENTERPRISE=TRUE
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=us
```

Or copy `examples/my_agent/.env.example` to `.env` next to your agent. `adk run` and `adk web` load it automatically.

### 4. ▶️ Run it once

Quickest check, a single prompt:

```bash
python examples/run_once.py "Explain retries in one sentence"
```

As an ADK agent folder:

```bash
adk run examples/my_agent        # chat in the terminal
adk web examples                 # browser UI, pick "my_agent"
```

In your own code, change the `model=` line of any `LlmAgent` to `resilient_model()`.

### 5. 🌎 Confirm the US multi-region endpoint (once)

```python
from resilient_gemini import resilient_model

m = resilient_model()
print(m.primary.api_client._api_client._http_options.base_url)
# expect: https://aiplatform.us.rep.googleapis.com/
```

This reads a private SDK attribute, so use it for debugging only. If it shows `https://us-aiplatform.googleapis.com/`, upgrade `google-genai`.

---

## 🛠️ Part 2 — Developing this package

### 1. 📥 Clone and set up the virtual environment

```bash
git clone https://github.com/YOUR_ORG/resilient-gemini.git
cd resilient-gemini
uv sync                       # creates .venv with Python 3.12, installs the package (editable) + dev tools, incl. google-adk
source .venv/bin/activate     # optional: `uv run ...` works without activating
```

`uv sync` writes `uv.lock`. Commit it so everyone gets the same versions.

### 2. 🧪 Test and lint

```bash
uv run pytest -q              # whole suite, ~1s: no network, no credentials, no real sleeping
uv run ruff check .
uv run ruff format .
```

#### 🎭 How the tests fake failures and timeouts

The unit tests never call Gemini and never wait. Two tricks make that work:

- 🤖 **Fake models.** `FakeLlm` and `ScriptedLlm` in `tests/test_retry_fallback.py` stand in for
  the primary and backup. They raise whatever error the test chooses for the first N calls
  (a 503, a 429, an `httpx.ReadTimeout`, ...) and then answer with their own model name. So
  `["backup"]` as the answer means the fallback ran.
- ⏸️ **Recorded sleeps.** The retry loop waits via `resilient_gemini.llm._sleep`. The `waits`
  fixture swaps that for a function that only records the number of seconds. The full
  60 + 120 + 180 + 240 s schedule runs instantly, and the test asserts on the list,
  e.g. `waits == [60, 120, 180, 240]`.

Run one group at a time with `-k` (add `-v` to see each case, `-rA` to see captured logs):

```bash
# Timeouts and dropped connections (httpx connect/read timeout, connect error, asyncio timeout) are retried
uv run pytest -v -k network_errors

# Every retryable status code (408, 429, 499, 500, 502, 503, 504) is retried
uv run pytest -v -k retryable_code

# 400/401/403/404 raise at once, no retry, no fallback
uv run pytest -v -k "client_errors or non_retryable"

# Five transient failures (503, 429, read timeout, 500, 504) -> 1-2-3-4 minute waits -> backup answers
uv run pytest -v -rA -k full_story

# The linear schedule, a custom schedule, and jitter
uv run pytest -v -k "backoff or schedule or jitter"

# A stream that already produced text is never retried (no duplicated output)
uv run pytest -v -k stream

# The FALLBACK log record carries the fields Cloud Logging alerts use
uv run pytest -v -k alerting_fields

# Priority PayGo, per-model location, config parsing
uv run pytest -v -k "priority or location or env"
```

To fake a new kind of failure, pass it to `ScriptedLlm`:

```python
m = make_scripted(primary_errors=[httpx.ReadTimeout("slow"), api_error(503)])
assert run(m) == ["primary"]  # two failures, then the primary answers on try 3
assert waits == [60, 120]
```

Rules for new tests: use fake models, never real Gemini, and use the `waits` fixture rather
than patching `asyncio.sleep`.

### 3. 🔌 Try your changes against real Gemini

```bash
cp examples/my_agent/.env.example examples/my_agent/.env   # fill in your project
uv run python examples/run_once.py "hello"
uv run adk web examples
```

#### 👀 Watch retries and fallback live

Real 429/5xx errors can't be produced on demand, so `examples/live_retry_demo.py` wraps the
real models in a `FaultyLlm`. It raises fake errors for the first N calls and then sends
the request to real Gemini. Waits are cut to 2 s steps, so a full fallback takes ~25 s
instead of ~10 minutes.

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=your-project-id      # or rely on `gcloud config set project ...`
uv run python examples/live_retry_demo.py        # all scenarios
uv run python examples/live_retry_demo.py 3 7    # pick some
```

| # | Scenario | Expect |
|---|---|---|
| 1 | no faults | primary answers, 1 call |
| 2 | 2 x 503 | waits 2 s, 4 s, then the real primary answers |
| 3 | 5 x 503 | waits 2/4/6/8 s, `FALLBACK:` ERROR, then the real backup answers |
| 4 | 400 | raised at once, backup never called |
| 5 | 429, streaming | one retry, then the primary streams |
| 6 | thinking low/high | fallback log shows `thinking=HIGH` |
| 7 | `backup_location="global"` | fallback answered from the global endpoint |

Each run ends with `SUMMARY:` and exits non-zero if a scenario failed. It makes a few
small real calls and never turns on Priority PayGo. This is a manual check, not part of
`pytest`: `tests/` stays offline.

To see the real schedule against a live model (e.g. under a load test that hits quota),
shorten it instead:

```bash
RESILIENT_GEMINI_MAX_ATTEMPTS=2 RESILIENT_GEMINI_STEP_SECONDS=1 \
  uv run python examples/run_once.py "hello"
```

### 4. 🏷️ Release

1. Bump `version` in `pyproject.toml` and `__version__` in `src/resilient_gemini/__init__.py`.
2. Add a `CHANGELOG.md` entry.
3. Build and tag:

```bash
uv build                      # writes dist/*.whl and dist/*.tar.gz
git commit -am "Release v0.3.0"
git tag v0.3.0 && git push && git push --tags
```

---

## ⚙️ Settings

| Env var | Keyword | Default |
|---|---|---|
| `RESILIENT_GEMINI_PRIMARY_MODEL` | `primary_model` | `gemini-3.5-flash` |
| `RESILIENT_GEMINI_BACKUP_MODEL` | `backup_model` | `gemini-3.6-flash` |
| `RESILIENT_GEMINI_PRIMARY_THINKING` | `primary_thinking` | model default |
| `RESILIENT_GEMINI_BACKUP_THINKING` | `backup_thinking` | model default |
| `RESILIENT_GEMINI_MAX_ATTEMPTS` | `max_attempts` | `5` (total calls to primary) |
| `RESILIENT_GEMINI_STEP_SECONDS` | `step_seconds` | `60` → waits 60, 120, 180, 240 |
| `RESILIENT_GEMINI_STEP_JITTER` | `step_jitter` | `0` (adds random 0..N s to each primary wait) |
| `RESILIENT_GEMINI_PRIMARY_LOCATION` | `primary_location` | unset → `GOOGLE_CLOUD_LOCATION` |
| `RESILIENT_GEMINI_BACKUP_LOCATION` | `backup_location` | unset → `GOOGLE_CLOUD_LOCATION` |
| `RESILIENT_GEMINI_BACKUP_ATTEMPTS` | `backup_attempts` | `5` |
| `RESILIENT_GEMINI_BACKUP_INITIAL_DELAY` | `backup_initial_delay` | `60` |
| `RESILIENT_GEMINI_BACKUP_EXP_BASE` | `backup_exp_base` | `2` |
| `RESILIENT_GEMINI_BACKUP_MAX_DELAY` | `backup_max_delay` | `300` |
| `RESILIENT_GEMINI_BACKUP_JITTER` | `backup_jitter` | `1` (SDK retry jitter) |
| `RESILIENT_GEMINI_PRIORITY_PAYGO` | *(env var only)* | `off` |

Precedence: keyword arguments > env vars > defaults.

---

## 💰 Priority PayGo (opt-in, ~2x cost)

Off by default. It can **only** be turned on with an environment variable; passing it in code raises an error.

| Value | Headers sent | Behaviour |
|---|---|---|
| unset / `off` | none | Standard PayGo |
| `priority` | `X-Vertex-AI-LLM-Request-Type: shared` + `X-Vertex-AI-LLM-Shared-Request-Type: priority` | Priority PayGo only, skips Provisioned Throughput |
| `spillover` | `X-Vertex-AI-LLM-Shared-Request-Type: priority` | Provisioned Throughput first, overflow to Priority PayGo |

Any other value (including `true`) raises an error so nobody pays priority rates by accident.

```bash
export RESILIENT_GEMINI_PRIORITY_PAYGO=priority
```

What gets logged:

```
WARNING resilient_gemini: PRIORITY PAYGO ENABLED via RESILIENT_GEMINI_PRIORITY_PAYGO=priority for gemini-3.5-flash and gemini-3.6-flash. Every request is billed at Priority PayGo rates, ~2x Standard PayGo cost. Headers: {...}
INFO    resilient_gemini: PRIORITY PAYGO (priority): request to gemini-3.5-flash is billed at Priority PayGo rates, ~2x Standard PayGo cost
INFO    resilient_gemini: PRIORITY PAYGO: gemini-3.5-flash served at priority (traffic_type=ON_DEMAND_PRIORITY), billed at Priority PayGo rates, ~2x Standard PayGo cost
WARNING resilient_gemini: PRIORITY PAYGO: gemini-3.5-flash was downgraded to Standard PayGo (traffic_type=ON_DEMAND), billed at the standard rate
```

Google downgrades a request to Standard PayGo only when there is no spare priority capacity; the response's `traffic_type` tells you which happened. Check the [pricing page](https://cloud.google.com/vertex-ai/generative-ai/pricing) for exact rates per model.

---

## 📜 Logs

Logger name: `resilient_gemini`.

```
WARNING resilient_gemini: Primary model gemini-3.5-flash attempt 1/5 failed (503 ...). Retrying in 60s
ERROR   resilient_gemini: FALLBACK: primary model gemini-3.5-flash failed 5 times (last error: ...). Switching to backup model gemini-3.6-flash (thinking=HIGH)
INFO    resilient_gemini: FALLBACK: backup model gemini-3.6-flash responded successfully
```

The fallback record carries `extra` fields (`resilient_gemini_event="fallback"`, `primary_model`, `backup_model`) for log-based alerts in Cloud Logging.

---

## ⚠️ Things to know

- ⏱️ **Worst case before fallback is about 10 minutes** (60+120+180+240 s, plus up to 4 x `step_jitter`). Your server, load balancer or Agent Engine request timeout must allow that, or lower `max_attempts` / `step_seconds`.
- 📍 **Per-model location:** set `backup_location` to a different location than the primary (e.g. primary `us`, backup `global`) so fallback also helps when one location is having trouble. Only the location is passed to the SDK. It picks the endpoint itself, with no custom `base_url`. Needs Vertex AI mode.
- 🌊 **Streaming:** a response that has already started streaming is never retried, so users never see duplicated text.
- 🧠 **Thinking:** don't also set `planner=BuiltInPlanner(thinking_config=...)`. The wrapper sets thinking per call and would overwrite it.
- 🧰 **Built-in tools** that require the agent's model to be a `Gemini` instance (for example Google Search grounding) may refuse the wrapper. Put them on a sub-agent with a plain `Gemini` model, or test them first.

---

## 📤 Publishing to your team

**Option A — Git tag (simplest).** Push to an internal repo and tag releases (see "Release"). Teams install with the `git+https://...@vX.Y.Z` line.

**Option B — Artifact Registry.** One-time:

```bash
gcloud artifacts repositories create python-libs --repository-format=python --location=us
```

Each release:

```bash
uv build
uv publish --publish-url https://us-python.pkg.dev/YOUR_PROJECT/python-libs/ \
  --username oauth2accesstoken --password "$(gcloud auth print-access-token)"
```

Consumers need `roles/artifactregistry.reader` on the repo.
