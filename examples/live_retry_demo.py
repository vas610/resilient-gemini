"""Watch retries and fallback happen against real Gemini.

Real 429/5xx errors can't be produced on demand, so this wraps the real primary
and backup models in FaultyLlm: it raises fake API errors for the first N calls,
then forwards to the real model. Waits are shortened to 2s steps.

    uv run python examples/live_retry_demo.py          # all scenarios
    uv run python examples/live_retry_demo.py 2 3      # just some

Needs GOOGLE_CLOUD_PROJECT (or a gcloud default project) and
`gcloud auth application-default login`. Makes a handful of small real calls.
Not a unit test: tests/ must never touch the network.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.genai import errors, types

from resilient_gemini import resilient_model

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us")
os.environ.pop("RESILIENT_GEMINI_PRIORITY_PAYGO", None)  # never pay priority rates in a demo
if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
    project = subprocess.run(
        ["gcloud", "config", "get-value", "project"], capture_output=True, text=True, check=False
    ).stdout.strip()
    if not project:
        sys.exit("Set GOOGLE_CLOUD_PROJECT or a gcloud default project.")
    os.environ["GOOGLE_CLOUD_PROJECT"] = project

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "google_genai", "google_adk"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


class FaultyLlm(BaseLlm):
    """Raises a fake ``error_code`` API error for the first ``fail_times`` calls."""

    inner: BaseLlm
    fail_times: int = 0
    error_code: int = 503
    calls: int = 0
    real_calls: int = 0

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False):
        self.calls += 1
        if self.calls <= self.fail_times:
            cls = errors.ServerError if self.error_code >= 500 else errors.ClientError
            raise cls(self.error_code, {"error": {"code": self.error_code, "message": "injected"}})
        self.real_calls += 1
        async for resp in self.inner.generate_content_async(llm_request, stream):
            yield resp


def build(primary_fails: int = 0, code: int = 503, **kw):
    m = resilient_model(step_seconds=2, **kw)
    m.primary = FaultyLlm(model=m.primary.model, inner=m.primary, fail_times=primary_fails, error_code=code)
    m.backup = FaultyLlm(model=m.backup.model, inner=m.backup)
    return m


async def ask(m, stream: bool) -> str:
    req = LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text="Reply with one short sentence.")])],
        config=types.GenerateContentConfig(),
    )
    text = []
    async for resp in m.generate_content_async(req, stream):
        if resp.partial or not (resp.content and resp.content.parts):
            continue  # streamed chunks are repeated in the final aggregated response
        text.extend(p.text or "" for p in resp.content.parts if not p.thought)
    return "".join(text).strip()


SCENARIOS = {
    "1": ("no faults: primary answers", {}, None, False),
    "2": ("2x 503 -> retry -> primary answers", {"primary_fails": 2}, None, False),
    "3": ("5x 503 -> FALLBACK -> backup answers", {"primary_fails": 99}, None, False),
    "4": ("400 -> raised at once, no fallback", {"primary_fails": 1, "code": 400}, errors.ClientError, False),
    "5": ("429 then streaming -> retry -> primary streams", {"primary_fails": 1, "code": 429}, None, True),
    "6": (
        "thinking low/high, fallback",
        {"primary_fails": 99, "primary_thinking": "low", "backup_thinking": "high"},
        None,
        False,
    ),
    "7": (
        "backup pinned to 'global' location, fallback",
        {"primary_fails": 99, "backup_location": "global"},
        None,
        False,
    ),
}


async def main(selected: list[str]) -> int:
    results = {}
    for key, (name, kw, expect_exc, stream) in SCENARIOS.items():
        if selected and key not in selected:
            continue
        print(f"\n===== {key}. {name} =====", flush=True)
        m = build(**kw)
        start = time.monotonic()
        try:
            outcome = f"answer: {(await ask(m, stream))[:80]!r}"
            ok = expect_exc is None
        except Exception as exc:  # noqa: BLE001  (report any failure as a scenario result)
            outcome = f"raised {type(exc).__name__}: {str(exc)[:120]}"
            ok = expect_exc is not None and isinstance(exc, expect_exc)
        print(
            f"-> {outcome}\n   primary calls={m.primary.calls} (real {m.primary.real_calls}), "
            f"backup calls={m.backup.calls} (real {m.backup.real_calls}), "
            f"{time.monotonic() - start:.1f}s, {'PASS' if ok else 'FAIL'}",
            flush=True,
        )
        results[key] = ok
    print("\nSUMMARY:", ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
