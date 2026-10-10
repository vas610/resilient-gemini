"""Configuration for resilient Gemini models.

Every setting can come from code (keyword arguments) or from environment
variables, so teams can tune behaviour per environment without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from enum import Enum

from google.genai import types

ThinkingInput = str | types.ThinkingLevel | None

DEFAULT_PRIMARY_MODEL = "gemini-3.5-flash"
DEFAULT_BACKUP_MODEL = "gemini-3.6-flash"

# Transient failures worth retrying. 499 (CANCELLED) is not in the
# google-genai default list, so it is added explicitly.
DEFAULT_RETRYABLE_CODES = frozenset({408, 429, 499, 500, 502, 503, 504})

_ENV_PREFIX = "RESILIENT_GEMINI_"

# ---------------------------------------------------------------------------
# Priority PayGo (opt-in, env var only)
# https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/priority-paygo
# ---------------------------------------------------------------------------
PRIORITY_PAYGO_ENV = _ENV_PREFIX + "PRIORITY_PAYGO"


class PriorityMode(str, Enum):
    OFF = "off"
    # Priority PayGo only (bypasses Provisioned Throughput).
    PRIORITY = "priority"
    # Use Provisioned Throughput first, spill over to Priority PayGo.
    SPILLOVER = "spillover"


PRIORITY_HEADERS = {
    PriorityMode.PRIORITY: {
        "X-Vertex-AI-LLM-Request-Type": "shared",
        "X-Vertex-AI-LLM-Shared-Request-Type": "priority",
    },
    PriorityMode.SPILLOVER: {
        "X-Vertex-AI-LLM-Shared-Request-Type": "priority",
    },
}

PRIORITY_COST_NOTE = "billed at Priority PayGo rates, ~2x Standard PayGo cost"


def parse_priority_mode(value: str | None) -> PriorityMode:
    """Strict parser: only the exact words 'priority' or 'spillover' turn it on.

    Anything ambiguous (e.g. 'true', 'yes', typos) raises instead of guessing,
    so nobody pays priority rates by accident.
    """
    text = (value or "").strip().lower()
    if text in ("", "off", "false", "0", "no"):
        return PriorityMode.OFF
    if text == PriorityMode.PRIORITY.value:
        return PriorityMode.PRIORITY
    if text == PriorityMode.SPILLOVER.value:
        return PriorityMode.SPILLOVER
    raise ValueError(
        f"{PRIORITY_PAYGO_ENV}={value!r} is not valid. Use 'priority' (Priority PayGo only), "
        "'spillover' (Provisioned Throughput first, then Priority PayGo), or 'off'."
    )


def parse_thinking_level(value: ThinkingInput) -> types.ThinkingLevel | None:
    """Turn 'low' / 'HIGH' / ThinkingLevel.MEDIUM / None into a ThinkingLevel.

    None, '' or 'default' mean "don't set it; use the model's default".
    """
    if value is None or isinstance(value, types.ThinkingLevel):
        return value
    text = str(value).strip().upper()
    if text in ("", "DEFAULT", "NONE"):
        return None
    try:
        return types.ThinkingLevel[text]
    except KeyError as exc:
        allowed = ", ".join(m.name for m in types.ThinkingLevel if m.name != "THINKING_LEVEL_UNSPECIFIED")
        raise ValueError(f"Unknown thinking level {value!r}. Use one of: {allowed}") from exc


@dataclass(frozen=True)
class ResilienceConfig:
    """All knobs for retry + fallback behaviour.

    Retry schedule for the primary model is linear: the wait before retry N is
    ``step_seconds * N`` (60s, 120s, 180s, 240s with the defaults).
    """

    primary_model: str = DEFAULT_PRIMARY_MODEL
    backup_model: str = DEFAULT_BACKUP_MODEL
    primary_thinking: types.ThinkingLevel | None = None
    backup_thinking: types.ThinkingLevel | None = None

    # Vertex AI location per model, e.g. "us" or "global". None means use
    # GOOGLE_CLOUD_LOCATION like a plain ADK Gemini model.
    primary_location: str | None = None
    backup_location: str | None = None

    # Primary: total calls (first try included) and linear step between them.
    # step_jitter adds a random 0..step_jitter seconds to each wait (0 = exact schedule).
    max_attempts: int = 5
    step_seconds: float = 60.0
    step_jitter: float = 0.0

    # Backup: the SDK's own exponential backoff.
    backup_attempts: int = 5
    backup_initial_delay: float = 60.0
    backup_exp_base: float = 2.0
    backup_max_delay: float = 300.0
    backup_jitter: float = 1.0

    retryable_codes: frozenset[int] = field(default=DEFAULT_RETRYABLE_CODES)

    # Set ONLY from the RESILIENT_GEMINI_PRIORITY_PAYGO env var (see from_env).
    priority_paygo: PriorityMode = PriorityMode.OFF

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.step_seconds < 0:
            raise ValueError("step_seconds must be >= 0")
        if self.backup_attempts < 1:
            raise ValueError("backup_attempts must be >= 1")
        if self.step_jitter < 0 or self.backup_jitter < 0:
            raise ValueError("step_jitter and backup_jitter must be >= 0")

    @classmethod
    def from_env(cls, **overrides) -> ResilienceConfig:
        """Build a config from RESILIENT_GEMINI_* env vars, then apply overrides."""
        env = os.environ

        def get(name: str, default):
            return env.get(_ENV_PREFIX + name, default)

        cfg = cls(
            primary_model=get("PRIMARY_MODEL", DEFAULT_PRIMARY_MODEL),
            backup_model=get("BACKUP_MODEL", DEFAULT_BACKUP_MODEL),
            primary_thinking=parse_thinking_level(get("PRIMARY_THINKING", None)),
            backup_thinking=parse_thinking_level(get("BACKUP_THINKING", None)),
            primary_location=get("PRIMARY_LOCATION", "").strip() or None,
            backup_location=get("BACKUP_LOCATION", "").strip() or None,
            max_attempts=int(get("MAX_ATTEMPTS", 5)),
            step_seconds=float(get("STEP_SECONDS", 60)),
            step_jitter=float(get("STEP_JITTER", 0)),
            backup_attempts=int(get("BACKUP_ATTEMPTS", 5)),
            backup_initial_delay=float(get("BACKUP_INITIAL_DELAY", 60)),
            backup_exp_base=float(get("BACKUP_EXP_BASE", 2)),
            backup_max_delay=float(get("BACKUP_MAX_DELAY", 300)),
            backup_jitter=float(get("BACKUP_JITTER", 1)),
            priority_paygo=parse_priority_mode(env.get(PRIORITY_PAYGO_ENV)),
        )
        return cfg.with_overrides(**overrides)

    def with_overrides(self, **overrides) -> ResilienceConfig:
        """Return a copy with some fields changed. Thinking levels accept strings."""
        if not overrides:
            return self
        if "priority_paygo" in overrides:
            raise ValueError(
                f"Priority PayGo can only be enabled with the {PRIORITY_PAYGO_ENV} "
                "environment variable, not in code."
            )
        for key in ("primary_thinking", "backup_thinking"):
            if key in overrides:
                overrides[key] = parse_thinking_level(overrides[key])
        if "retryable_codes" in overrides:
            overrides["retryable_codes"] = frozenset(overrides["retryable_codes"])
        return replace(self, **overrides)
