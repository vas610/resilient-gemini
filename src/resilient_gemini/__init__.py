"""Plug-and-play retry + fallback for Google ADK Gemini agents.

    from google.adk.agents import LlmAgent
    from resilient_gemini import resilient_model

    root_agent = LlmAgent(name="my_agent", model=resilient_model(), instruction="...")
"""

try:
    import google.adk  # noqa: F401  (peer dependency, not installed by this package)
except ImportError as exc:  # pragma: no cover - exercised in tests via sys.modules
    raise ImportError(
        "resilient_gemini needs google-adk, which it does not install for you. "
        "Add it to your agent project, e.g. `uv add 'google-adk>=1.39.1,<2'` "
        "or install this package with the extra: `pip install 'resilient-gemini[adk]'`."
    ) from exc

from .config import (
    PriorityMode,
    ResilienceConfig,
    parse_priority_mode,
    parse_thinking_level,
)
from .factory import resilient_model
from .llm import RetryThenFallbackLlm, is_retryable

__version__ = "0.3.0"

__all__ = [
    "PriorityMode",
    "ResilienceConfig",
    "RetryThenFallbackLlm",
    "is_retryable",
    "parse_priority_mode",
    "parse_thinking_level",
    "resilient_model",
    "__version__",
]
