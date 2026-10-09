"""Plug-and-play retry + fallback for Google ADK Gemini agents.

    from google.adk.agents import LlmAgent
    from resilient_gemini import resilient_model

    root_agent = LlmAgent(name="my_agent", model=resilient_model(), instruction="...")
"""

from .config import (
    PriorityMode,
    ResilienceConfig,
    parse_priority_mode,
    parse_thinking_level,
)
from .factory import resilient_model
from .llm import RetryThenFallbackLlm, is_retryable

__version__ = "0.2.0"

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
