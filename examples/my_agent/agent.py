"""Example agent. Works with `adk web`, `adk run my_agent` and Agent Engine.

The only change from a normal ADK agent is the `model=` line.
"""

import logging

from google.adk.agents import LlmAgent

from resilient_gemini import resilient_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

root_agent = LlmAgent(
    name="my_agent",
    model=resilient_model(),  # or resilient_model(primary_thinking="low", backup_thinking="high")
    instruction="You are a helpful assistant.",
)
