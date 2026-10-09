"""Send one prompt through a resilient agent and print the answer.

    python examples/run_once.py "Explain retries in one sentence"

Needs GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION
set (see examples/my_agent/.env.example) and `gcloud auth application-default login`.
"""

import asyncio
import logging
import sys

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

from resilient_gemini import resilient_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

APP = "resilient_demo"
USER = "local-user"


async def main(prompt: str) -> None:
    agent = LlmAgent(
        name="demo_agent",
        model=resilient_model(),
        instruction="You are a helpful assistant. Answer briefly.",
    )
    runner = InMemoryRunner(agent=agent, app_name=APP)
    session = await runner.session_service.create_session(app_name=APP, user_id=USER)

    message = types.Content(role="user", parts=[types.Part(text=prompt)])
    async for event in runner.run_async(user_id=USER, session_id=session.id, new_message=message):
        if event.is_final_response() and event.content and event.content.parts:
            print("\n" + "".join(p.text or "" for p in event.content.parts))


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "Say hello in one sentence."))
