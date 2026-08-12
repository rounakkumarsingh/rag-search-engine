import os

from dotenv import load_dotenv
from openai import OpenAI

from cli.lib.config import (
    LLM_API_KEY_ENV,
    LLM_BASE_URL,
    LLM_DEFAULT_MODEL,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT,
)
from cli.lib.exceptions import GenerationError


def create_openai_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get(LLM_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{LLM_API_KEY_ENV} environment variable not set")
    return OpenAI(
        base_url=LLM_BASE_URL,
        api_key=api_key,
        timeout=LLM_TIMEOUT,
        max_retries=LLM_MAX_RETRIES,
    )


class LLMWrapper:
    def __init__(self, model: str = LLM_DEFAULT_MODEL, client: OpenAI | None = None):
        self.client = client if client is not None else create_openai_client()
        self.model = model

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            raise GenerationError(f"LLM request failed: {exc}") from exc
        content = response.choices[0].message.content
        if content is None or not content.strip():
            raise GenerationError("LLM returned empty or null content")
        return content
