import os

from cli.lib.config import LLM_API_KEY_ENV, LLM_DEFAULT_MODEL
from cli.lib.llm import create_openai_client


def main() -> None:
    if not os.environ.get(LLM_API_KEY_ENV):
        raise RuntimeError(f"{LLM_API_KEY_ENV} environment variable not set; set it to use LLM features")

    client = create_openai_client()
    messages = [
        {
            "role": "user",
            "content": "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
        }
    ]
    response = client.chat.completions.create(model=LLM_DEFAULT_MODEL, messages=messages)

    print(response.choices[0].message.content)
    print(f"Prompt tokens: {response.usage.prompt_tokens}")
    print(f"Response tokens: {response.usage.completion_tokens}")


if __name__ == "__main__":
    main()
