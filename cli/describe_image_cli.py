import argparse
import base64
import mimetypes
import os

from dotenv import load_dotenv
from openai import OpenAI

from cli.lib.config import LLM_API_KEY_ENV, LLM_BASE_URL, LLM_DEFAULT_MODEL

SYSTEM_PROMPT = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe an image and rewrite a text query")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--query", type=str, required=True, help="Text query to rewrite based on the image")
    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, "rb") as f:
        img = f.read()

    load_dotenv()
    api_key = os.environ[LLM_API_KEY_ENV]
    client = OpenAI(base_url=LLM_BASE_URL, api_key=api_key)

    data_url = f"data:{mime};base64,{base64.b64encode(img).decode()}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT.strip()},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": args.query.strip()},
            ],
        }
    ]

    response = client.chat.completions.create(model=LLM_DEFAULT_MODEL, messages=messages)

    content = response.choices[0].message.content
    print(f"Rewritten query: {content.strip()}")
    if response.usage is not None:
        print(f"Total tokens:    {response.usage.total_tokens}")


if __name__ == "__main__":
    main()
