import os

from dotenv import load_dotenv
from openai import OpenAI

from cli.lib.config import LLM_API_KEY_ENV, LLM_BASE_URL, LLM_DEFAULT_MODEL
from cli.lib.document import Document
from cli.lib.exceptions import GenerationError


def create_openai_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get(LLM_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{LLM_API_KEY_ENV} environment variable not set")
    return OpenAI(base_url=LLM_BASE_URL, api_key=api_key)


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


def spell_fix_prompt(query: str) -> str:
    return f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""


def rewrite_query_prompt(query: str) -> str:
    return f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""


def expand_query_prompt(query: str) -> str:
    return f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""


def rerank_single_prompt(query: str, document: Document) -> str:
    return f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {document.get_title()} - {document.get_description()}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""


def rerank_batch_prompt(query: str, doc_list_str: str) -> str:
    return f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""
