import sys
from cli.lib.movies import load_movies
from cli.lib.inverted_index import InvertedIndex
from cli.lib.document import Document
from cli.lib.search_utils import DEFAULT_SEARCH_LIMIT, tokenize_text


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[Document]:
    inverted_index = InvertedIndex(load_movies)
    try:
        inverted_index.load()
    except Exception:
        print("Index not found. Please build index first.")
        sys.exit(1)
    query_tokens = tokenize_text(query)
    result = []
    for query_token in query_tokens:
        if len(result) >= limit:
            break
        docs = inverted_index.get_documents(query_token)
        result.extend(docs)
    return result[:limit]
