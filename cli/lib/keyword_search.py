from cli.lib.movies import load_movies
from cli.lib.inverted_index import InvertedIndex
from cli.lib.document import Document
from cli.lib.search_utils import DEFAULT_SEARCH_LIMIT, BM25_K1, BM25_B, tokenize_text, tokenize_single_term


def _load_index() -> InvertedIndex:
    return InvertedIndex.load_or_build(load_movies)


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[Document]:
    inverted_index = _load_index()
    query_tokens = tokenize_text(query)
    result = []
    for query_token in query_tokens:
        if len(result) >= limit:
            break
        docs = inverted_index.get_documents(query_token)
        result.extend(docs)
    return result[:limit]


def bm25_tf_command(doc_id: str, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    inverted_index = _load_index()
    token = tokenize_single_term(term)
    return inverted_index.get_bm25_tf(doc_id, token, k1, b)


def bm25_search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[tuple[float, Document]]:
    inverted_index = _load_index()
    return inverted_index.bm25_search(query, limit)
