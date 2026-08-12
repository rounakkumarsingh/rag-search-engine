from typing import Callable

from cli.lib.chunked_semantic_search import ChunkedSemanticSearch
from cli.lib.config import DEFAULT_SEARCH_LIMIT, SEMANTIC_CANDIDATE_MULTIPLIER
from cli.lib.document import Document
from cli.lib.exceptions import EmptyQueryError
from cli.lib.inverted_index import InvertedIndex
from cli.lib.ranking import (
    RRFResult,
    WeightedResult,
    hybrid_score,
    normalize_scores,
    rrf_score,
)


class HybridSearch:
    def __init__(self, doc_loader: Callable[[], list[Document]]) -> None:
        self.doc_loader = doc_loader
        self.semantic_search = ChunkedSemanticSearch(self.doc_loader)
        self.semantic_search.load_or_create_chunk_embeddings()
        self.idx = InvertedIndex.load_or_build(self.doc_loader)

    def _bm25_search(self, query: str, limit: int) -> list[tuple[float, Document]]:
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = DEFAULT_SEARCH_LIMIT) -> list[WeightedResult]:
        if not query.strip():
            raise EmptyQueryError("Query is empty")
        candidate_limit = limit * SEMANTIC_CANDIDATE_MULTIPLIER
        keyword_results = normalize_scores(self._bm25_search(query, candidate_limit))
        semantic_results = normalize_scores(self.semantic_search.search(query, candidate_limit))

        combined_scores: dict[str, dict[str, float]] = {}
        docs_by_id: dict[str, Document] = {}
        for score, doc in keyword_results:
            doc_id = doc.get_id()
            combined = combined_scores.setdefault(doc_id, {})
            combined["bm25_score"] = score
            docs_by_id[doc_id] = doc
        for score, doc in semantic_results:
            doc_id = doc.get_id()
            combined = combined_scores.setdefault(doc_id, {})
            combined["semantic_score"] = score
            docs_by_id[doc_id] = doc

        results: list[WeightedResult] = []
        for doc_id, scores in combined_scores.items():
            results.append(WeightedResult(
                document=docs_by_id[doc_id],
                bm25_score=scores.get("bm25_score", 0.0),
                semantic_score=scores.get("semantic_score", 0.0),
                hybrid_score=hybrid_score(
                    scores.get("bm25_score", 0.0),
                    scores.get("semantic_score", 0.0),
                    alpha,
                ),
            ))

        results.sort(key=lambda item: item.hybrid_score, reverse=True)
        return results[:limit]

    def rrf_search(self, query: str, k: int, limit: int = DEFAULT_SEARCH_LIMIT) -> list[RRFResult]:
        if not query.strip():
            raise EmptyQueryError("Query is empty")
        candidate_limit = limit * SEMANTIC_CANDIDATE_MULTIPLIER
        bm25_results = self._bm25_search(query, candidate_limit)
        semantic_results = self.semantic_search.search(query, candidate_limit)

        entry_by_id: dict[str, dict] = {}
        for rank, (_, doc) in enumerate(bm25_results, start=1):
            entry = entry_by_id.setdefault(doc.get_id(), {"document": doc})
            entry["bm25_rank"] = rank
        for rank, (_, doc) in enumerate(semantic_results, start=1):
            entry = entry_by_id.setdefault(doc.get_id(), {"document": doc})
            entry["semantic_rank"] = rank

        results: list[RRFResult] = []
        for entry in entry_by_id.values():
            bm25_rank = entry.get("bm25_rank")
            semantic_rank = entry.get("semantic_rank")
            results.append(RRFResult(
                rrf_score=(rrf_score(bm25_rank, k) if bm25_rank is not None else 0.0)
                          + (rrf_score(semantic_rank, k) if semantic_rank is not None else 0.0),
                bm25_rank=bm25_rank,
                semantic_rank=semantic_rank,
                document=entry["document"],
            ))

        results.sort(key=lambda item: item.rrf_score, reverse=True)
        return results[:limit]
