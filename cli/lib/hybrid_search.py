from collections import defaultdict
from cli.lib.config import INDEX_CACHE_PATH
from cli.lib.chunked_semantic_search import ChunkedSemanticSearch
from cli.lib.document import Document
from typing import Callable, TypedDict
from cli.lib.inverted_index import InvertedIndex


def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    return [(score - min_score) / (max_score - min_score) for score in scores]


def normalize_scores(results: list[tuple[float, Document]]):
    scores = [score for score, _ in results]
    documents = [doc for _, doc in results]
    new_scores = normalize(scores)
    return list(zip(new_scores, documents))


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


class CombinedScores(TypedDict):
    semantic_score: float | None
    bm25_score: float | None
    hybrid_score: float


class RRFResults(TypedDict):
    bm25_rank: int | None
    semantic_rank: int | None
    rrf_score: float


class HybridSearch:
    def __init__(self, doc_loader: Callable[[], list[Document]]) -> None:
        self.doc_loader = doc_loader
        self.semantic_search = ChunkedSemanticSearch(self.doc_loader)
        self.semantic_search.load_or_create_chunk_embeddings()

        self.idx = InvertedIndex(self.doc_loader)
        if not INDEX_CACHE_PATH.exists():
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[tuple[CombinedScores, Document]]:
        keywords_search_results = self._bm25_search(query, limit * 500)
        semantic_search_results = self.semantic_search.search(query, 500 * limit)

        # score normalization from 0 - 1 using Min-Max Normalization
        keywords_search_results = normalize_scores(keywords_search_results)
        semantic_search_results = normalize_scores(semantic_search_results)

        # combining scores
        combined_scores: dict[str, CombinedScores] = defaultdict(dict)
        docs_by_id: dict[str, Document] = {}
        for keyword_score, keyword_doc in keywords_search_results:
            doc_id = keyword_doc.get_id()
            combined_scores[doc_id]["bm25_score"] = keyword_score
            docs_by_id[doc_id] = keyword_doc

        for semantic_score, semantic_doc in semantic_search_results:
            doc_id = semantic_doc.get_id()
            combined_scores[doc_id]["semantic_score"] = semantic_score
            docs_by_id[doc_id] = semantic_doc

        ranked: list[tuple[CombinedScores, Document]] = []
        for doc_id, scores in combined_scores.items():
            scores["hybrid_score"] = hybrid_score(scores["bm25_score"] or 0, scores["semantic_score"] or 0, alpha)
            ranked.append((scores, docs_by_id[doc_id]))

        ranked.sort(key=lambda item: item[0]["hybrid_score"], reverse=True)
        return ranked[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[tuple[RRFResults, Document]]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search(query, 500 * limit)

        # map doc id -> document, the document's BM25/semantic ranks, and the summed RRF score
        rank_by_id: dict[str, dict] = defaultdict(dict)
        for rank, (_, doc) in enumerate(bm25_results, start=1):
            entry = rank_by_id[doc.get_id()]
            entry.setdefault("document", doc)
            entry["bm25_rank"] = rank
        for rank, (_, doc) in enumerate(semantic_results, start=1):
            entry = rank_by_id[doc.get_id()]
            entry.setdefault("document", doc)
            entry["semantic_rank"] = rank

        # compute the summed RRF score for each document
        results: list[tuple[RRFResults, Document]] = []
        for entry in rank_by_id.values():
            rrf = (rrf_score(entry["bm25_rank"], k) if "bm25_rank" in entry else 0.0) + \
                  (rrf_score(entry["semantic_rank"], k) if "semantic_rank" in entry else 0.0)
            ranks: RRFResults = {
                "bm25_rank": entry.get("bm25_rank"),
                "semantic_rank": entry.get("semantic_rank"),
                "rrf_score": rrf,
            }
            results.append((ranks, entry["document"]))

        results.sort(key=lambda item: item[0]["rrf_score"], reverse=True)
        return results[:limit]
