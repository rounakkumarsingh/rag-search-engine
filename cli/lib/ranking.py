from dataclasses import dataclass, field
from typing import Final

from cli.lib.document import Document

RRF_K: Final = 60


@dataclass(frozen=True)
class ScoredResult:
    score: float
    document: Document


@dataclass(frozen=True)
class WeightedResult:
    hybrid_score: float
    document: Document
    bm25_score: float = 0.0
    semantic_score: float = 0.0


@dataclass(frozen=True)
class RRFResult:
    rrf_score: float
    document: Document
    bm25_rank: int | None = None
    semantic_rank: int | None = None
    rr_score: float | None = field(default=None, compare=False, hash=False)
    rr_rank: int | None = field(default=None, compare=False, hash=False)


def rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank)


def normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    return [(score - min_score) / (max_score - min_score) for score in scores]


def normalize_scores(results: list[tuple[float, Document]]) -> list[tuple[float, Document]]:
    scores = [score for score, _ in results]
    documents = [doc for _, doc in results]
    new_scores = normalize(scores)
    return list(zip(new_scores, documents))


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score
