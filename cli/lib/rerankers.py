import json
import time
from dataclasses import replace
from typing import Protocol

from cli.lib.document import Document
from cli.lib.llm import LLMWrapper
from cli.lib.models import get_cross_encoder
from cli.lib.prompts import rerank_batch_prompt, rerank_single_prompt
from cli.lib.ranking import RRFResult


class Reranker(Protocol):
    def rerank(self, query: str, results: list[RRFResult]) -> list[RRFResult]: ...


class IndividualLlmReranker:
    def __init__(self, llm: LLMWrapper, delay: float = 3.0) -> None:
        self.llm = llm
        self.delay = delay

    def rerank(self, query: str, results: list[RRFResult]) -> list[RRFResult]:
        reranked: list[RRFResult] = []
        for result in results:
            response = self.llm.generate(rerank_single_prompt(query, result.document))
            try:
                score = float(response.strip())
            except ValueError:
                score = 0.0
            reranked.append(replace(result, rr_score=score))
            if self.delay:
                time.sleep(self.delay)
        reranked.sort(key=lambda item: item.rr_score, reverse=True)
        return reranked


class BatchLlmReranker:
    def __init__(self, llm: LLMWrapper) -> None:
        self.llm = llm

    def rerank(self, query: str, results: list[RRFResult]) -> list[RRFResult]:
        doc_list_str = "\n".join(
            f"{result.document.get_id()}: {result.document.get_title()} - {result.document.get_description()}"
            for result in results
        )
        response = self.llm.generate(rerank_batch_prompt(query, doc_list_str))
        try:
            ranked_ids = json.loads(response.strip())
        except json.JSONDecodeError:
            ranked_ids = []

        docs_by_id: dict[str, RRFResult] = {result.document.get_id(): result for result in results}
        reranked: list[RRFResult] = []
        for new_rank, doc_id in enumerate(ranked_ids):
            result = docs_by_id.get(str(doc_id))
            if result is None:
                continue
            reranked.append(replace(result, rr_rank=new_rank + 1))
        return reranked


class CrossEncoderReranker:
    def __init__(self) -> None:
        self.model = get_cross_encoder()

    def rerank(self, query: str, results: list[RRFResult]) -> list[RRFResult]:
        pairs: list[list[str]] = [[query, result.document.to_text()] for result in results]
        scores = self.model.predict(pairs)
        reranked = [
            replace(result, rr_score=float(score))
            for result, score in zip(results, scores)
        ]
        reranked.sort(key=lambda item: item.rr_score, reverse=True)
        return reranked


def make_reranker(method: str, llm: LLMWrapper | None = None) -> Reranker | None:
    if method == "individual":
        if llm is None:
            raise ValueError("IndividualLlmReranker requires an LLMWrapper")
        return IndividualLlmReranker(llm)
    if method == "batch":
        if llm is None:
            raise ValueError("BatchLlmReranker requires an LLMWrapper")
        return BatchLlmReranker(llm)
    if method == "cross_encoder":
        return CrossEncoderReranker()
    return None
