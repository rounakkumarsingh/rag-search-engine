"""Tests for reranker abstractions."""
import pytest

from cli.lib.exceptions import GenerationError
from cli.lib.ranking import RRFResult
from cli.lib.rerankers import BatchLlmReranker, IndividualLlmReranker, make_reranker


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def _results(docs, base=0.5):
    return [RRFResult(rrf_score=base, document=doc) for doc in docs]


def test_individual_sorts_by_score(fake_docs):
    llm = FakeLLM(["3", "10", "7"])
    reranked = IndividualLlmReranker(llm, delay=0.0).rerank("query", _results(fake_docs))
    assert [r.rr_score for r in reranked] == [10.0, 7.0, 3.0]
    assert llm.calls == 3


def test_individual_falls_back_to_rrf_order_on_failure(fake_docs):
    llm = FakeLLM(["10", GenerationError("boom"), "8"])
    reranked = IndividualLlmReranker(llm, delay=0.0).rerank("query", _results(fake_docs, base=0.9))
    scored = [r for r in reranked if r.rr_score is not None]
    failed = [r for r in reranked if r.rr_score is None]
    assert [r.rr_score for r in scored] == [10.0, 8.0]
    # The failed document is relegated below scored ones, not dropped.
    assert len(failed) == 1
    assert failed[0].document is fake_docs[1]


def test_individual_handles_non_numeric_response(fake_docs):
    llm = FakeLLM(["not a number"])
    reranked = IndividualLlmReranker(llm, delay=0.0).rerank("query", _results(fake_docs[:1]))
    assert reranked[0].rr_score is None


def test_batch_reorders_by_ids(fake_docs):
    llm = FakeLLM(["[2, 0, 1]"])
    reranked = BatchLlmReranker(llm).rerank("query", _results(fake_docs))
    assert [r.document.get_id() for r in reranked] == ["2", "0", "1"]
    assert [r.rr_rank for r in reranked] == [1, 2, 3]


def test_batch_invalid_json_returns_empty(fake_docs):
    llm = FakeLLM(["not json"])
    reranked = BatchLlmReranker(llm).rerank("query", _results(fake_docs))
    assert reranked == []


def test_make_reranker_requires_llm_for_individual():
    with pytest.raises(ValueError):
        make_reranker("individual", None)


def test_make_reranker_none_without_llm():
    assert make_reranker("none", None) is None
