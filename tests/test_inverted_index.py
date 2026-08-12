"""Tests for the inverted index lifecycle and exception boundary."""
import pytest

import cli.lib.inverted_index as ii
import cli.lib.search_utils as su
from cli.lib.exceptions import EmptyQueryError, IndexNotFoundError


def _patch_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(ii, "INDEX_CACHE_PATH", tmp_path / "index.pkl")
    monkeypatch.setattr(ii, "DOCMAP_CACHE_PATH", tmp_path / "docmap.pkl")
    monkeypatch.setattr(ii, "TERM_FREQUENCIES_CACHE_PATH", tmp_path / "term_frequencies.pkl")
    monkeypatch.setattr(ii, "DOC_LENGTHS_CACHE_PATH", tmp_path / "doc_lengths.pkl")
    monkeypatch.setattr(
        ii, "INDEX_ARTIFACTS",
        [tmp_path / "index.pkl", tmp_path / "docmap.pkl",
         tmp_path / "term_frequencies.pkl", tmp_path / "doc_lengths.pkl"],
    )


def test_load_raises_index_not_found(monkeypatch, tmp_path, fake_docs):
    _patch_paths(monkeypatch, tmp_path)
    with pytest.raises(IndexNotFoundError):
        ii.InvertedIndex(lambda: fake_docs).load()


def test_load_or_build_persists_and_reloads(monkeypatch, tmp_path, fake_docs):
    _patch_paths(monkeypatch, tmp_path)
    built = ii.InvertedIndex.load_or_build(lambda: fake_docs)
    assert len(built.docmap) == len(fake_docs)

    loaded = ii.InvertedIndex(lambda: fake_docs)
    loaded.load()
    assert list(loaded.docmap.keys()) == list(fake_docs[i].get_id() for i in range(len(fake_docs)))


def test_bm25_search_raises_empty_query(monkeypatch, tmp_path, fake_docs):
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(su, "STOPWORDS", ["the", "and"])
    idx = ii.InvertedIndex.from_documents(fake_docs)
    with pytest.raises(EmptyQueryError):
        idx.bm25_search("the and")


def test_from_documents_returns_ranked_docs(fake_docs):
    idx = ii.InvertedIndex.from_documents(fake_docs)
    results = idx.bm25_search("bear")
    assert results
    assert all(isinstance(doc, type(fake_docs[0])) for _, doc in results)
