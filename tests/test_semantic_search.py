"""Tests for the whole-document embedding cache lifecycle (fingerprint-aware)."""
import json

import numpy as np
import pytest

import cli.lib.semantic_search as sem
from cli.lib.exceptions import CacheInvalidError

FINGERPRINT = "abc123"


def _patch_config(monkeypatch, tmp_path):
    monkeypatch.setattr(sem, "EMBEDDINGS_CACHE_PATH", tmp_path / "embeddings.npy")
    monkeypatch.setattr(sem, "EMBEDDINGS_META_CACHE_PATH", tmp_path / "embeddings.meta.json")
    monkeypatch.setattr(sem, "source_fingerprint", lambda: FINGERPRINT)


def test_load_or_create_reuses_valid_cache(monkeypatch, tmp_path, fake_docs, fake_embedder):
    _patch_config(monkeypatch, tmp_path)
    monkeypatch.setattr(sem, "get_embedder", lambda: fake_embedder)

    first = sem.SemanticSearch(lambda: fake_docs).load_or_create_embeddings()
    assert fake_embedder.encode_calls == 1
    assert first.shape == (3, 8)

    second = sem.SemanticSearch(lambda: fake_docs).load_or_create_embeddings()
    assert fake_embedder.encode_calls == 1  # cached, no re-embed
    assert second.shape == (3, 8)


def test_load_or_create_rebuilds_on_fingerprint_drift(monkeypatch, tmp_path, fake_docs, fake_embedder):
    _patch_config(monkeypatch, tmp_path)
    monkeypatch.setattr(sem, "get_embedder", lambda: fake_embedder)

    sem.SemanticSearch(lambda: fake_docs).load_or_create_embeddings()
    assert fake_embedder.encode_calls == 1

    monkeypatch.setattr(sem, "source_fingerprint", lambda: "changed")
    sem.SemanticSearch(lambda: fake_docs).load_or_create_embeddings()
    assert fake_embedder.encode_calls == 2  # rebuilt


def test_load_or_create_raises_on_corrupt_meta(monkeypatch, tmp_path, fake_docs, fake_embedder):
    _patch_config(monkeypatch, tmp_path)
    monkeypatch.setattr(sem, "get_embedder", lambda: fake_embedder)

    sem.SemanticSearch(lambda: fake_docs).load_or_create_embeddings()
    sem.EMBEDDINGS_META_CACHE_PATH.write_text("{not json!!")
    with pytest.raises(CacheInvalidError):
        sem.SemanticSearch(lambda: fake_docs).load_or_create_embeddings()


def test_search_raises_empty_query(monkeypatch, tmp_path, fake_docs, fake_embedder):
    _patch_config(monkeypatch, tmp_path)
    monkeypatch.setattr(sem, "get_embedder", lambda: fake_embedder)
    ss = sem.SemanticSearch(lambda: fake_docs)
    ss.load_or_create_embeddings()
    with pytest.raises(Exception):
        ss.search("   ")


def test_build_embeddings_writes_meta(monkeypatch, tmp_path, fake_docs, fake_embedder):
    _patch_config(monkeypatch, tmp_path)
    monkeypatch.setattr(sem, "get_embedder", lambda: fake_embedder)
    sem.SemanticSearch(lambda: fake_docs).build_embeddings()
    meta = json.loads(sem.EMBEDDINGS_META_CACHE_PATH.read_text())
    assert meta["fingerprint"] == FINGERPRINT
    assert meta["embedder"] == sem.EMBEDDING_MODEL


def test_all_ones_vectors_produce_equal_scores(monkeypatch, tmp_path, fake_docs, fake_embedder):
    _patch_config(monkeypatch, tmp_path)
    monkeypatch.setattr(sem, "get_embedder", lambda: fake_embedder)
    ss = sem.SemanticSearch(lambda: fake_docs)
    ss.load_or_create_embeddings()
    results = ss.search("bear")
    assert len(results) == len(fake_docs)
    scores = [s for s, _ in results]
    assert np.allclose(scores, scores[0])
