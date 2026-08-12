"""Tests for chunked semantic search: chunking, cache lifecycle, and legacy compat."""
import numpy as np
import pytest

import cli.lib.chunked_semantic_search as cse
import cli.lib.semantic_search as sem
from cli.lib.exceptions import CacheInvalidError

FINGERPRINT = "chunk-fp1"


def _patch_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(cse, "CHUNK_EMBEDDINGS_CACHE_PATH", tmp_path / "chunk_embeddings.npy")
    monkeypatch.setattr(cse, "CHUNK_METADATA_CACHE_PATH", tmp_path / "chunk_metadata.json")
    monkeypatch.setattr(cse, "source_fingerprint", lambda: FINGERPRINT)


def _patch_embedder(monkeypatch, fake_embedder):
    # get_embedder is resolved in both module namespaces: the base class stores
    # self.model, the subclass encodes chunks.
    monkeypatch.setattr(cse, "get_embedder", lambda: fake_embedder)
    monkeypatch.setattr(sem, "get_embedder", lambda: fake_embedder)


def test_semantic_chunks_simple(monkeypatch) -> None:
    text = "First sentence here. Second one. Third one. Fourth one. Fifth one."
    chunks = cse.semantic_chunks(text, max_chunk_size=2, overlap=1)
    assert chunks
    assert all(isinstance(c, str) and c for c in chunks)


def test_semantic_chunks_empty() -> None:
    assert cse.semantic_chunks("   ") == []


def test_load_or_create_builds_and_reuses(monkeypatch, tmp_path, fake_docs, fake_embedder):
    _patch_paths(monkeypatch, tmp_path)
    _patch_embedder(monkeypatch, fake_embedder)

    first = cse.ChunkedSemanticSearch(lambda: fake_docs).load_or_create_chunk_embeddings()
    assert fake_embedder.encode_calls == 1

    second = cse.ChunkedSemanticSearch(lambda: fake_docs).load_or_create_chunk_embeddings()
    assert fake_embedder.encode_calls == 1  # reused from cache
    assert first.shape == second.shape


def test_legacy_cache_without_fingerprint_is_reused(monkeypatch, tmp_path, fake_docs, fake_embedder):
    _patch_paths(monkeypatch, tmp_path)
    _patch_embedder(monkeypatch, fake_embedder)
    engine = cse.ChunkedSemanticSearch(lambda: fake_docs)
    engine.load_or_create_chunk_embeddings()
    assert fake_embedder.encode_calls == 1

    # Strip the fingerprint to simulate a legacy cache; it must still be valid.
    import json
    meta = json.loads(cse.CHUNK_METADATA_CACHE_PATH.read_text())
    del meta["fingerprint"]
    cse.CHUNK_METADATA_CACHE_PATH.write_text(json.dumps(meta))

    cse.ChunkedSemanticSearch(lambda: fake_docs).load_or_create_chunk_embeddings()
    assert fake_embedder.encode_calls == 1  # no rebuild


def test_rebuilds_when_fingerprint_changes(monkeypatch, tmp_path, fake_docs, fake_embedder):
    _patch_paths(monkeypatch, tmp_path)
    _patch_embedder(monkeypatch, fake_embedder)
    cse.ChunkedSemanticSearch(lambda: fake_docs).load_or_create_chunk_embeddings()
    assert fake_embedder.encode_calls == 1

    monkeypatch.setattr(cse, "source_fingerprint", lambda: "changed")
    cse.ChunkedSemanticSearch(lambda: fake_docs).load_or_create_chunk_embeddings()
    assert fake_embedder.encode_calls == 2


def test_corrupt_metadata_raises_cache_invalid(monkeypatch, tmp_path, fake_docs, fake_embedder):
    _patch_paths(monkeypatch, tmp_path)
    _patch_embedder(monkeypatch, fake_embedder)
    cse.CHUNK_METADATA_CACHE_PATH.write_text("{broken")
    with pytest.raises(CacheInvalidError):
        cse.ChunkedSemanticSearch(lambda: fake_docs).load_or_create_chunk_embeddings()


def test_search_legacy_metadata_without_document_id(monkeypatch, tmp_path, fake_docs, fake_embedder):
    _patch_paths(monkeypatch, tmp_path)
    _patch_embedder(monkeypatch, fake_embedder)
    engine = cse.ChunkedSemanticSearch(lambda: fake_docs)
    engine.load_or_create_chunk_embeddings()

    # Simulate legacy metadata keyed only by document_idx (no document_id).
    import json
    meta = json.loads(cse.CHUNK_METADATA_CACHE_PATH.read_text())
    for entry in meta["chunks"]:
        entry.pop("document_id")
    cse.CHUNK_METADATA_CACHE_PATH.write_text(json.dumps(meta))
    engine2 = cse.ChunkedSemanticSearch(lambda: fake_docs)
    engine2.load_or_create_chunk_embeddings()

    results = engine2.search("bear")
    assert results
    assert all(np.isfinite(s) for s, _ in results)
