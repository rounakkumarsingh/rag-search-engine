"""Tests for the FAISS flat-index helpers (cosine scoring + persistence)."""
import numpy as np
import pytest

from cli.lib.faiss_index import build_flat_index, cosine_search, load_index, save_index


def _brute_cosine(embeddings, query):
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query)
    dots = embeddings @ query
    return dots / np.where(norms == 0, 1.0, norms)


def test_build_flat_index_matches_brute_force_cosine():
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(50, 16)).astype("float32")
    query = rng.normal(size=(16,)).astype("float32")

    index = build_flat_index(embeddings)
    scores, indices = cosine_search(index, query, 5)

    expected = np.argsort(-_brute_cosine(embeddings, query))[:5]
    assert indices.tolist() == expected.tolist()
    assert np.allclose(scores, _brute_cosine(embeddings, query)[expected], atol=1e-6)


def test_cosine_search_k_greater_than_ntotal_returns_sentinels():
    embeddings = np.eye(3, dtype="float32")
    index = build_flat_index(embeddings)
    scores, indices = cosine_search(index, embeddings[0], k=5)

    assert np.sum(indices >= 0) == 3
    assert set(indices[indices >= 0].tolist()) == {0, 1, 2}
    assert np.all(np.isfinite(scores[:3]))


def test_save_load_roundtrip(tmp_path):
    embeddings = np.random.default_rng(1).normal(size=(10, 8)).astype("float32")
    index = build_flat_index(embeddings)

    path = tmp_path / "faiss.index"
    save_index(index, path)
    assert path.exists()

    loaded = load_index(path)
    scores_a, idx_a = cosine_search(index, embeddings[0], 3)
    scores_b, idx_b = cosine_search(loaded, embeddings[0], 3)
    np.testing.assert_array_equal(idx_a, idx_b)
    np.testing.assert_allclose(scores_a, scores_b)


def test_build_flat_index_rejects_bad_shape():
    with pytest.raises(ValueError):
        build_flat_index(np.zeros((384,), dtype="float32"))