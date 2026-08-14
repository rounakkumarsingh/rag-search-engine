from pathlib import Path

import faiss
import numpy as np


def _as_float32(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(array, dtype="float32"))


def build_flat_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build an inner-product FAISS index over L2-normalized embeddings.

    Normalizing vectors before indexing makes inner-product equal to cosine
    similarity, which matches the scoring used by :func:`cosine_similarity`.
    """
    vectors = _as_float32(embeddings)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Expected a 2D array of embeddings, got shape %r" % (vectors.shape,))
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_index(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))


def cosine_search(
    index: faiss.Index,
    query_embedding: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Search an IP index for the k nearest vectors to a query embedding.

    Returns a pair of ``(scores, indices)`` arrays of length ``k``. Slots with
    index ``-1`` mean the index has fewer than ``k`` vectors; scores there are
    unset (``-inf``) and must be filtered by callers.
    """
    query = _as_float32(query_embedding).reshape(1, -1)
    faiss.normalize_L2(query)
    distances, indices = index.search(query, k)
    return distances[0], indices[0]