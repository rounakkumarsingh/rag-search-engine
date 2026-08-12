import json
from typing import Callable

import numpy as np

from cli.lib.caches import (
    load_json,
    load_numpy_if_valid,
    save_json,
    save_numpy,
    source_fingerprint,
)
from cli.lib.config import (
    EMBEDDINGS_CACHE_PATH,
    EMBEDDINGS_META_CACHE_PATH,
    EMBEDDING_MODEL,
)
from cli.lib.models import get_embedder
from cli.lib.document import Document
from cli.lib.search_utils import DEFAULT_SEARCH_LIMIT
from cli.lib.exceptions import CacheInvalidError, EmptyQueryError

class SemanticSearch():
    def __init__(self, loader: Callable[[], list[Document]]):
        self.model = get_embedder()
        self.embeddings: np.ndarray | None = None
        self.documents: list[Document] = []
        self.document_map: dict[str, Document] = {}
        self.doc_loader = loader

    def _load_documents(self) -> None:
        self.documents = self.doc_loader()
        self.document_map = {document.get_id(): document for document in self.documents}

    def generate_embedding(self, text: str) -> np.ndarray:
        if text.strip() == "":
            raise ValueError("Text cannot be empty")
        embeddings = self.model.encode([text])
        return embeddings[0]

    def build_embeddings(self) -> np.ndarray:
        self._load_documents()
        doc_strings = [document.to_text() for document in self.documents]
        self.embeddings = self.model.encode(doc_strings, show_progress_bar=True)
        save_numpy(EMBEDDINGS_CACHE_PATH, self.embeddings)
        save_json(EMBEDDINGS_META_CACHE_PATH, {
            "fingerprint": source_fingerprint(),
            "embedder": EMBEDDING_MODEL,
        })
        return self.embeddings

    def _cache_is_valid(self) -> bool:
        if not EMBEDDINGS_META_CACHE_PATH.exists():
            return False
        meta = load_json(EMBEDDINGS_META_CACHE_PATH)
        return (
            meta.get("fingerprint") == source_fingerprint()
            and meta.get("embedder") == EMBEDDING_MODEL
        )

    def load_or_create_embeddings(self) -> np.ndarray:
        self._load_documents()
        valid = load_numpy_if_valid(EMBEDDINGS_CACHE_PATH, expected_rows=len(self.documents))
        if valid is not None:
            if EMBEDDINGS_META_CACHE_PATH.exists():
                try:
                    if self._cache_is_valid():
                        self.embeddings = valid
                        return self.embeddings
                except json.JSONDecodeError as exc:
                    raise CacheInvalidError("Embeddings metadata cache is unreadable") from exc
            return self.build_embeddings()
        return self.build_embeddings()

    def search(self, query: str, limit = DEFAULT_SEARCH_LIMIT) -> list[tuple[float, Document]]:
        if not query.strip():
            raise EmptyQueryError("Query is empty")
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embeddings = self.generate_embedding(query)
        scores: list[tuple[float, Document]] = []
        for pos, document in enumerate(self.documents):
            scores.append(((cosine_similarity(self.embeddings[pos], query_embeddings)), document))
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:limit]

    def verify(self) -> str:
        return (
            f"Model loaded: {self.model}\n"
            f"Max sequence length: {self.model.max_seq_length}"
        )


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
