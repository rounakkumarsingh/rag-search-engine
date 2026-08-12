from typing import Callable
from cli.lib.caches import load_numpy_if_valid, save_numpy
from cli.lib.config import EMBEDDINGS_CACHE_PATH
from cli.lib.models import get_embedder
from cli.lib.movies import load_movies
from cli.lib.document import Document
from cli.lib.search_utils import DEFAULT_SEARCH_LIMIT
import numpy as np

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
        return self.embeddings

    def load_or_create_embeddings(self) -> np.ndarray:
        self._load_documents()
        cached = load_numpy_if_valid(EMBEDDINGS_CACHE_PATH, expected_rows=len(self.documents))
        if cached is not None:
            self.embeddings = cached
            return self.embeddings
        return self.build_embeddings()

    def search(self, query: str, limit = DEFAULT_SEARCH_LIMIT) -> list[tuple[float, Document]]:
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
