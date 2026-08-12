import re

import numpy

from cli.lib.caches import load_json, load_numpy, load_numpy_if_valid, save_json, save_numpy, source_fingerprint
from cli.lib.config import CHUNK_EMBEDDINGS_CACHE_PATH, CHUNK_METADATA_CACHE_PATH, DATA_PATH, EMBEDDING_MODEL
from cli.lib.document import Document
from cli.lib.models import get_embedder
from cli.lib.search_utils import DEFAULT_SEARCH_LIMIT
from cli.lib.semantic_search import SemanticSearch
from typing import Callable, TypedDict


def semantic_chunks(text: str, max_chunk_size: int = 4, overlap: int = 1) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", stripped)
    if len(sentences) == 1 and not sentences[0].rstrip().endswith((".", "!", "?")):
        sentences = [stripped]

    sentences = [s.strip() for s in sentences]
    chunks: list[str] = []
    cnt = 0
    while cnt < len(sentences):
        start = max(0, cnt - overlap)
        chunk = " ".join(sentences[start: start + max_chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        cnt = start + max_chunk_size
    return chunks


class ChunkMetadata(TypedDict):
    document_idx: int
    chunk_idx: int
    total_chunks: int
    document_id: str


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, loader: Callable[[], list[Document]]):
        super().__init__(loader)
        self.chunk_embeddings: numpy.ndarray | None = None
        self.chunk_metadata: list[ChunkMetadata] = []
        self.document_positions: dict[str, int] = {}

    def _load_documents(self) -> None:
        super()._load_documents()
        self.document_positions = {
            document.get_id(): pos for pos, document in enumerate(self.documents)
        }

    def build_chunk_embeddings(self) -> numpy.ndarray:
        self._load_documents()
        self.chunk_metadata = []
        chunks: list[str] = []
        for document_idx, document in enumerate(self.documents):
            if not document.get_semantic_text().strip():
                continue
            doc_start = len(chunks)
            chunks.extend(semantic_chunks(document.get_semantic_text()))
            doc_total_chunks = len(chunks) - doc_start
            for chunk_idx in range(doc_start, len(chunks)):
                self.chunk_metadata.append({
                    "chunk_idx": chunk_idx,
                    "document_idx": document_idx,
                    "total_chunks": doc_total_chunks,
                    "document_id": document.get_id(),
                })
        self.chunk_embeddings = numpy.asarray(get_embedder().encode(chunks, show_progress_bar=True))
        save_numpy(CHUNK_EMBEDDINGS_CACHE_PATH, self.chunk_embeddings)
        save_json(CHUNK_METADATA_CACHE_PATH, {
            "chunks": self.chunk_metadata,
            "total_chunks": len(chunks),
            "fingerprint": source_fingerprint(),
            "embedder": EMBEDDING_MODEL,
        })
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self) -> numpy.ndarray:
        self._load_documents()
        metadata_payload = load_json(CHUNK_METADATA_CACHE_PATH) if CHUNK_METADATA_CACHE_PATH.exists() else None
        if metadata_payload is not None:
            cached = load_numpy_if_valid(CHUNK_EMBEDDINGS_CACHE_PATH, expected_rows=metadata_payload.get("total_chunks"))
            metadata = metadata_payload.get("chunks")
            fingerprint = metadata_payload.get("fingerprint")
            legacy_or_current = fingerprint is None or fingerprint == source_fingerprint()
            if cached is not None and isinstance(metadata, list) and len(cached) == len(metadata) and legacy_or_current:
                self.chunk_embeddings = cached
                self.chunk_metadata = metadata
                return self.chunk_embeddings
        return self.build_chunk_embeddings()

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[tuple[float, Document]]:
        if self.chunk_embeddings is None:
            raise ValueError("No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first.")
        query_embedding = self.generate_embedding(query)

        query_norm = numpy.linalg.norm(query_embedding)
        chunk_norms = numpy.linalg.norm(self.chunk_embeddings, axis=1)
        denominators = chunk_norms * query_norm
        similarities = numpy.zeros(len(self.chunk_embeddings))
        nonzero = denominators != 0
        similarities[nonzero] = (
            self.chunk_embeddings[nonzero] @ query_embedding
        ) / denominators[nonzero]

        movies_best_scores: dict[int, float] = {}
        for chunk_idx, metadata in enumerate(self.chunk_metadata):
            score = float(similarities[chunk_idx])
            doc_id = metadata.get("document_id")
            if doc_id is not None:
                doc_pos = self.document_positions[doc_id]
            else:
                doc_pos = metadata["document_idx"]
            if score > movies_best_scores.get(doc_pos, -1.0):
                movies_best_scores[doc_pos] = score

        ranked = sorted(movies_best_scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [(score, self.documents[pos]) for pos, score in ranked]
