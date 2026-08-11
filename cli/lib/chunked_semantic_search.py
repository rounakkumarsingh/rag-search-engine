import json
from cli.lib.movies import PROJECT_ROOT
import re
import numpy
from cli.lib.document import Document
from typing import Callable
from cli.lib.semantic_search import SemanticSearch, cosine_similarity
from typing import TypedDict


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

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, loader: Callable[[], list[Document]]):
        super().__init__(loader)
        self.chunk_embeddings: numpy.ndarray | None = None
        self.chunk_metadata: list[ChunkMetadata] = []

    def build_chunk_embeddings(self)-> numpy.ndarray:
        self.documents = self.doc_loader()
        for document in self.documents:
            self.document_map[document.get_id()] = document
        chunks:list[str] = []
        for document_idx, document in enumerate(self.documents):
            if not document.get_semantic_text().strip():
                continue
            doc_start = len(chunks)
            chunks.extend(semantic_chunks(document.get_semantic_text()))
            doc_total_chunks = len(chunks) - doc_start
            for chunk_idx in range(doc_start, len(chunks)):
                self.chunk_metadata.append({"chunk_idx": chunk_idx, "document_idx": document_idx, "total_chunks": doc_total_chunks})
        self.chunk_embeddings = numpy.asarray(self.model.encode(chunks, show_progress_bar=True))
        CHUNK_EMBEDDINGS_CACHE_PATH = PROJECT_ROOT / "cache" / "chunk_embeddings.npy"
        numpy.save(CHUNK_EMBEDDINGS_CACHE_PATH, self.chunk_embeddings)
        CHUNK_METADATA_CACHE_PATH = PROJECT_ROOT / "cache" / "chunk_metadata.json"
        with open(CHUNK_METADATA_CACHE_PATH, "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(chunks)}, f, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self) -> numpy.ndarray:
        self.documents = self.doc_loader()
        for document in self.documents:
            self.document_map[document.get_id()] = document
        CHUNK_EMBEDDINGS_CACHE_PATH = PROJECT_ROOT / "cache" / "chunk_embeddings.npy"
        CHUNK_METADATA_CACHE_PATH = PROJECT_ROOT / "cache" / "chunk_metadata.json"
        if CHUNK_EMBEDDINGS_CACHE_PATH.exists() and CHUNK_METADATA_CACHE_PATH.exists():
            self.chunk_embeddings = numpy.load(CHUNK_EMBEDDINGS_CACHE_PATH)
            with open(CHUNK_METADATA_CACHE_PATH) as f:
                self.chunk_metadata = json.load(f)["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings()

    def search(self, query: str, limit = 10):
        if self.chunk_embeddings is None:
            raise ValueError("No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        scores: list[dict] = []
        for chunk_idx in range(len(self.chunk_embeddings)):
            similarity_score = cosine_similarity(query_embedding, self.chunk_embeddings[chunk_idx])
            scores.append({"chunk_idx": chunk_idx, "movie_idx": self.chunk_metadata[chunk_idx].get("document_idx"), "score": similarity_score})
        movies_best_scores:dict[Document, float] = {}
        for chunk_score in scores:
            if chunk_score["movie_idx"] in movies_best_scores:
                if chunk_score["score"] > movies_best_scores[chunk_score["movie_idx"]]:
                    movies_best_scores[chunk_score["movie_idx"]] = chunk_score["score"]
            else:
                movies_best_scores[chunk_score["movie_idx"]] = chunk_score["score"]

        return (sorted(movies_best_scores.items(), key=lambda item: item[1], reverse=True)[:limit])
