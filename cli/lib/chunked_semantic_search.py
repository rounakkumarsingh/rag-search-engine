import json
from cli.lib.movies import PROJECT_ROOT
import re
import numpy
from cli.lib.document import Document
from typing import Callable
from cli.lib.semantic_search import SemanticSearch
from typing import TypedDict


class ChunkMetadata(TypedDict):
    document_idx: int
    chunk_idx: int
    total_chunks: int

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, loader: Callable[[], list[Document]]):
        super().__init__(loader)
        self.chunk_embeddings = None
        self.chunk_metadata: list[ChunkMetadata] = []

    def build_chunk_embeddings(self)-> numpy.ndarray:
        self.documents = self.doc_loader()
        for document in self.documents:
            self.document_map[document.get_id()] = document
        chunks:list[str] = []
        for document_idx, document in enumerate(self.documents):
            if not document.get_semantic_text():
                continue
            sentences = re.split(r"(?<=[.!?])\s+", document.get_semantic_text())
            MAX_CHUNK_SIZE = 4
            OVERLAP = 1
            cnt = 0
            doc_start = len(chunks)
            while cnt < len(sentences):
                chunks.append(" ".join(sentences[max(0, cnt - OVERLAP): max(0,cnt - OVERLAP) + MAX_CHUNK_SIZE]))
                cnt  = max(0,cnt - OVERLAP) + MAX_CHUNK_SIZE
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
