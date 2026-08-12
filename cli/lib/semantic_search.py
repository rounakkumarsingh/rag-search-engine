from cli.lib.search_utils import DEFAULT_SEARCH_LIMIT
from typing import Callable
import os
from cli.lib.config import EMBEDDINGS_CACHE_PATH, EMBEDDING_MODEL
from cli.lib.movies import load_movies
from cli.lib.document import Document
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSearch():
    def __init__(self, loader: Callable[[], list[Document]]):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents: list[Document] = []
        self.document_map: dict[str, Document] = {}
        self.doc_loader = loader

    def generate_embedding(self, text: str):
        if text.strip() == "":
            raise ValueError("Text cannot be empty")
        embeddings = self.model.encode([text])
        return embeddings[0]

    def build_embeddings(self):
        self.documents = self.doc_loader()
        doc_strings = []
        for document in self.documents:
            self.document_map[document.get_id()] = document
            doc_strings.append(document.to_text())
        self.embeddings = self.model.encode(doc_strings, show_progress_bar=True)
        np.save(EMBEDDINGS_CACHE_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self):
        self.documents = self.doc_loader()
        for document in self.documents:
            self.document_map[document.get_id()] = document
        if (os.path.exists(EMBEDDINGS_CACHE_PATH)):
            self.embeddings = np.load(EMBEDDINGS_CACHE_PATH)
            if (self.embeddings.shape[0] == len(self.documents)):
                return self.embeddings
        return self.build_embeddings()

    def search(self, query: str, limit = DEFAULT_SEARCH_LIMIT):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embeddings = self.generate_embedding(query)
        scores: list[tuple[float, Document]] = []
        for pos, document in enumerate(self.documents):
            scores.append(((cosine_similarity(self.embeddings[pos], query_embeddings)), document))
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:limit]

    def verify(self):
        print(f"Model loaded: {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")


def embed_text(text: str):
    ss = SemanticSearch(lambda: [])
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def embed_query_text(query: str):
    ss = SemanticSearch(lambda: [])
    embedding = ss.generate_embedding(query)
    print(f"Text: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def verify_embeddings(doc_loader: Callable[[], list[Document]]):
    ss = SemanticSearch(doc_loader)
    embeddings = ss.load_or_create_embeddings()
    print(f"Number of docs:   {len(ss.documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
