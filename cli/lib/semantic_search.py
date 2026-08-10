from typing import Callable
import os
from cli.lib.movies import PROJECT_ROOT, load_movies
from cli.lib.document import Document
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSearch():
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents: list[Document] = []
        self.document_map: dict[str, Document] = {}

    def generate_embedding(self, text: str):
        if text.strip() == "":
            raise ValueError("Text cannot be empty")
        embeddings = self.model.encode([text])
        return embeddings[0]

    def build_embeddings(self, documents: list[Document]):
        self.documents = documents
        doc_strings = []
        for document in documents:
            self.document_map[document.get_id()] = document
            doc_strings.append(document.to_text())
        self.embeddings = self.model.encode(doc_strings, show_progress_bar=True)
        EMBEDDING_CACHE_PATH = PROJECT_ROOT / "cache" / "embeddings.npy"
        np.save(EMBEDDING_CACHE_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Document]):
        self.documents = documents
        for document in documents:
            self.document_map[document.get_id()] = document
        if (os.path.exists(PROJECT_ROOT / "cache" / "embeddings.npy")):
            self.embeddings = np.load(PROJECT_ROOT / "cache" / "embeddings.npy")
            if (self.embeddings.shape[0] == len(documents)):
                return self.embeddings
        return self.build_embeddings(documents)

    def verify(self):
        print(f"Model loaded: {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")


def embed_text(text: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings(doc_loader: Callable[[], list[Document]]):
    documents = doc_loader()
    ss = SemanticSearch()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )
