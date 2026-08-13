from typing import Callable

from PIL import Image
from sentence_transformers import SentenceTransformer

from cli.lib.document import Document
from cli.lib.movies import load_movies
from cli.lib.semantic_search import cosine_similarity


class MultimodalSearch:
    def __init__(self, loader: Callable[[], list[Document]]) -> None:
        self.model = SentenceTransformer("clip-ViT-B-32")
        self.documents: list[Document] = loader()
        self.texts = [
            f"{doc.get_title()}: {doc.get_description()}" for doc in self.documents
        ]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        embedding = self.model.encode([image])[0]
        return embedding

    def search_with_image(self, image_path: str, limit: int = 5) -> list[dict]:
        image_embedding = self.embed_image(image_path)
        results = []
        for pos, doc in enumerate(self.documents):
            score = cosine_similarity(self.text_embeddings[pos], image_embedding)
            results.append({
                "id": doc.get_id(),
                "title": doc.get_title(),
                "description": doc.get_description(),
                "similarity": score,
            })
        results.sort(key=lambda result: result["similarity"], reverse=True)
        return results[:limit]


def image_search_command(image_path: str) -> list[dict]:
    multimodal = MultimodalSearch(load_movies)
    return multimodal.search_with_image(image_path)


def verify_image_embedding(image_path: str) -> None:
    multimodal = MultimodalSearch(load_movies)
    embedding = multimodal.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
