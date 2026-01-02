from cli.lib.inverted_index import InvertedIndex
from cli.lib.models import Movie


class TFService:
    def __init__(self) -> None:
        self.index: None | InvertedIndex[Movie] = None

    def find_tf(self, doc_id: int, term: str) -> int:
        if self.index is None:
            self.index = InvertedIndex[Movie]()
            self.index.load()

        if doc_id not in self.index.docmap:
            raise ValueError("Invalid doc_id")
        return self.index.get_tf(doc_id, term)
