from cli.lib.inverted_index import InvertedIndex
from cli.lib.models import Movie


class IDFService:
    def __init__(self) -> None:
        self.index: InvertedIndex[Movie] | None = None

    def idf_score(self, term: str) -> float:
        if self.index is None:
            self.index = InvertedIndex[Movie]()
            self.index.load()

        return self.index.get_idf(term)
