from cli.lib.inverted_index import InvertedIndex
from cli.lib.models import Movie
from cli.lib.search_utils import DEFAULT_SEARCH_LIMIT


class SearchService:
    def __init__(self) -> None:
        self.idx: InvertedIndex[Movie] | None = None

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[Movie]:
        try:
            if self.idx is None:
                self.idx = InvertedIndex[Movie]()
                self.idx.load()

            return self.idx.get_documents_by_ids(self.idx.get_document_ids(query))[
                :limit
            ]
        except FileNotFoundError:
            print("Cache was not found")
            return []
