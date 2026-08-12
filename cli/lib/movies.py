import json

from cli.lib.config import DATA_PATH
from cli.lib.document import Document


class MovieDocument:
    def __init__(self, raw: dict):
        self._raw = raw

    def get_id(self) -> str:
        return str(self._raw["id"])

    def get_title(self) -> str:
        return self._raw["title"]

    def get_description(self) -> str:
        return self._raw["description"]

    def to_text(self) -> str:
        return f"{self._raw['title']} {self._raw['description']}"

    def get_semantic_text(self) -> str:
        return self._raw["description"]


def load_movies() -> list[Document]:
    with open(DATA_PATH) as f:
        data = json.load(f)
    return [MovieDocument(m) for m in data["movies"]]
