import json
from pathlib import Path
from cli.lib.document import Document

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "movies.json"


class MovieDocument:
    def __init__(self, raw: dict):
        self._raw = raw

    def get_id(self) -> str:
        return str(self._raw["id"])

    def to_text(self) -> str:
        return f"{self._raw['title']} {self._raw['description']}"


def load_movies() -> list[Document]:
    with open(DATA_PATH) as f:
        data = json.load(f)
    return [MovieDocument(m) for m in data["movies"]]
