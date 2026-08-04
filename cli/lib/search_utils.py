from typing import TypedDict, Final
import json
import os

class Movie(TypedDict):
    id: int
    title: str
    description: str

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
DEFAULT_SEARCH_LIMIT: Final = 5

def load_movies() -> list[Movie]:
    with open(DATA_PATH, "r") as f:
        return json.load(f)["movies"]

STOPWORDS:list[str] = []
def get_stopwords():
    if not STOPWORDS:
        with open(STOPWORDS_PATH, "r") as f:
            STOPWORDS[:] = f.read().splitlines()
    return STOPWORDS
