import json
from os import path

from .models import Movie, Root

DEFAULT_SEARCH_LIMIT = 10

PROJECT_ROOT = path.dirname(path.dirname(path.dirname(__file__)))  # ../..
MOVIE_DATA_PATH = path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_DATA_PATH = path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_PATH = path.join(PROJECT_ROOT, "cache")


def load_movies() -> list[Movie]:
    with open(MOVIE_DATA_PATH, "r") as f:
        data: Root = json.load(f)
    return [
        Movie(movie["id"], movie["title"], movie["description"])
        for movie in data["movies"]
    ]


def load_stopwords() -> list[str]:
    with open(STOPWORDS_DATA_PATH, "r") as fp:
        data = fp.read()
    return data.splitlines()
