from typing import Final
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
DEFAULT_SEARCH_LIMIT: Final = 5

STOPWORDS: list[str] = []
def get_stopwords():
    if not STOPWORDS:
        with open(STOPWORDS_PATH, "r") as f:
            STOPWORDS[:] = f.read().splitlines()
    return STOPWORDS
