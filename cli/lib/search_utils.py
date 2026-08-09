from typing import Final
import os
from nltk.stem.porter import PorterStemmer
from string import punctuation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
DEFAULT_SEARCH_LIMIT: Final = 5

STOPWORDS: list[str] = []
def get_stopwords():
    if not STOPWORDS:
        with open(STOPWORDS_PATH, "r") as f:
            STOPWORDS[:] = f.read().splitlines()
    return STOPWORDS

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", punctuation))
    return text

def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    stopwords = get_stopwords()
    valid_tokens = list(filter(lambda token: token not in stopwords and token != "", tokens))
    stemmer = PorterStemmer()
    valid_tokens = list(set(map(stemmer.stem, valid_tokens)))
    return valid_tokens
