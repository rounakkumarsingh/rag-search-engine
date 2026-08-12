from typing import Final
from nltk.stem.porter import PorterStemmer
from string import punctuation

from cli.lib.config import BM25_B, BM25_K1, DEFAULT_SEARCH_LIMIT, STOPWORDS_PATH

STOPWORDS: list[str] = []
_STEMMER = PorterStemmer()

def get_stopwords() -> list[str]:
    if not STOPWORDS:
        with open(STOPWORDS_PATH, "r") as f:
            STOPWORDS[:] = [preprocess_text(word) for word in f.read().splitlines()]
    return STOPWORDS

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", punctuation))
    return text

def tokenize_text_all(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    stopwords = get_stopwords()
    valid_tokens = list(filter(lambda token: token not in stopwords and token != "", tokens))
    return list(map(_STEMMER.stem, valid_tokens))


def tokenize_text(text: str) -> list[str]:
    # Order-preserving dedup: set() ordering varies per PYTHONHASHSEED
    # and made keyword-search results non-deterministic.
    return list(dict.fromkeys(tokenize_text_all(text)))


def tokenize_single_term(term: str) -> str:
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError(f"Expected exactly one token, but got {len(tokens)}: {tokens}")
    return tokens[0]
