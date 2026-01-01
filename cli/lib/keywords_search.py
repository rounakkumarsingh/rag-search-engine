from .search_utils import DEFAULT_SEARCH_LIMIT, Movie, load_movies, load_stopwords
import string
from nltk.stem import PorterStemmer

STEMMER = PorterStemmer()
MOVIES = load_movies()
STOPWORDS = load_stopwords()


def search_query(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[Movie]:
    query = query
    results: list[Movie] = []
    query_tokens = preprocess_text(query)
    for movie in MOVIES:
        title_tokens = preprocess_text(movie["title"])
        if match_tokens(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break

    return results


def match_tokens(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [x for x in text.split() if x not in STOPWORDS]
    stemmed_tokens: set[str] = set()
    for token in tokens:
        stemmed_token = STEMMER.stem(token)
        if stemmed_token not in stemmed_tokens:
            stemmed_tokens.add(stemmed_token)
    return list(stemmed_tokens)
