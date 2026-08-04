from nltk.stem.porter import PorterStemmer
from string import punctuation

from cli.lib.search_utils import load_movies, Movie, DEFAULT_SEARCH_LIMIT, get_stopwords


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", punctuation))
    return text

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[Movie]:
    movies = load_movies()
    results = []
    query_tokens = tokenize_text(query)
    for movie in movies:
        title_tokens = tokenize_text(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break

    return results


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False




def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    stopwords = get_stopwords()
    valid_tokens = list(filter(lambda token: token not in stopwords and token != "", tokens))
    stemmer = PorterStemmer()
    valid_tokens = list(set(map(stemmer.stem, valid_tokens)))
    return valid_tokens
