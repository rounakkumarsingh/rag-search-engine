from nltk.stem.porter import PorterStemmer
from string import punctuation

from cli.lib.search_utils import get_stopwords


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
