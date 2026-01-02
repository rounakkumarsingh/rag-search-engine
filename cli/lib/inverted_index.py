from collections import Counter
from math import log
import os
import pickle
import string
from typing import Generic, Protocol, TypeVar

from nltk import PorterStemmer

from cli.lib.search_utils import CACHE_PATH, load_stopwords

STEMMER = PorterStemmer()

STOPWORDS = load_stopwords()


class Preprocessable(Protocol):
    def doc_id(self) -> int: ...
    def __str__(self) -> str: ...


Document = TypeVar("Document", bound=Preprocessable)


class InvertedIndex(Generic[Document]):
    def __init__(self) -> None:
        self.index: dict[str, list[int]] = {}
        self.docmap: dict[int, Document] = {}
        self.term_frequencies: dict[int, Counter[str]] = {}

    def __add_document(self, doc: Document) -> None:
        doc_id = doc.doc_id()
        tokens = preprocess_text(str(doc))
        self.term_frequencies[doc_id] = Counter(tokens)
        for token in tokens:
            self.index.setdefault(token, []).append(doc_id)

        self.docmap[doc_id] = doc

    def build(self, items: list[Document]) -> None:
        for item in items:
            self.__add_document(item)

    def get_tf(self, doc_id: int, term: str) -> int:
        if doc_id not in self.term_frequencies:
            raise ValueError("Invalid doc_id")
        return self.term_frequencies[doc_id][term]

    def get_document_ids(self, term: str) -> list[int]:
        tokens = preprocess_text(term)
        doc_ids: set[int] = set()

        for token in tokens:
            doc_ids.update(self.index.get(token, []))

        return sorted(doc_ids)

    def get_documents_by_ids(self, doc_ids: list[int]) -> list[Document]:
        return [self.docmap[doc_id] for doc_id in doc_ids]

    def save(self) -> None:
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(os.path.join(CACHE_PATH, "index.pkl"), "wb") as fp:
            pickle.dump(self.index, fp)
        with open(os.path.join(CACHE_PATH, "docmap.pkl"), "wb") as fp:
            pickle.dump(self.docmap, fp)
        with open(os.path.join(CACHE_PATH, "term_frequencies.pkl"), "wb") as fp:
            pickle.dump(self.term_frequencies, fp)

    def load(self) -> None:
        INDEX_CACHE = os.path.join(CACHE_PATH, "index.pkl")
        DOCMAP_CACHE = os.path.join(CACHE_PATH, "docmap.pkl")
        TF_CACHE = os.path.join(CACHE_PATH, "term_frequencies.pkl")
        if not os.path.exists(INDEX_CACHE) or not os.path.exists(DOCMAP_CACHE):
            raise FileNotFoundError("Cache is not available")
        with open(INDEX_CACHE, "rb") as fp:
            self.index = pickle.load(fp)
        with open(DOCMAP_CACHE, "rb") as fp:
            self.docmap = pickle.load(fp)
        with open(TF_CACHE, "rb") as fp:
            self.term_frequencies = pickle.load(fp)

    def get_idf(self, term: str) -> float:
        tokens = preprocess_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return log((doc_count + 1) / (term_doc_count + 1))

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        return self.get_idf(term) * self.get_tf(doc_id, term)


def preprocess_text(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [x for x in text.split() if x not in STOPWORDS]
    stemmed_tokens: list[str] = []
    for token in tokens:
        stemmed_token = STEMMER.stem(token)
        stemmed_tokens.append(stemmed_token)
    return stemmed_tokens
