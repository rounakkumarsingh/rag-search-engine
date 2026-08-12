from itertools import islice
import math
from collections import defaultdict, Counter
from typing import Callable
from cli.lib.caches import exists_all, load_pickle, save_pickle
from cli.lib.config import (
    DOC_LENGTHS_CACHE_PATH,
    DOCMAP_CACHE_PATH,
    INDEX_CACHE_PATH,
    TERM_FREQUENCIES_CACHE_PATH,
)
from cli.lib.search_utils import tokenize_text_all, BM25_K1, BM25_B, DEFAULT_SEARCH_LIMIT
from cli.lib.document import Document
from cli.lib.exceptions import EmptyQueryError, IndexNotFoundError

INDEX_ARTIFACTS: list = [
    INDEX_CACHE_PATH,
    DOCMAP_CACHE_PATH,
    TERM_FREQUENCIES_CACHE_PATH,
    DOC_LENGTHS_CACHE_PATH,
]


class InvertedIndex:
    def __init__(self, load_documents: Callable[[], list[Document]]):
        self.load_documents = load_documents
        self.index: dict[str, list[str]] = defaultdict(list)
        self.docmap: dict[str, Document] = {}
        self.doc_lengths: dict[str, int] = defaultdict(int)
        self.term_frequencies: dict[str, Counter] = defaultdict(Counter)

    def __add_document(self, doc_id: str, text: str) -> None:
        text_tokens = tokenize_text_all(text)
        for token in text_tokens:
            self.term_frequencies[doc_id][token] += 1
        self.doc_lengths[doc_id] = len(text_tokens)
        unique_tokens = set(text_tokens)
        for token in unique_tokens:
            self.index[token].append(doc_id)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_documents(self, term: str) -> list[Document]:
        doc_ids = self.index.get(term, [])
        return [self.docmap[doc_id] for doc_id in doc_ids]

    def build(self) -> None:
        documents = self.load_documents()
        for doc in documents:
            self.__add_document(doc.get_id(), doc.to_text())
            self.docmap[doc.get_id()] = doc

    @classmethod
    def from_documents(cls, documents: list[Document]) -> "InvertedIndex":
        index = cls(lambda: documents)
        index.build()
        return index

    @classmethod
    def load_or_build(cls, load_documents: Callable[[], list[Document]]) -> "InvertedIndex":
        index = cls(load_documents)
        if exists_all(INDEX_ARTIFACTS):
            index.load()
        else:
            index.build()
            index.save()
        return index

    def get_tf(self, doc_id: str, term: str) -> int:
        return self.term_frequencies.get(doc_id, Counter()).get(term, 0)

    def idf(self, term: str) -> float:
        N = len(self.docmap)
        df = len(self.index.get(term, []))
        return math.log((N + 1) / (df + 1))

    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        df = len(self.index.get(term, []))
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id: str, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        length_norm = 1 - b + b * (doc_length / self.__get_avg_doc_length())
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: str, term: str) -> float:
        return self.get_bm25_idf(term) * self.get_bm25_tf(doc_id, term)

    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[tuple[float, Document]]:
        query_tokens = tokenize_text_all(query)
        if not query_tokens:
            raise EmptyQueryError("Query produced no searchable tokens")
        doc_scores = defaultdict(float)
        for doc_id in self.docmap.keys():
            doc_scores[doc_id] = sum(self.bm25(doc_id, query_token) for query_token in query_tokens)
        ranked = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
        return [(score, self.docmap[doc_id]) for doc_id, score in islice(ranked, limit)]

    def tfidf(self, doc_id: str, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.idf(term)
        return tf * idf

    def save(self) -> None:
        save_pickle(INDEX_CACHE_PATH, self.index)
        save_pickle(DOCMAP_CACHE_PATH, self.docmap)
        save_pickle(TERM_FREQUENCIES_CACHE_PATH, self.term_frequencies)
        save_pickle(DOC_LENGTHS_CACHE_PATH, self.doc_lengths)

    def load(self) -> None:
        if not exists_all(INDEX_ARTIFACTS):
            raise IndexNotFoundError(
                "Inverted index cache is incomplete; rebuild it with `build` or `load_or_build`."
            )
        self.index = load_pickle(INDEX_CACHE_PATH)
        self.docmap = load_pickle(DOCMAP_CACHE_PATH)
        self.term_frequencies = load_pickle(TERM_FREQUENCIES_CACHE_PATH)
        self.doc_lengths = load_pickle(DOC_LENGTHS_CACHE_PATH)
