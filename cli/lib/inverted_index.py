from itertools import islice
import math
from pickle import dump, load
import os
from collections import defaultdict, Counter
from typing import Callable
from cli.lib.search_utils import PROJECT_ROOT, tokenize_text_all, BM25_K1, BM25_B, tokenize_text, DEFAULT_SEARCH_LIMIT
from cli.lib.document import Document


class InvertedIndex:
    def __init__(self, load_documents: Callable[[], list[Document]]):
        self.load_documents = load_documents
        self.index: dict[str, list[str]] = defaultdict(list)
        self.docmap: dict[str, Document] = {}
        self.doc_lengths: dict[str, int] = defaultdict(int)
        self.term_frequencies:dict[str, Counter] = defaultdict(Counter)

    def __add_document(self, doc_id: str, text: str):
        text_tokens = tokenize_text_all(text)
        for token in text_tokens:
            self.term_frequencies[doc_id][token] += 1
        self.doc_lengths[doc_id] = len(text_tokens)
        unique_tokens = set(text_tokens)
        for token in unique_tokens:
            self.index[token].append(doc_id)
    
    def __get_avg_doc_length(self) -> float:
        if (len(self.doc_lengths) == 0):
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_documents(self, term: str) -> list[Document]:
        doc_ids = self.index.get(term, [])
        return [self.docmap[doc_id] for doc_id in doc_ids]

    def build(self):
        documents = self.load_documents()
        for doc in documents:
            self.__add_document(doc.get_id(), doc.to_text())
            self.docmap[doc.get_id()] = doc

    def get_tf(self, doc_id, term):
        # Assuming term is a single token
        return self.term_frequencies[doc_id][term]

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
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)
    
    def bm25(self, doc_id: str, term: str) :
        return self.get_bm25_idf(term) * self.get_bm25_tf(doc_id, term)

    def bm25_search(self, query: str, limit: int= DEFAULT_SEARCH_LIMIT):
        query_tokens = tokenize_text(query)
        doc_scores = defaultdict(float)
        for doc_id in self.docmap.keys():
            doc_scores[doc_id] = sum(self.bm25(doc_id, query_token) for query_token in query_tokens)
        doc_scores = dict(sorted(doc_scores.items(), key=lambda item: item[1], reverse=True))
        return dict(islice(({self.docmap[doc_id]: score for doc_id, score in doc_scores.items() if doc_id in self.docmap}).items(), limit))

    def tfidf(self, doc_id, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.idf(term)
        return tf * idf

    def save(self):
        cache_path = os.path.join(PROJECT_ROOT, "cache")
        if not os.path.isdir(cache_path):
            os.makedirs(cache_path)
        INDEX_CACHE_PATH = os.path.join(cache_path, "index.pkl")
        DOCMAP_CACHE_PATH = os.path.join(cache_path, "docmap.pkl")
        TERM_FREQ_CACHE_PATH = os.path.join(cache_path, "term_frequencies.pkl")
        DOC_LENGTH_CACHE_PATH = os.path.join(cache_path, "doc_lengths.pkl")
        with open(INDEX_CACHE_PATH, "wb") as f:
            dump(self.index, f)
        with open(DOCMAP_CACHE_PATH, "wb") as f:
            dump(self.docmap, f)
        with open(TERM_FREQ_CACHE_PATH, "wb") as f:
            dump(self.term_frequencies, f)
        with open(DOC_LENGTH_CACHE_PATH, "wb") as f:
            dump(self.doc_lengths, f)

    def load(self):
        cache_path = os.path.join(PROJECT_ROOT, "cache")
        INDEX_CACHE_PATH = os.path.join(cache_path, "index.pkl")
        DOCMAP_CACHE_PATH = os.path.join(cache_path, "docmap.pkl")
        TERM_FREQ_CACHE_PATH = os.path.join(cache_path, "term_frequencies.pkl")
        DOC_LENGTH_CACHE_PATH = os.path.join(cache_path, "doc_lengths.pkl")
        with open(INDEX_CACHE_PATH, "rb") as f:
            self.index = load(f)
        with open(DOCMAP_CACHE_PATH, "rb") as f:
            self.docmap = load(f)
        with open(TERM_FREQ_CACHE_PATH, "rb") as f:
            self.term_frequencies = load(f)
        with open(DOC_LENGTH_CACHE_PATH, "rb") as f:
            self.doc_lengths = load(f)


