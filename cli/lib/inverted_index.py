from pickle import dump, load
import os
from collections import defaultdict, Counter
from typing import Callable
from cli.lib.search_utils import PROJECT_ROOT, tokenize_text_all
from cli.lib.document import Document


class InvertedIndex:
    def __init__(self, load_documents: Callable[[], list[Document]]):
        self.load_documents = load_documents
        self.index: dict[str, list[str]] = defaultdict(list)
        self.docmap: dict[str, Document] = {}
        self.term_frequencies:dict[str, Counter] = defaultdict(Counter)

    def __add_document(self, doc_id: str, text: str):
        text_tokens = tokenize_text_all(text)
        for token in text_tokens:
            self.index[token].append(doc_id)
            self.term_frequencies[doc_id][token] += 1

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

    def save(self):
        cache_path = os.path.join(PROJECT_ROOT, "cache")
        if not os.path.isdir(cache_path):
            os.makedirs(cache_path)
        INDEX_CACHE_PATH = os.path.join(cache_path, "index.pkl")
        DOCMAP_CACHE_PATH = os.path.join(cache_path, "docmap.pkl")
        TERM_FREQ_CACHE_PATH = os.path.join(cache_path, "term_frequencies.pkl")
        with open(INDEX_CACHE_PATH, "wb") as f:
            dump(self.index, f)
        with open(DOCMAP_CACHE_PATH, "wb") as f:
            dump(self.docmap, f)
        with open(TERM_FREQ_CACHE_PATH, "wb") as f:
            dump(self.term_frequencies, f)

    def load(self):
        cache_path = os.path.join(PROJECT_ROOT, "cache")
        INDEX_CACHE_PATH = os.path.join(cache_path, "index.pkl")
        DOCMAP_CACHE_PATH = os.path.join(cache_path, "docmap.pkl")
        TERM_FREQ_CACHE_PATH = os.path.join(cache_path, "term_frequencies.pkl")
        with open(INDEX_CACHE_PATH, "rb") as f:
            self.index = load(f)
        with open(DOCMAP_CACHE_PATH, "rb") as f:
            self.docmap = load(f)
        with open(TERM_FREQ_CACHE_PATH, "rb") as f:
            self.term_frequencies = load(f)


