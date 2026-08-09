from pickle import dump
import os
from collections import defaultdict
from typing import Callable
from cli.lib.keyword_search import tokenize_text
from cli.lib.search_utils import PROJECT_ROOT
from cli.lib.document import Document


class InvertedIndex:
    def __init__(self, load_documents: Callable[[], list[Document]]):
        self.load_documents = load_documents
        self.index: dict[str, list[str]] = defaultdict(list)
        self.docmap: dict[str, Document] = {}

    def __add_document(self, doc_id: str, text: str):
        text_tokens = tokenize_text(text)
        for token in text_tokens:
            self.index[token].append(doc_id)

    def get_documents(self, term: str) -> list[Document]:
        doc_ids = self.index.get(term, [])
        return [self.docmap[doc_id] for doc_id in doc_ids]

    def build(self):
        documents = self.load_documents()
        for doc in documents:
            self.__add_document(doc.get_id(), doc.to_text())
            self.docmap[doc.get_id()] = doc

    def save(self):
        cache_path = os.path.join(PROJECT_ROOT, "cache")
        if not os.path.isdir(cache_path):
            os.makedirs(cache_path)
        INDEX_CACHE_PATH = os.path.join(cache_path, "index.pkl")
        DOCMAP_CACHE_PATH = os.path.join(cache_path, "docmap.pkl")
        with open(INDEX_CACHE_PATH, "wb") as f:
            dump(self.index, f)
        with open(DOCMAP_CACHE_PATH, "wb") as f:
            dump(self.docmap, f)


