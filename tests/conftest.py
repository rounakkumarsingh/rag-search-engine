"""Shared fixtures/helpers for the test suite."""
import numpy as np
import pytest

import cli.lib.search_utils as su


@pytest.fixture(autouse=True)
def _seed_stopwords(monkeypatch):
    # Keep stopword lookups off the filesystem (the worktree has no data dir);
    # individual tests may override with their own lists.
    monkeypatch.setattr(su, "STOPWORDS", ["zzz"])


class FakeDoc:
    def __init__(self, i: int, title: str | None = None, desc: str | None = None):
        self.i = i
        self._title = title or f"Title {i}"
        self._desc = desc or f"Description {i}"

    def get_id(self) -> str:
        return str(self.i)

    def get_title(self) -> str:
        return self._title

    def get_description(self) -> str:
        return self._desc

    def to_text(self) -> str:
        return f"{self._title} {self._desc}"

    def get_semantic_text(self) -> str:
        return self._desc


class FakeEmbedder:
    def __init__(self, dim: int = 8):
        self.dim = dim
        self.encode_calls = 0
        self.last_encoded: list[str] = []

    def encode(self, texts, show_progress_bar: bool = False):
        self.encode_calls += 1
        self.last_encoded = list(texts)
        return np.ones((len(texts), self.dim), dtype=float)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "FakeEmbedder"


@pytest.fixture
def fake_docs() -> list[FakeDoc]:
    return [
        FakeDoc(0, "Bear Attack", "A grizzly bear stalks hikers in the woods."),
        FakeDoc(1, "London Comedy", "A marmalade-loving bear causes trouble in London."),
        FakeDoc(2, "Space Drama", "Astronauts fight to survive deep in space."),
    ]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
