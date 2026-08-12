"""Tests for keyword preprocessing determinism and dedup."""
import pytest

import cli.lib.search_utils as su


def _seed_stopwords(monkeypatch, words: list[str]) -> None:
    # Avoid hitting the stopwords file: pre-populate the module-level list.
    monkeypatch.setattr(su, "STOPWORDS", list(words))


def test_tokenize_text_order_preserving_dedup(monkeypatch) -> None:
    _seed_stopwords(monkeypatch, [])
    tokens = su.tokenize_text("Alpha beta Beta gamma alpha")
    assert tokens == ["alpha", "beta", "gamma"]


def test_tokenize_text_is_deterministic(monkeypatch) -> None:
    _seed_stopwords(monkeypatch, [])
    text = "The fast and the furious rooftop chase"
    assert su.tokenize_text(text) == su.tokenize_text(text)


def test_tokenize_text_drops_stopwords(monkeypatch) -> None:
    _seed_stopwords(monkeypatch, ["the", "a"])
    tokens = su.tokenize_text("A the bear movie")
    assert tokens == ["bear", "movi"]


def test_tokenize_single_term_rejects_multiple(monkeypatch) -> None:
    _seed_stopwords(monkeypatch, [])
    with pytest.raises(ValueError):
        su.tokenize_single_term("two words")
