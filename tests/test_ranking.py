"""Unit tests for pure ranking helpers (no heavy dependencies)."""
import pytest

from cli.lib.ranking import RRF_K, hybrid_score, normalize, normalize_scores, rrf_score


def test_rrf_score_matches_rank() -> None:
    assert rrf_score(1) == pytest.approx(1.0 / (RRF_K + 1))
    assert rrf_score(2) == pytest.approx(1.0 / (RRF_K + 2))
    assert rrf_score(1, k=0) == 1.0


def test_normalize_empty() -> None:
    assert normalize([]) == []


def test_normalize_all_equal_scales_to_one() -> None:
    assert normalize([0.5, 0.5, 0.5]) == [1.0, 1.0, 1.0]


def test_normalize_bounds_and_proportions() -> None:
    scores = [2.0, 4.0, 10.0]
    out = normalize(scores)
    assert out == pytest.approx([0.0, 0.25, 1.0])


def test_normalize_scores_zipped() -> None:
    fake = object()
    results = normalize_scores([(3.0, fake), (1.0, fake)])
    assert [s for s, _ in results] == pytest.approx([1.0, 0.0])


def test_hybrid_score_alpha() -> None:
    assert hybrid_score(1.0, 0.0, alpha=0.5) == pytest.approx(0.5)
    assert hybrid_score(1.0, 0.0, alpha=1.0) == 1.0
    assert hybrid_score(1.0, 0.0, alpha=0.0) == 0.0
