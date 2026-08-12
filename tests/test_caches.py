"""Tests for cache serialization helpers."""
import numpy as np

import cli.lib.caches as caches


def test_source_fingerprint_deterministic(tmp_path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"alpha-beta")
    assert caches.source_fingerprint(path) == caches.source_fingerprint(path)


def test_source_fingerprint_changes_on_content(tmp_path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"vvv1")
    first = caches.source_fingerprint(path)
    path.write_bytes(b"vvv2")
    assert caches.source_fingerprint(path) != first


def test_numpy_cache_validity(tmp_path) -> None:
    path = tmp_path / "arr.npy"
    arr = np.arange(12).reshape(3, 4)
    caches.save_numpy(path, arr)

    assert caches.load_numpy_if_valid(path, expected_rows=3) is not None
    assert caches.load_numpy_if_valid(path, expected_rows=99) is None
    assert caches.load_numpy_if_valid(tmp_path / "missing.npy") is None


def test_json_roundtrip(tmp_path) -> None:
    path = tmp_path / "meta.json"
    caches.save_json(path, {"fingerprint": "x", "chunks": [1, 2]})
    assert caches.load_json(path) == {"fingerprint": "x", "chunks": [1, 2]}
