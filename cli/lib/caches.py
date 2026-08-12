import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from cli.lib.config import CACHE_DIR, DATA_PATH


PathLike = str | Path


def _as_path(path: PathLike) -> Path:
    return path if isinstance(path, Path) else Path(path)


def save_pickle(path: PathLike, obj: Any) -> None:
    path = _as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: PathLike) -> Any:
    with open(_as_path(path), "rb") as f:
        return pickle.load(f)


def save_numpy(path: PathLike, array: np.ndarray) -> None:
    path = _as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def load_numpy(path: PathLike) -> np.ndarray:
    return np.load(_as_path(path))


def save_json(path: PathLike, obj: Any) -> None:
    path = _as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: PathLike) -> Any:
    with open(_as_path(path)) as f:
        return json.load(f)


def exists_all(paths: list[PathLike]) -> bool:
    return all(Path(path).exists() for path in paths)


def load_numpy_if_valid(path: PathLike, expected_rows: int | None = None) -> np.ndarray | None:
    path = _as_path(path)
    if not path.exists():
        return None
    array = load_numpy(path)
    if expected_rows is not None and array.shape[0] != expected_rows:
        return None
    return array


def source_fingerprint(path: PathLike | None = None) -> str:
    """SHA-256 of the source data file, used to invalidate stale caches."""
    path = _as_path(path if path is not None else DATA_PATH)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_dir() -> Path:
    return CACHE_DIR
