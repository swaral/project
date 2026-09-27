"""MovieLens-1M loading and deterministic session-example construction."""

from __future__ import annotations

from pathlib import Path
import shutil
from urllib.request import urlopen
import zipfile

import pandas as pd

from .session_dataset import (
    CandidatePool,
    SessionExample,
    build_candidate_pools,
    build_leave_one_out_examples,
    build_leave_one_out_training_ratings,
    write_candidate_pools_jsonl,
    write_jsonl,
)

__all__ = [
    "CandidatePool",
    "SessionExample",
    "build_candidate_pools",
    "build_leave_one_out_examples",
    "build_leave_one_out_training_ratings",
    "write_candidate_pools_jsonl",
    "write_jsonl",
    "MOVIELENS_1M_URL",
    "download_movielens_1m",
    "load_ratings",
    "load_movies",
]


MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
_REQUIRED_FILES = ("ratings.dat", "movies.dat", "users.dat")


def _dataset_directory(data_dir: str | Path) -> Path:
    """Resolve either ``data_dir`` or ``data_dir/ml-1m``."""

    root = Path(data_dir)
    nested = root / "ml-1m"
    if all((nested / filename).is_file() for filename in _REQUIRED_FILES):
        return nested
    if all((root / filename).is_file() for filename in _REQUIRED_FILES):
        return root
    raise FileNotFoundError(
        f"MovieLens-1M files were not found under {root}. "
        "Run scripts/prepare_movielens.py to download the dataset."
    )


def _safe_extract(archive_path: Path, destination: Path) -> None:
    """Extract an archive while rejecting paths outside ``destination``."""

    destination = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if not member_path.is_relative_to(destination):
                raise ValueError(f"Unsafe archive member: {member.filename}")
        archive.extractall(destination)


def download_movielens_1m(
    data_dir: str | Path,
    *,
    url: str = MOVIELENS_1M_URL,
    force: bool = False,
) -> Path:
    """Download and extract MovieLens-1M, returning its dataset directory."""

    root = Path(data_dir)
    dataset_dir = root / "ml-1m"
    if not force and all((dataset_dir / filename).is_file() for filename in _REQUIRED_FILES):
        return dataset_dir

    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "ml-1m.zip"
    if force or not archive_path.is_file():
        with urlopen(url, timeout=120) as response, archive_path.open("wb") as output:
            shutil.copyfileobj(response, output)

    _safe_extract(archive_path, root)
    if not all((dataset_dir / filename).is_file() for filename in _REQUIRED_FILES):
        raise FileNotFoundError(
            f"Downloaded archive did not contain the expected MovieLens-1M files: {dataset_dir}"
        )
    return dataset_dir


def load_ratings(data_dir: str | Path) -> pd.DataFrame:
    """Load ratings with stable column names and integer identifiers."""

    dataset_dir = _dataset_directory(data_dir)
    return pd.read_csv(
        dataset_dir / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={
            "user_id": "int32",
            "item_id": "int32",
            "rating": "float32",
            "timestamp": "int64",
        },
        header=None,
    )


def load_movies(data_dir: str | Path) -> pd.DataFrame:
    """Load movie metadata using the dataset's latin-1 encoding."""

    dataset_dir = _dataset_directory(data_dir)
    return pd.read_csv(
        dataset_dir / "movies.dat",
        sep="::",
        engine="python",
        names=["item_id", "title", "genres"],
        dtype={"item_id": "int32", "title": "string", "genres": "string"},
        encoding="latin-1",
        header=None,
    )
