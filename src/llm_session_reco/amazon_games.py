"""Amazon Video Games (2023) loading: a second domain for cross-domain comparison.

Ratings come from the McAuley-Lab "Amazon-Reviews-2023" 5-core rating-only
benchmark file (already deduplicated <user, item> interactions with a minimum
of 5 interactions per user and per item -- the same "eligible users have
enough history" spirit as MovieLens' ``min_history_length``). Item metadata
(title, category) comes from the corresponding raw per-category metadata
file. Both are downloaded directly over HTTPS; no ``datasets`` library
dependency is needed.

Amazon user/item identifiers are strings (reviewer ID / ASIN). The rest of
this project's pipeline (parser, prompts, candidate pools) assumes integer
item IDs, so ratings and metadata are remapped to deterministic synthetic
integer IDs (sorted string -> dense index) and the mapping can be persisted
for traceability via ``load_ratings(..., id_map_output=...)``.

Source: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
(benchmark/5core/rating_only/Video_Games.csv, ~47 MB; raw/meta_categories/
meta_Video_Games.jsonl, ~417 MB -- no smaller metadata-only file is published
for this category, so the metadata download is the larger of the two).
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

RATINGS_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/benchmark/5core/rating_only/Video_Games.csv"
)
METADATA_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/raw/meta_categories/meta_Video_Games.jsonl"
)

_RATINGS_FILENAME = "video_games_ratings.csv"
_METADATA_FILENAME = "video_games_meta.jsonl"
_DATASET_SUBDIR = "amazon-games"


def _stream_download(url: str, destination: Path, *, timeout: int = 300) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    with urlopen(url, timeout=timeout) as response, tmp_path.open("wb") as output:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            output.write(chunk)
    tmp_path.replace(destination)


def download_amazon_games(data_dir: str | Path, *, force: bool = False) -> Path:
    """Download the Amazon Video Games ratings-only CSV and item metadata JSONL.

    Returns the directory the raw files were written to. This is a one-time,
    roughly 465 MB combined download (47 MB ratings + 417 MB metadata); no
    smaller metadata-only file is published for this category.
    """

    root = Path(data_dir) / _DATASET_SUBDIR
    ratings_path = root / _RATINGS_FILENAME
    metadata_path = root / _METADATA_FILENAME
    if force or not ratings_path.is_file():
        _stream_download(RATINGS_URL, ratings_path)
    if force or not metadata_path.is_file():
        _stream_download(METADATA_URL, metadata_path)
    return root


def _dataset_directory(data_dir: str | Path) -> Path:
    """Resolve either ``data_dir`` or ``data_dir/amazon-games``."""

    root = Path(data_dir)
    nested = root / _DATASET_SUBDIR
    if (nested / _RATINGS_FILENAME).is_file():
        return nested
    if (root / _RATINGS_FILENAME).is_file():
        return root
    raise FileNotFoundError(
        f"Amazon Video Games files were not found under {root}. "
        "Run scripts/prepare_amazon_games.py to download the dataset."
    )


def build_id_map(raw_ids: pd.Series) -> dict[str, int]:
    """Map raw string IDs to deterministic, sorted, dense integer IDs.

    Sorting first makes the mapping reproducible across runs regardless of
    the order rows appear in the source file.
    """

    unique_ids = sorted(str(value) for value in set(raw_ids))
    return {raw_id: index for index, raw_id in enumerate(unique_ids, start=1)}


def load_ratings(
    data_dir: str | Path,
    *,
    id_map_output: str | Path | None = None,
) -> pd.DataFrame:
    """Load Amazon Video Games ratings with synthetic integer user/item IDs.

    Returns a DataFrame with the same column schema as
    :func:`llm_session_reco.movielens.load_ratings` (``user_id``, ``item_id``,
    ``rating``, ``timestamp``), so the dataset-agnostic session and
    candidate-pool builders in :mod:`llm_session_reco.session_dataset` work
    unchanged. Timestamps are converted from milliseconds (the source format)
    to seconds, matching MovieLens' unit.

    When ``id_map_output`` is given, the user/item ID maps are written there
    as JSON (``{"user_id_map": {...}, "item_id_map": {...}}``) so
    :func:`load_items` and downstream trial records can trace synthetic IDs
    back to the original Amazon reviewer ID / ASIN.
    """

    return load_ratings_file(_dataset_directory(data_dir) / _RATINGS_FILENAME, id_map_output)


def load_ratings_file(path: str | Path, id_map_output: str | Path | None = None) -> pd.DataFrame:
    """Read an Amazon-Reviews-2023 rating-only CSV; see :func:`load_ratings`."""

    raw = pd.read_csv(
        path,
        dtype={"user_id": "string", "parent_asin": "string", "rating": "float64"},
    )

    user_id_map = build_id_map(raw["user_id"])
    item_id_map = build_id_map(raw["parent_asin"])

    ratings = pd.DataFrame(
        {
            "user_id": raw["user_id"].map(user_id_map).astype("int32"),
            "item_id": raw["parent_asin"].map(item_id_map).astype("int32"),
            "rating": raw["rating"].astype("float32"),
            "timestamp": (raw["timestamp"].astype("int64") // 1000),
        }
    )

    if id_map_output is not None:
        output_path = Path(id_map_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {"user_id_map": user_id_map, "item_id_map": item_id_map}, indent=2
            ),
            encoding="utf-8",
        )

    return ratings


def load_items(data_dir: str | Path, item_id_map: dict[str, int]) -> pd.DataFrame:
    """Load item metadata (title, category) for the mapped integer item IDs.

    Only items present in ``item_id_map`` (i.e. items that appear in the
    ratings file) are kept; the metadata file is streamed line by line so the
    full ~137K-item, ~417 MB catalog is never held in memory at once.

    Returns a DataFrame with columns ``item_id``, ``title``, ``genres`` --
    the same column names as :func:`llm_session_reco.movielens.load_movies`,
    with Amazon's product ``categories`` standing in for MovieLens genres, so
    ``prompts.render_baseline_prompt``'s ``item_genres`` argument works
    unchanged for either domain.
    """

    dataset_dir = _dataset_directory(data_dir)
    rows: list[dict[str, object]] = []
    with (dataset_dir / _METADATA_FILENAME).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            parent_asin = record.get("parent_asin")
            if parent_asin not in item_id_map:
                continue
            title = str(record.get("title") or "").strip() or f"item {parent_asin}"
            categories = record.get("categories") or []
            category = "|".join(str(value) for value in categories) or "(unknown)"
            rows.append(
                {
                    "item_id": item_id_map[parent_asin],
                    "title": title,
                    "genres": category,
                }
            )

    return pd.DataFrame(rows, columns=["item_id", "title", "genres"]).astype(
        {"item_id": "int32", "title": "string", "genres": "string"}
    )
