"""Amazon Movies & TV (2023): the movie domain with real consumption order.

MovieLens-1M ratings were mostly entered in bulk on sign-up (53% of
consecutive ratings share a second), so its "next item" carries no
sequential signal. Amazon Movies & TV comes from the same McAuley-Lab
"Amazon-Reviews-2023" release as the Video Games domain (5-core, rating-only
ratings plus the raw per-category metadata) and has purchase/review events
days apart, like the games data.

The full file is about 7x the games domain (7.4M ratings, 657K users,
198K items), so preprocessing is two steps:

1. :func:`build_subset` (run once, locally): keep items with a title in the
   metadata, keep a hash-selected fraction of users, re-apply the 5-core
   filter and write a compact subset (ratings CSV + items JSONL with title and
   canonical genres). The user selection hashes the reviewer ID, so it does
   not depend on library versions or row order and any machine rebuilds the
   identical subset; its SHA-256 is recorded in the summary.
2. :func:`load_ratings` / :func:`load_items` read that subset with the same
   integer remapping as the games domain.

Only dataset fields are used: title and the ``categories`` list. Categories
mix genres with formats, studios and moods; :func:`movie_genres` keeps only
genres, mapped to a small canonical set.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .amazon_games import _stream_download, load_ratings_file

RATINGS_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/benchmark/5core/rating_only/Movies_and_TV.csv"
)
METADATA_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/raw/meta_categories/meta_Movies_and_TV.jsonl"
)

DATASET_SUBDIR = "amazon-movies"
RAW_RATINGS = "movies_tv_ratings.csv"
RAW_METADATA = "movies_tv_meta.jsonl"
SUBSET_RATINGS = "movies_tv_ratings_subset.csv"
SUBSET_ITEMS = "movies_tv_items_subset.jsonl"

# Category label -> canonical genre. Anything not listed (formats such as
# "Blu-ray", studios, store sections, moods such as "Cerebral") is dropped.
GENRE_MAP = {
    "Drama": "Drama",
    "Comedy": "Comedy",
    "Documentary": "Documentary",
    "Suspense": "Thriller",
    "Mystery & Thrillers": "Thriller",
    "Thriller": "Thriller",
    "Horror": "Horror",
    "Action": "Action & Adventure",
    "Adventure": "Action & Adventure",
    "Action & Adventure": "Action & Adventure",
    "Science Fiction": "Sci-Fi & Fantasy",
    "Fantasy": "Sci-Fi & Fantasy",
    "Science Fiction & Fantasy": "Sci-Fi & Fantasy",
    "Romance": "Romance",
    "Animation": "Animation",
    "Anime": "Animation",
    "Anime & Manga": "Animation",
    "Kids": "Kids & Family",
    "Kids & Family": "Kids & Family",
    "Family": "Kids & Family",
    "Special Interest": "Special Interest",
    "Special Interests": "Special Interest",
    "Exercise & Fitness": "Fitness",
    "Music Videos and Concerts": "Music & Performing Arts",
    "Music Videos & Concerts": "Music & Performing Arts",
    "Musicals & Performing Arts": "Music & Performing Arts",
    "Sports": "Sports",
    "Western": "Western",
    "Westerns": "Western",
    "Military and War": "War",
    "Military & War": "War",
    "Historical": "Historical",
    "International": "International",
    "Arthouse": "Arthouse",
    "Unscripted": "Unscripted",
    "LGBTQ": "LGBTQ",
    "Faith & Spirituality": "Faith & Spirituality",
}


def movie_genres(categories: list[str]) -> list[str]:
    """Canonical genres from an Amazon category list, in first-seen order."""

    genres: list[str] = []
    for label in categories:
        genre = GENRE_MAP.get(str(label).strip())
        if genre and genre not in genres:
            genres.append(genre)
    return genres


def keep_user(user_id: str, fraction: float, seed: int = 0) -> bool:
    """Deterministic, platform-independent user selection by hashed reviewer ID."""

    digest = hashlib.sha256(f"{seed}:{user_id}".encode()).hexdigest()
    return int(digest[:8], 16) < fraction * 2**32


def k_core(ratings: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Repeatedly drop items and users with fewer than ``k`` interactions."""

    while True:
        item_counts = ratings["parent_asin"].map(ratings["parent_asin"].value_counts())
        ratings = ratings[item_counts >= k]
        user_counts = ratings["user_id"].map(ratings["user_id"].value_counts())
        kept = ratings[user_counts >= k]
        if len(kept) == len(ratings):
            return kept
        ratings = kept


def download_amazon_movies(data_dir: str | Path, *, force: bool = False) -> Path:
    """Download the raw ratings (~430 MB) and metadata (~1.3 GB) once."""

    root = Path(data_dir) / DATASET_SUBDIR
    if force or not (root / RAW_RATINGS).is_file():
        _stream_download(RATINGS_URL, root / RAW_RATINGS)
    if force or not (root / RAW_METADATA).is_file():
        _stream_download(METADATA_URL, root / RAW_METADATA)
    return root


def build_subset(
    raw_dir: str | Path,
    *,
    user_fraction: float = 0.15,
    seed: int = 0,
    min_interactions: int = 5,
) -> dict[str, object]:
    """Write the compact subset files next to the raw files; return a summary."""

    raw_dir = Path(raw_dir)
    raw = pd.read_csv(
        raw_dir / RAW_RATINGS,
        dtype={"user_id": "string", "parent_asin": "string", "rating": "float64", "timestamp": "int64"},
    )
    asins = set(raw["parent_asin"].unique())

    items: dict[str, dict[str, object]] = {}
    with (raw_dir / RAW_METADATA).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            asin = record.get("parent_asin")
            title = " ".join(str(record.get("title") or "").split())
            if asin in asins and title:
                items[asin] = {"title": title, "genres": movie_genres(record.get("categories") or [])}

    with_title = raw[raw["parent_asin"].isin(items)]
    users = with_title["user_id"].unique()
    selected = {u for u in users if keep_user(str(u), user_fraction, seed)}
    subset = k_core(with_title[with_title["user_id"].isin(selected)], min_interactions)
    subset = subset.sort_values(["user_id", "timestamp", "parent_asin"], kind="mergesort")

    ratings_path = raw_dir / SUBSET_RATINGS
    subset.to_csv(ratings_path, index=False, columns=["user_id", "parent_asin", "rating", "timestamp"])
    kept_items = sorted(subset["parent_asin"].unique())
    with (raw_dir / SUBSET_ITEMS).open("w", encoding="utf-8") as handle:
        for asin in kept_items:
            handle.write(json.dumps({"parent_asin": asin, **items[asin]}, sort_keys=True) + "\n")

    gaps = subset.groupby("user_id")["timestamp"].diff().dropna() / 1000
    return {
        "raw_ratings": int(len(raw)),
        "raw_users": int(raw["user_id"].nunique()),
        "raw_items": int(len(asins)),
        "items_with_title": len(items),
        "user_fraction": user_fraction,
        "user_selection": f"sha256('{seed}:<user_id>')[:8] < fraction * 2^32",
        "min_interactions": min_interactions,
        "ratings": int(len(subset)),
        "users": int(subset["user_id"].nunique()),
        "items": len(kept_items),
        "items_with_genre": sum(bool(items[a]["genres"]) for a in kept_items),
        "same_second_gap_share": round(float((gaps == 0).mean()), 4),
        "over_one_day_gap_share": round(float((gaps > 86400).mean()), 4),
        "ratings_sha256": hashlib.sha256(ratings_path.read_bytes()).hexdigest(),
    }


def _dataset_directory(data_dir: str | Path) -> Path:
    root = Path(data_dir)
    for candidate in (root / DATASET_SUBDIR, root):
        if (candidate / SUBSET_RATINGS).is_file():
            return candidate
    raise FileNotFoundError(
        f"Amazon Movies & TV subset not found under {root}. "
        "Run scripts/prepare_amazon_movies.py first."
    )


def load_ratings(data_dir: str | Path, *, id_map_output: str | Path | None = None) -> pd.DataFrame:
    """Subset ratings with integer IDs; same schema as the games domain."""

    return load_ratings_file(_dataset_directory(data_dir) / SUBSET_RATINGS, id_map_output)


def load_items(data_dir: str | Path, item_id_map: dict[str, int]) -> pd.DataFrame:
    """Items as ``item_id``, ``title``, ``genres`` ('|'-joined canonical genres)."""

    rows = []
    with (_dataset_directory(data_dir) / SUBSET_ITEMS).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["parent_asin"] in item_id_map:
                rows.append({
                    "item_id": item_id_map[record["parent_asin"]],
                    "title": record["title"],
                    "genres": "|".join(record["genres"]),
                })
    return pd.DataFrame(rows, columns=["item_id", "title", "genres"]).astype(
        {"item_id": "int32", "title": "string", "genres": "string"}
    )
