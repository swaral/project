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

Only dataset fields are used: title, the ``categories`` list and, for the
movie/TV split, run time from ``details``. Categories mix genres with formats,
studios and moods; :func:`movie_genres` keeps only genres, mapped to a small
canonical set.

Addendum v10 splits the category into two domains, movies and TV, with
:func:`media_type`. Amazon has no movie/TV field, so the label comes from
explicit categories, then TV words in the title, then run time; items with no
usable signal, or with conflicting signals, are dropped from both domains.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from .amazon_games import _stream_download, load_ratings_file
from .item_metadata import FIELD_NAMES, item_fields, items_frame, store_format, subgenre_below

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

MEDIA_TYPES = ("movie", "tv")
_TV_CATEGORIES = frozenset({"TV", "Television"})
_MOVIE_CATEGORIES = frozenset({"Movies"})
_TV_TITLE = re.compile(
    r"\b(seasons?|complete series|series\s+(\d+|[ivx]+|one|two|three)|episodes?"
    r"|mini-?series|tv series|tv show)\b",
    re.IGNORECASE,
)
# Multi-film bundles have long total run times but are not TV.
_COLLECTION_TITLE = re.compile(
    r"\b(collection|trilogy|double feature|triple feature|\d+[- ](movies?|films?))\b| / |/.*/",
    re.IGNORECASE,
)
MIN_FEATURE_MINUTES = 40  # shorter items are mostly single TV episodes or clips
MAX_FEATURE_MINUTES = 240  # longer items are mostly season or series box sets

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


_TITLE_FORMAT = re.compile(r"(4K Ultra HD|4K UHD|Blu-ray|DVD|VHS)", re.IGNORECASE)


def movie_format(record: dict[str, object]) -> str:
    """How a Movies & TV item is sold: store format, title tag, Prime Video, details."""

    label = store_format(record.get("store"))
    if label:
        return label
    match = _TITLE_FORMAT.search(str(record.get("title") or ""))
    if match:
        found = match.group(1).lower()
        return {"4k ultra hd": "4K UHD", "4k uhd": "4K UHD", "blu-ray": "Blu-ray",
                "dvd": "DVD", "vhs": "VHS Tape"}[found]
    if record.get("main_category") == "Prime Video":
        return "Prime Video"
    media = str((record.get("details") or {}).get("Media Format") or "")
    for label in ("Blu-ray", "DVD"):
        if label in media:
            return label
    return ""


def movie_item_fields(record: dict[str, object]) -> dict[str, object]:
    """Experiment 2b fields (see item_metadata): sub-genre below the genre, format, price."""

    return item_fields(
        subgenre_below(record.get("categories") or [], set(GENRE_MAP)),
        movie_format(record),
        record.get("price"),
    )


def movie_genres(categories: list[str]) -> list[str]:
    """Canonical genres from an Amazon category list, in first-seen order."""

    genres: list[str] = []
    for label in categories:
        genre = GENRE_MAP.get(str(label).strip())
        if genre and genre not in genres:
            genres.append(genre)
    return genres


def runtime_minutes(details: dict[str, object] | None) -> int | None:
    """Run time in minutes from a metadata ``details`` dict, e.g. '1 hour and 42 minutes'."""

    details = details or {}
    text = str(details.get("Run time") or details.get("Runtime") or "")
    hours = re.search(r"(\d+)\s*hour", text)
    minutes = re.search(r"(\d+)\s*min", text)
    total = (int(hours.group(1)) * 60 if hours else 0) + (int(minutes.group(1)) if minutes else 0)
    return total or None


def media_type(record: dict[str, object]) -> str | None:
    """'movie', 'tv' or None (unknown) for one raw metadata record.

    An explicit TV/Television or Movies category, or TV words in the title
    ('Season 2', 'Complete Series', 'Series 1', 'Episodes'), decide first;
    conflicting explicit signals give None. Otherwise run time decides: under
    40 minutes is None (episodes and clips), up to 4 hours is a movie, and
    longer is TV unless the title marks a multi-film collection (None).
    """

    title = str(record.get("title") or "")
    categories = {str(c).strip() for c in record.get("categories") or []}
    tv = bool(categories & _TV_CATEGORIES) or bool(_TV_TITLE.search(title))
    movie = bool(categories & _MOVIE_CATEGORIES)
    if tv and movie:
        return None
    if tv:
        return "tv"
    if movie:
        return "movie"
    minutes = runtime_minutes(record.get("details"))
    if minutes is None or minutes < MIN_FEATURE_MINUTES:
        return None
    if minutes > MAX_FEATURE_MINUTES:
        return None if _COLLECTION_TITLE.search(title) else "tv"
    return "movie"


def subset_file_names(media: str | None) -> tuple[str, str]:
    """(ratings CSV, items JSONL) names for the combined set or one media type."""

    if media is None:
        return SUBSET_RATINGS, SUBSET_ITEMS
    if media not in MEDIA_TYPES:
        raise ValueError(f"media must be one of {MEDIA_TYPES} or None, got {media!r}")
    return f"{media}_ratings_subset.csv", f"{media}_items_subset.jsonl"


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
    media: str | None = None,
) -> dict[str, object]:
    """Write the compact subset files next to the raw files; return a summary.

    With ``media`` set to 'movie' or 'tv', only items of that type (see
    :func:`media_type`) are kept, before user selection and the 5-core
    filter, so each domain is filtered on its own interactions.
    """

    ratings_name, items_name = subset_file_names(media)
    raw_dir = Path(raw_dir)
    raw = pd.read_csv(
        raw_dir / RAW_RATINGS,
        dtype={"user_id": "string", "parent_asin": "string", "rating": "float64", "timestamp": "int64"},
    )
    asins = set(raw["parent_asin"].unique())

    items: dict[str, dict[str, object]] = {}
    media_counts = {"movie": 0, "tv": 0, "unknown": 0}
    with (raw_dir / RAW_METADATA).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            asin = record.get("parent_asin")
            title = " ".join(str(record.get("title") or "").split())
            if asin in asins and title:
                kind = media_type(record)
                media_counts[kind or "unknown"] += 1
                if media is not None and kind != media:
                    continue
                items[asin] = {
                    "title": title,
                    "genres": movie_genres(record.get("categories") or []),
                    **movie_item_fields(record),
                }

    return {
        "raw_ratings": int(len(raw)),
        "raw_users": int(raw["user_id"].nunique()),
        "raw_items": int(len(asins)),
        "media": media or "all",
        "titled_items_by_media": media_counts,
        "items_with_title": len(items),
        **write_subset(
            raw, items, raw_dir / ratings_name, raw_dir / items_name,
            user_fraction=user_fraction, seed=seed, min_interactions=min_interactions,
        ),
    }


def write_subset(
    raw: pd.DataFrame,
    items: dict[str, dict[str, object]],
    ratings_path: Path,
    items_path: Path,
    *,
    user_fraction: float,
    seed: int,
    min_interactions: int,
) -> dict[str, object]:
    """Keep items in ``items``, hash-select users, re-5-core, write both files.

    Shared by every sampled Amazon domain so they all apply the same rule.
    ``items`` maps ASIN to the record written to the items file.
    """

    with_title = raw[raw["parent_asin"].isin(items)]
    users = with_title["user_id"].unique()
    selected = {u for u in users if keep_user(str(u), user_fraction, seed)}
    subset = k_core(with_title[with_title["user_id"].isin(selected)], min_interactions)
    subset = subset.sort_values(["user_id", "timestamp", "parent_asin"], kind="mergesort")

    # Unix line endings on every platform, so the SHA-256 below is identical
    # on Windows and on Kaggle (pandas otherwise writes "\r\n" on Windows).
    subset.to_csv(
        ratings_path, index=False, columns=["user_id", "parent_asin", "rating", "timestamp"],
        lineterminator="\n",
    )
    kept_items = sorted(subset["parent_asin"].unique())
    with items_path.open("w", encoding="utf-8", newline="\n") as handle:
        for asin in kept_items:
            handle.write(json.dumps({"parent_asin": asin, **items[asin]}, sort_keys=True) + "\n")

    gaps = subset.groupby("user_id")["timestamp"].diff().dropna() / 1000
    return {
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


def _dataset_directory(data_dir: str | Path, ratings_name: str = SUBSET_RATINGS) -> Path:
    root = Path(data_dir)
    for candidate in (root / DATASET_SUBDIR, root):
        if (candidate / ratings_name).is_file():
            return candidate
    raise FileNotFoundError(
        f"Amazon Movies & TV subset {ratings_name} not found under {root}. "
        "Run scripts/prepare_amazon_movies.py first."
    )


def load_ratings(
    data_dir: str | Path,
    *,
    id_map_output: str | Path | None = None,
    media: str | None = None,
) -> pd.DataFrame:
    """Subset ratings with integer IDs; same schema as the games domain."""

    ratings_name, _ = subset_file_names(media)
    return load_ratings_file(_dataset_directory(data_dir, ratings_name) / ratings_name, id_map_output)


def load_items(
    data_dir: str | Path, item_id_map: dict[str, int], *, media: str | None = None
) -> pd.DataFrame:
    """Items as ``item_id``, ``title``, ``genres`` ('|'-joined canonical genres)
    plus the Experiment 2b fields (``item_metadata.FIELD_NAMES``)."""

    ratings_name, items_name = subset_file_names(media)
    rows = []
    with (_dataset_directory(data_dir, ratings_name) / items_name).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["parent_asin"] in item_id_map:
                rows.append({
                    "item_id": item_id_map[record["parent_asin"]],
                    "title": record["title"],
                    "genres": "|".join(record["genres"]),
                    **{field: record[field] for field in FIELD_NAMES if field in record},
                })
    return items_frame(rows)
