"""Uniform access to each domain's ratings and item metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DOMAINS = ("movielens", "amazon_games", "amazon_movies", "amazon_film", "amazon_tv")

# Amazon Movies & TV domains: None is the combined category (Addendum v9);
# 'movie' and 'tv' are the two halves of the Addendum v10 split.
AMAZON_MOVIES_MEDIA = {"amazon_movies": None, "amazon_film": "movie", "amazon_tv": "tv"}


@dataclass(frozen=True)
class DomainData:
    name: str
    file_prefix: str
    ratings: pd.DataFrame
    item_titles: dict[int, str]
    item_genres: dict[int, set[str]]


def load_domain(
    domain: str,
    data_root: Path = Path("data/raw"),
    processed_root: Path = Path("data/processed"),
) -> DomainData:
    """Ratings plus titles and genre/platform sets for one domain.

    MovieLens reads the raw ml-1m files; Amazon Video Games reads the raw
    reviews plus the item file written by scripts/prepare_amazon_games.py.
    Amazon Movies & TV reads the subset and item file written by
    scripts/prepare_amazon_movies.py (canonical genres only); its movie-only
    and TV-only halves read the files written with ``--media``. Games genres
    drop the catch-all Amazon categories ("Video Games", "Games") so the
    remaining labels carry platform and sub-category information.
    """

    if domain == "movielens":
        from .movielens import load_movies, load_ratings

        ratings = load_ratings(data_root / "ml-1m")
        movies = load_movies(data_root / "ml-1m")
        titles = {int(i): str(t) for i, t in zip(movies["item_id"], movies["title"])}
        genres = {int(i): set(str(g).split("|")) for i, g in zip(movies["item_id"], movies["genres"])}
        return DomainData(domain, "ml1m", ratings, titles, genres)
    if domain == "amazon_games":
        from .amazon_games import load_ratings

        ratings = load_ratings(data_root / "amazon-games")
        titles, genres = {}, {}
        items_file = processed_root / "amazon_games_items.jsonl"
        with items_file.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                titles[int(record["item_id"])] = str(record["title"])
                genres[int(record["item_id"])] = set(str(record["genres"]).split("|")) - {
                    "Video Games",
                    "Games",
                }
        return DomainData(domain, "amazon_games", ratings, titles, genres)
    if domain in AMAZON_MOVIES_MEDIA:
        from .amazon_movies import load_ratings

        ratings = load_ratings(data_root, media=AMAZON_MOVIES_MEDIA[domain])
        titles, genres = {}, {}
        with (processed_root / f"{domain}_items.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                titles[int(record["item_id"])] = str(record["title"])
                genres[int(record["item_id"])] = {g for g in str(record["genres"]).split("|") if g}
        return DomainData(domain, domain, ratings, titles, genres)
    raise ValueError(f"Unknown domain {domain!r}; expected one of {DOMAINS}")
