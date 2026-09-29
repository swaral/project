"""Tests for the Amazon Movies & TV subset preprocessing."""

from __future__ import annotations

import json

import pandas as pd

from llm_session_reco.amazon_movies import (
    RAW_METADATA,
    RAW_RATINGS,
    build_subset,
    k_core,
    keep_user,
    load_items,
    load_ratings,
    media_type,
    movie_genres,
    runtime_minutes,
    subset_file_names,
)


def test_movie_genres_keep_only_canonical_genres_in_order():
    categories = ["Movies & TV", "Genre for Featured Categories", "Suspense", "Blu-ray",
                  "Mystery & Thrillers", "Cerebral", "Kids & Family", "Warner Home Video"]
    assert movie_genres(categories) == ["Thriller", "Kids & Family"]
    assert movie_genres([]) == []


def test_keep_user_is_deterministic_and_close_to_fraction():
    users = [f"U{i}" for i in range(20000)]
    kept = [u for u in users if keep_user(u, 0.15)]
    assert kept == [u for u in users if keep_user(u, 0.15)]
    assert 0.13 < len(kept) / len(users) < 0.17
    assert set(kept) != {u for u in users if keep_user(u, 0.15, seed=1)}


def test_k_core_iterates_until_stable():
    rows = [("a", "x"), ("a", "y"), ("b", "x"), ("b", "y"), ("c", "x")]
    frame = pd.DataFrame(rows, columns=["user_id", "parent_asin"])
    assert set(k_core(frame, 2)["user_id"]) == {"a", "b"}
    assert k_core(frame, 3).empty


def test_runtime_minutes_parses_amazon_formats():
    assert runtime_minutes({"Run time": "1 hour and 42 minutes"}) == 102
    assert runtime_minutes({"Runtime": "85 minutes"}) == 85
    assert runtime_minutes({"Run time": "3 hours"}) == 180
    assert runtime_minutes({}) is None
    assert runtime_minutes(None) is None


def _record(title, categories=(), runtime=None):
    details = {"Run time": runtime} if runtime else {}
    return {"title": title, "categories": ["Movies & TV", *categories], "details": details}


def test_media_type_explicit_signals_win():
    assert media_type(_record("Dollhouse", ["Blu-ray", "TV"])) == "tv"
    assert media_type(_record("Underbelly", ["Boxed Sets", "Television"], "9 hours")) == "tv"
    assert media_type(_record("Triple Dog", ["Blu-ray", "Movies"], "92 minutes")) == "movie"
    assert media_type(_record("South Park: Season 16", [], "5 hours")) == "tv"
    assert media_type(_record("Kath and Kim: Series 1", [], "3 hours")) == "tv"
    assert media_type(_record("The IT Crowd: The Complete Series")) == "tv"
    # Conflicting explicit signals are not guessed.
    assert media_type(_record("Father Brown Season 1", ["Movies"])) is None


def test_media_type_falls_back_to_run_time():
    assert media_type(_record("Kiss of the Dragon", [], "1 hour and 38 minutes")) == "movie"
    assert media_type(_record("Law & Order: The Eighth Year", [], "19 hours")) == "tv"
    assert media_type(_record("The Smurfs: Smurfy Hollow", [], "22 minutes")) is None
    assert media_type(_record("Die Hard")) is None
    # Long multi-film bundles are neither.
    assert media_type(_record("Hallmark Christmas Collection", [], "5 hours")) is None
    assert media_type(_record("Born To Fight / The Stranger", [], "5 hours")) is None
    assert media_type(_record("Jaws 4-Movie Pack", [], "7 hours")) is None
    # "Series" alone (e.g. a product line) is not a TV signal.
    assert media_type(_record("Collector's Series Edition", [], "2 hours")) == "movie"


def test_subset_file_names():
    assert subset_file_names(None) == ("movies_tv_ratings_subset.csv", "movies_tv_items_subset.jsonl")
    assert subset_file_names("tv") == ("tv_ratings_subset.csv", "tv_items_subset.jsonl")


def _write_raw(root):
    rows, meta = [], []
    for u in range(12):
        for i in range(6):
            rows.append({"user_id": f"U{u}", "parent_asin": f"I{i}", "rating": 4.0,
                         "timestamp": 1_600_000_000_000 + (u * 10 + i) * 86_400_000 * 2})
    rows.append({"user_id": "U0", "parent_asin": "NOTITLE", "rating": 5.0, "timestamp": 1})
    pd.DataFrame(rows).to_csv(root / RAW_RATINGS, index=False)
    for i in range(6):
        meta.append({"parent_asin": f"I{i}", "title": f"Film  {i}",
                     "categories": ["Movies & TV", "Comedy" if i % 2 else "Horror"]})
    meta.append({"parent_asin": "NOTITLE", "title": "", "categories": ["Drama"]})
    (root / RAW_METADATA).write_text("\n".join(json.dumps(m) for m in meta) + "\n")


def test_build_subset_round_trips_through_loaders(tmp_path):
    root = tmp_path / "amazon-movies"
    root.mkdir()
    _write_raw(root)
    summary = build_subset(root, user_fraction=1.0)
    assert summary["users"] == 12 and summary["items"] == 6
    assert summary["over_one_day_gap_share"] == 1.0
    first_hash = summary["ratings_sha256"]
    assert build_subset(root, user_fraction=1.0)["ratings_sha256"] == first_hash

    ratings = load_ratings(tmp_path, id_map_output=tmp_path / "map.json")
    assert list(ratings.columns) == ["user_id", "item_id", "rating", "timestamp"]
    item_map = json.loads((tmp_path / "map.json").read_text())["item_id_map"]
    items = load_items(tmp_path, item_map)
    assert "NOTITLE" not in item_map
    assert set(items["title"]) == {f"Film {i}" for i in range(6)}
    assert set(items["genres"]) == {"Comedy", "Horror"}


def test_build_subset_by_media_keeps_only_that_type(tmp_path):
    root = tmp_path / "amazon-movies"
    root.mkdir()
    rows, meta = [], []
    for u in range(12):
        for i in range(12):
            rows.append({"user_id": f"U{u}", "parent_asin": f"I{i}", "rating": 5.0,
                         "timestamp": 1_600_000_000_000 + (u * 20 + i) * 86_400_000})
    pd.DataFrame(rows).to_csv(root / RAW_RATINGS, index=False)
    for i in range(12):
        # Even items are films, odd items are TV seasons.
        title = f"Film {i}" if i % 2 == 0 else f"Show {i}: Season 1"
        meta.append({"parent_asin": f"I{i}", "title": title, "categories": ["Movies & TV", "Drama"],
                     "details": {"Run time": "1 hour and 30 minutes"}})
    (root / RAW_METADATA).write_text("\n".join(json.dumps(m) for m in meta) + "\n")

    tv = build_subset(root, user_fraction=1.0, media="tv")
    film = build_subset(root, user_fraction=1.0, media="movie")
    assert tv["items"] == film["items"] == 6
    assert tv["titled_items_by_media"] == {"movie": 6, "tv": 6, "unknown": 0}
    # The combined subset files are untouched by the split.
    assert not (root / "movies_tv_ratings_subset.csv").exists()

    item_map_path = tmp_path / "tv_map.json"
    ratings = load_ratings(tmp_path, id_map_output=item_map_path, media="tv")
    items = load_items(tmp_path, json.loads(item_map_path.read_text())["item_id_map"], media="tv")
    assert len(ratings) == 12 * 6
    assert all(title.startswith("Show") for title in items["title"])
