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
    movie_genres,
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
