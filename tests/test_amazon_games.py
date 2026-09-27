import json

import pandas as pd
import pytest

from llm_session_reco.amazon_games import build_id_map, load_items, load_ratings
from llm_session_reco.session_dataset import build_leave_one_out_examples


def test_build_id_map_is_sorted_dense_and_deterministic():
    id_map = build_id_map(pd.Series(["B002", "A001", "B002", "C003"]))
    assert id_map == {"A001": 1, "B002": 2, "C003": 3}


def _write_ratings_fixture(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "user_id,parent_asin,rating,timestamp\n"
        "userB,itemY,5.0,2000000\n"
        "userA,itemX,4.0,1000000\n"
        "userA,itemY,3.0,3000000\n",
        encoding="utf-8",
    )


def _write_metadata_fixture(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {"parent_asin": "itemX", "title": "Game X", "categories": ["Video Games", "PC"]},
        {"parent_asin": "itemY", "title": "Game Y", "categories": ["Video Games"]},
        {"parent_asin": "itemZ", "title": "Unused Game Z", "categories": []},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def test_load_ratings_remaps_string_ids_to_dense_integers(tmp_path):
    dataset_dir = tmp_path / "amazon-games"
    _write_ratings_fixture(dataset_dir / "video_games_ratings.csv")

    ratings = load_ratings(tmp_path)

    assert set(ratings.columns) == {"user_id", "item_id", "rating", "timestamp"}
    assert ratings["user_id"].nunique() == 2
    assert ratings["item_id"].nunique() == 2
    # timestamps convert from milliseconds to seconds
    assert sorted(ratings["timestamp"].tolist()) == [1000, 2000, 3000]


def test_load_ratings_writes_id_map_when_requested(tmp_path):
    dataset_dir = tmp_path / "amazon-games"
    _write_ratings_fixture(dataset_dir / "video_games_ratings.csv")
    id_map_path = tmp_path / "processed" / "id_map.json"

    load_ratings(tmp_path, id_map_output=id_map_path)

    saved = json.loads(id_map_path.read_text(encoding="utf-8"))
    assert saved["user_id_map"] == {"userA": 1, "userB": 2}
    assert saved["item_id_map"] == {"itemX": 1, "itemY": 2}


def test_load_items_keeps_only_mapped_items_and_joins_categories(tmp_path):
    dataset_dir = tmp_path / "amazon-games"
    _write_ratings_fixture(dataset_dir / "video_games_ratings.csv")
    _write_metadata_fixture(dataset_dir / "video_games_meta.jsonl")
    item_id_map = {"itemX": 1, "itemY": 2}  # itemZ intentionally excluded

    items = load_items(tmp_path, item_id_map)

    assert sorted(items["item_id"].tolist()) == [1, 2]
    row_x = items[items["item_id"] == 1].iloc[0]
    assert row_x["title"] == "Game X"
    assert row_x["genres"] == "Video Games|PC"


def test_load_items_raises_for_missing_metadata_file(tmp_path):
    (tmp_path / "amazon-games").mkdir()
    (tmp_path / "amazon-games" / "video_games_ratings.csv").write_text(
        "user_id,parent_asin,rating,timestamp\n", encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError):
        load_items(tmp_path, {"itemX": 1})


def test_remapped_ratings_work_with_the_generic_session_dataset_builder(tmp_path):
    dataset_dir = tmp_path / "amazon-games"
    _write_ratings_fixture(dataset_dir / "video_games_ratings.csv")
    ratings = load_ratings(tmp_path)

    # session_dataset.build_leave_one_out_examples is dataset-agnostic and
    # already covered by tests/test_movielens.py; this just proves the
    # column names/dtypes produced by load_ratings are compatible with it.
    examples = build_leave_one_out_examples(ratings, min_history_length=1)

    assert len(examples) == 1  # only userA has more than 1 interaction
    assert examples[0].target_item_id not in examples[0].prefix_item_ids
