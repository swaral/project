import pandas as pd
import pytest

from llm_session_reco.movielens import (
    build_candidate_pools,
    build_leave_one_out_examples,
    build_leave_one_out_training_ratings,
)


def test_build_leave_one_out_examples_is_chronological_and_target_safe():
    ratings = pd.DataFrame(
        [
            {"user_id": 2, "item_id": 30, "timestamp": 30},
            {"user_id": 1, "item_id": 20, "timestamp": 20},
            {"user_id": 1, "item_id": 10, "timestamp": 10},
            {"user_id": 1, "item_id": 30, "timestamp": 30},
            {"user_id": 2, "item_id": 10, "timestamp": 10},
            {"user_id": 2, "item_id": 20, "timestamp": 20},
        ]
    )

    examples = build_leave_one_out_examples(ratings, min_history_length=2)

    assert [example.user_id for example in examples] == [1, 2]
    assert examples[0].prefix_item_ids == (10, 20)
    assert examples[0].target_item_id == 30
    assert examples[0].target_item_id not in examples[0].prefix_item_ids


def test_history_window_keeps_most_recent_items():
    ratings = pd.DataFrame(
        [
            {"user_id": 1, "item_id": item_id, "timestamp": item_id}
            for item_id in range(1, 6)
        ]
    )

    examples = build_leave_one_out_examples(
        ratings,
        min_history_length=2,
        max_history_length=2,
    )

    assert examples[0].prefix_item_ids == (3, 4)
    assert examples[0].target_item_id == 5


def test_short_histories_are_excluded():
    ratings = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 10, "timestamp": 1},
            {"user_id": 1, "item_id": 20, "timestamp": 2},
        ]
    )

    assert build_leave_one_out_examples(ratings, min_history_length=2) == []


def test_missing_columns_fail_loudly():
    with pytest.raises(ValueError, match="missing required columns"):
        build_leave_one_out_examples(pd.DataFrame({"user_id": [1]}))


def test_training_split_removes_each_users_final_interaction():
    ratings = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 10, "timestamp": 1},
            {"user_id": 1, "item_id": 20, "timestamp": 2},
            {"user_id": 1, "item_id": 30, "timestamp": 3},
            {"user_id": 2, "item_id": 40, "timestamp": 1},
            {"user_id": 2, "item_id": 50, "timestamp": 2},
            {"user_id": 2, "item_id": 60, "timestamp": 3},
        ]
    )

    training = build_leave_one_out_training_ratings(ratings)

    assert set(training["item_id"]) == {10, 20, 40, 50}


def test_candidate_pools_are_fixed_sized_target_safe_and_balanced():
    ratings = pd.DataFrame(
        [
            {"user_id": 1, "item_id": item_id, "timestamp": item_id}
            for item_id in range(1, 5)
        ]
        + [
            {"user_id": 2, "item_id": item_id, "timestamp": item_id}
            for item_id in range(5, 9)
        ]
    )
    examples = build_leave_one_out_examples(ratings, min_history_length=2)
    training = build_leave_one_out_training_ratings(ratings)

    pools = build_candidate_pools(
        training,
        examples,
        candidate_pool_size=4,
    )

    assert len(pools) == 2
    for index, pool in enumerate(pools):
        assert len(pool.candidate_item_ids) == 4
        assert len(set(pool.candidate_item_ids)) == 4
        assert pool.candidate_item_ids[pool.target_position] == pool.target_item_id
        assert pool.target_position == index % 4
        prefix = set(examples[index].prefix_item_ids)
        assert prefix.isdisjoint(pool.candidate_item_ids)
