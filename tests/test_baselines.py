"""Tests for the non-LLM reference baselines."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from llm_session_reco.baselines import (
    build_item_popularity,
    expected_random_metrics,
    load_item_popularity,
    popularity_ranking,
    random_ranking,
    rank_of_target,
    write_item_popularity,
)
from llm_session_reco.session_dataset import (
    build_candidate_pools,
    build_leave_one_out_examples,
    build_leave_one_out_training_ratings,
)


def test_popularity_ranking_orders_by_count_then_ascending_item_id():
    item_popularity = {10: 5, 20: 9, 30: 5, 40: 1}

    # 20 has the highest count; 10 and 30 tie at 5 and break by item ID.
    assert popularity_ranking([40, 30, 20, 10], item_popularity) == (20, 10, 30, 40)


def test_items_absent_from_training_rank_last():
    item_popularity = {10: 2}

    assert popularity_ranking([99, 10, 77], item_popularity) == (10, 77, 99)


def test_popularity_ranking_rejects_duplicate_candidates():
    with pytest.raises(ValueError, match="duplicate"):
        popularity_ranking([1, 2, 1], {1: 1, 2: 1})


def test_random_ranking_is_deterministic_per_session_and_seed():
    candidates = [1, 2, 3, 4, 5]
    first = random_ranking(candidates, session_id="user-7", seed=0)
    again = random_ranking(candidates, session_id="user-7", seed=0)
    other_session = random_ranking(candidates, session_id="user-8", seed=0)

    assert first == again
    assert sorted(first) == candidates
    assert first != other_session


def test_random_ranking_does_not_depend_on_processing_order():
    """A session's ranking must be identical when a subset is re-evaluated."""

    sessions = ["user-1", "user-2", "user-3"]
    full = {s: random_ranking([1, 2, 3, 4], session_id=s, seed=3) for s in sessions}
    subset = random_ranking([1, 2, 3, 4], session_id="user-3", seed=3)

    assert subset == full["user-3"]


def test_rank_of_target_is_one_indexed_and_raises_when_absent():
    assert rank_of_target((7, 8, 9), 9) == 3
    with pytest.raises(ValueError, match="not present"):
        rank_of_target((7, 8, 9), 42)


def test_build_and_round_trip_item_popularity(tmp_path):
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "item_id": [10, 20, 10, 30, 10, 20],
            "rating": [5, 4, 3, 5, 4, 2],
            "timestamp": [1, 2, 1, 2, 1, 2],
        }
    )
    training_ratings = build_leave_one_out_training_ratings(ratings)
    item_popularity = build_item_popularity(training_ratings)

    # Each user's final interaction is held out, so only the first remains.
    assert item_popularity == {10: 3}

    path = write_item_popularity(item_popularity, tmp_path / "pop.json")
    assert json.loads(path.read_text(encoding="utf-8")) == {"10": 3}
    assert load_item_popularity(path) == item_popularity


def test_popularity_ranks_the_target_last_on_popularity_built_pools():
    """The pools are popularity-adversarial by construction.

    Negatives are the most popular eligible training items, so a popularity
    ranker pushes the held-out target to the bottom of the pool. This test
    pins that property down: it is the reason the popularity row is a floor to
    report alongside random, not a competitor to beat.
    """

    head_items = [100, 101, 102, 103, 104, 105]
    rows = []
    for user_id in range(1, len(head_items) + 1):
        # Rotating prefixes keep every head item popular in training while
        # leaving some of them eligible as negatives for each user. Each user
        # ends on a rare item that training never sees.
        prefix = [head_items[user_id - 1], head_items[user_id % len(head_items)]]
        rare_target = 900 + user_id
        for position, item_id in enumerate(prefix + [rare_target]):
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "rating": 5,
                    "timestamp": position + 1,
                }
            )
    ratings = pd.DataFrame(rows)

    examples = build_leave_one_out_examples(ratings, min_history_length=1)
    training_ratings = build_leave_one_out_training_ratings(ratings)
    pools = build_candidate_pools(
        training_ratings, examples, candidate_pool_size=3, balance_target_positions=True
    )
    item_popularity = build_item_popularity(training_ratings)

    for pool in pools:
        ranking = popularity_ranking(pool.candidate_item_ids, item_popularity)
        assert rank_of_target(ranking, pool.target_item_id) == len(
            pool.candidate_item_ids
        )


def test_expected_random_metrics_matches_the_closed_form_for_20_item_pools():
    metrics = expected_random_metrics([20] * 50)

    assert metrics["HR@1"] == pytest.approx(0.05)
    assert metrics["HR@5"] == pytest.approx(0.25)
    assert metrics["HR@10"] == pytest.approx(0.50)
    assert metrics["NDCG@5"] == pytest.approx(0.147423, abs=1e-6)
    assert metrics["NDCG@10"] == pytest.approx(0.227178, abs=1e-6)


def test_expected_random_metrics_handles_pools_smaller_than_k():
    # With only 3 candidates the target is always inside the top 5 and top 10.
    metrics = expected_random_metrics([3])

    assert metrics["HR@1"] == pytest.approx(1 / 3)
    assert metrics["HR@5"] == pytest.approx(1.0)
    assert metrics["HR@10"] == pytest.approx(1.0)


def test_expected_random_metrics_rejects_empty_input():
    with pytest.raises(ValueError, match="cannot be empty"):
        expected_random_metrics([])
