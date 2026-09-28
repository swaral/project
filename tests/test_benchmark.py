"""Tests for the repaired benchmark: clean targets, matched and retrieval pools."""

from __future__ import annotations

import numpy as np
import pandas as pd

from llm_session_reco.benchmark import (
    BenchmarkExample,
    ItemKNNRetriever,
    assign_split,
    build_clean_examples,
    build_popularity_matched_pools,
    build_retrieval_pools,
    popularity_midrank,
)
from llm_session_reco.metrics import tie_aware_metrics


def _ratings(rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def test_clean_examples_keep_only_positive_uniquely_last_targets():
    ratings = _ratings([
        # user 1: clean positive target
        (1, 10, 5, 1), (1, 11, 3, 2), (1, 12, 4, 3), (1, 13, 5, 4),
        # user 2: final rating is negative
        (2, 10, 5, 1), (2, 11, 5, 2), (2, 12, 5, 3), (2, 13, 2, 4),
        # user 3: final timestamp tied with the previous rating
        (3, 10, 5, 1), (3, 11, 5, 2), (3, 12, 5, 3), (3, 13, 5, 3),
        # user 4: too short
        (4, 10, 5, 1), (4, 11, 5, 2),
    ])
    examples, dropped = build_clean_examples(ratings, validation_fraction=0.0)

    assert [e.session_id for e in examples] == ["user-1"]
    assert examples[0].target_item_id == 13
    assert examples[0].prefix_item_ids == (10, 11, 12)
    assert examples[0].split == "test"
    assert dropped == {
        "short_history": 1,
        "non_positive_target": 1,
        "tied_final_timestamp": 1,
        "target_in_history": 0,
    }


def test_split_assignment_is_deterministic_and_respects_the_fraction():
    splits = [assign_split(f"user-{i}", validation_fraction=0.3, seed=0) for i in range(4000)]
    assert splits == [assign_split(f"user-{i}", validation_fraction=0.3, seed=0) for i in range(4000)]
    assert 0.27 < splits.count("validation") / len(splits) < 0.33


def _example(i, prefix, target):
    return BenchmarkExample(
        user_id=i, session_id=f"user-{i}", prefix_item_ids=tuple(prefix),
        target_item_id=target, target_timestamp=0, target_rating=5.0, split="test",
    )


def test_matched_pools_remove_the_popularity_shortcut():
    catalog = list(range(1, 2001))
    popularity = {item: item for item in catalog}  # distinct counts, item i has count i
    rng = np.random.default_rng(0)
    examples = [
        _example(i, prefix=rng.choice(catalog, 5, replace=False).tolist(), target=int(t))
        for i, t in enumerate(rng.integers(200, 1800, size=2000))
    ]
    examples = [e for e in examples if e.target_item_id not in e.prefix_item_ids]
    pools = build_popularity_matched_pools(examples, popularity, catalog, seed=0)

    for example, pool in zip(examples, pools):
        assert len(pool.candidate_item_ids) == 20
        assert len(set(pool.candidate_item_ids)) == 20
        assert pool.candidate_item_ids[pool.target_position] == example.target_item_id
        assert not set(pool.candidate_item_ids) & set(example.prefix_item_ids)
    # Target popularity rank is uniform, so popularity rankers score as random.
    ranks = np.array([p.target_popularity_rank for p in pools])
    assert abs(ranks.mean() - 10.5) < 0.5
    random_rr = sum(1 / k for k in range(1, 21)) / 20
    for sign in (1, -1):
        rr = np.mean([
            tie_aware_metrics({i: sign * popularity[i] for i in p.candidate_item_ids}, p.target_item_id)["RR"]
            for p in pools
        ])
        assert abs(rr - random_rr) < 0.03
    assert pools == build_popularity_matched_pools(examples, popularity, catalog, seed=0)


def test_popularity_midrank_averages_ties():
    popularity = {1: 5, 2: 5, 3: 1, 4: 9}
    assert popularity_midrank([1, 2, 3, 4], 1, popularity) == 2.5


def test_retrieval_pools_record_whether_the_target_was_found():
    # Users who watch 1 also watch 2; item 9 never co-occurs with 1.
    training = _ratings([(u, i, 5, t) for u in range(10) for t, i in enumerate((1, 2, 3))]
                        + [(99, 9, 5, 0), (99, 8, 5, 1)])
    catalog = [1, 2, 3, 8, 9]
    popularity = {1: 10, 2: 10, 3: 10, 8: 1, 9: 1}
    retriever = ItemKNNRetriever(training, catalog, popularity)

    top, rank = retriever.retrieve([1], 2, target_item_id=9)
    assert top == [2, 3] and rank == 4
    assert 1 not in retriever.retrieve([1], 4)[0]

    found, missed = build_retrieval_pools(
        [_example(0, [1], 2), _example(1, [1], 9)], retriever, popularity, pool_size=2
    )
    assert found.target_retrieved and found.retrieved_item_ids == (2, 3)
    assert set(found.candidate_item_ids) == {2, 3}
    assert not missed.target_retrieved and missed.target_retrieval_rank == 4
    # The missed target replaces the lowest-ranked retrieved item.
    assert set(missed.candidate_item_ids) == {2, 9}
