"""Tests for the non-LLM reference rankers."""

from __future__ import annotations

import pandas as pd

from llm_session_reco.benchmark import ItemKNNRetriever
from llm_session_reco.reference_rankers import build_reference_rankers


def test_newest_first_ranks_by_first_training_timestamp():
    training = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "item_id": [10, 11, 10, 12],
            "timestamp": [100, 300, 200, 250],
            "rating": [5.0, 4.0, 3.0, 5.0],
        }
    )
    popularity = {10: 2, 11: 1, 12: 1}
    titles = {10: "A", 11: "B", 12: "C", 13: "D"}
    retriever = ItemKNNRetriever(training, titles, popularity)
    rankers = build_reference_rankers(training, popularity, titles, {}, retriever)

    scores = rankers["newest_first"]([10], [10, 11, 12, 13])
    # First seen: 10 at 100, 12 at 250, 11 at 300; 13 never seen counts as newest.
    assert sorted(scores, key=scores.get, reverse=True) == [13, 11, 12, 10]
