"""Non-LLM reference baselines: popularity and random.

These are *reference floors*, not components of the proposed method. They
exist so the ensemble results table can answer the question "is this task
actually hard, or would a trivial ranker do just as well?" without adding any
trained model to the pipeline -- neither baseline issues a single LLM call.

Read the two together. Candidate pools in this project are built from
training-only popularity (see :func:`llm_session_reco.session_dataset.build_candidate_pools`),
so the 19 negatives in every pool *are* the popularity ranking. That makes the
pools popularity-adversarial by construction and drives the popularity
baseline far below the random floor. Reporting popularity alone would
overstate the proposed method's advantage; random is the honest floor.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pandas as pd

from .metrics import DEFAULT_HR_KS, DEFAULT_NDCG_KS, ndcg_at_k

#: Popularity assigned to an item that never appears in the training ratings.
#: Such items sort below every observed item, then by ascending item ID.
UNSEEN_ITEM_COUNT = 0


def build_item_popularity(training_ratings: pd.DataFrame) -> dict[int, int]:
    """Count training interactions per item.

    Uses the same training-only ratings frame as candidate-pool construction,
    so the baseline never sees the held-out target interaction.
    """

    if "item_id" not in training_ratings.columns:
        raise ValueError("training_ratings is missing required column: item_id")

    counts = training_ratings.groupby("item_id").size()
    return {int(item_id): int(count) for item_id, count in counts.items()}


def write_item_popularity(
    item_popularity: Mapping[int, int], path: str | Path
) -> Path:
    """Persist a popularity table as JSON with string keys."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(item_id): int(count) for item_id, count in item_popularity.items()}
    destination.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return destination


def load_item_popularity(path: str | Path) -> dict[int, int]:
    """Load a popularity table written by :func:`write_item_popularity`."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return {int(item_id): int(count) for item_id, count in raw.items()}


def _validated_candidates(candidate_item_ids: Iterable[int]) -> list[int]:
    candidates = [int(item_id) for item_id in candidate_item_ids]
    if not candidates:
        raise ValueError("candidate_item_ids cannot be empty")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate_item_ids contains duplicate items")
    return candidates


def popularity_ranking(
    candidate_item_ids: Iterable[int], item_popularity: Mapping[int, int]
) -> tuple[int, ...]:
    """Rank candidates by descending training popularity.

    Ties break by ascending item ID, matching
    :func:`llm_session_reco.session_dataset._item_popularity`, so this baseline
    orders items exactly the way the candidate pools were built.
    """

    candidates = _validated_candidates(candidate_item_ids)
    return tuple(
        sorted(
            candidates,
            key=lambda item_id: (
                -int(item_popularity.get(item_id, UNSEEN_ITEM_COUNT)),
                item_id,
            ),
        )
    )


def random_ranking(
    candidate_item_ids: Iterable[int], *, session_id: str, seed: int = 0
) -> tuple[int, ...]:
    """Shuffle candidates deterministically per session.

    The RNG is seeded from ``seed`` and ``session_id`` together, so a session's
    ranking is reproducible and independent of how many sessions were processed
    before it -- re-running on a subset gives the same per-session ordering.
    """

    candidates = _validated_candidates(candidate_item_ids)
    rng = random.Random(f"{seed}:{session_id}")
    rng.shuffle(candidates)
    return tuple(candidates)


def rank_of_target(ranked_item_ids: Sequence[int], target_item_id: int) -> int:
    """1-indexed position of the target in a ranking.

    Raises rather than returning a sentinel when the target is missing: a pool
    without its target is a data defect, not a low-scoring trial.
    """

    target = int(target_item_id)
    for position, item_id in enumerate(ranked_item_ids, start=1):
        if int(item_id) == target:
            return position
    raise ValueError(f"target item {target} is not present in the ranking")


def expected_random_metrics(
    pool_sizes: Iterable[int],
    *,
    hr_ks: Sequence[int] = DEFAULT_HR_KS,
    ndcg_ks: Sequence[int] = DEFAULT_NDCG_KS,
) -> dict[str, float]:
    """Closed-form expectation of a uniformly random ranker over these pools.

    A shuffled baseline is a Monte Carlo estimate of this quantity, so one
    unlucky seed can sit a couple of standard errors off (e.g. HR@1 = 0.042
    instead of 0.050 on 6,040 sessions). Reporting the analytic value next to
    the shuffled one keeps the floor exact without choosing a flattering seed.

    For a pool of size ``N``: ``P(rank <= k) = min(k, N) / N``, and the
    expected NDCG@k is the mean discounted gain over the first ``min(k, N)``
    positions. Values are averaged over pools so unequal pool sizes are handled.
    """

    sizes = [int(size) for size in pool_sizes]
    if not sizes:
        raise ValueError("pool_sizes cannot be empty")
    if any(size < 1 for size in sizes):
        raise ValueError("pool sizes must be at least 1")

    metrics: dict[str, float] = {}
    for k in hr_ks:
        metrics[f"HR@{k}"] = sum(min(k, size) / size for size in sizes) / len(sizes)
    for k in ndcg_ks:
        metrics[f"NDCG@{k}"] = (
            sum(
                sum(ndcg_at_k(rank, k) for rank in range(1, min(k, size) + 1)) / size
                for size in sizes
            )
            / len(sizes)
        )
    return metrics
