"""Shared HR/NDCG aggregation over valid parsed-ranking target ranks.

Used by both the single-prompt evaluator (scripts/evaluate_baseline.py) and
the ensemble evaluator (scripts/evaluate_ensemble.py) so the metric
definitions never drift between the two.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

DEFAULT_HR_KS: tuple[int, ...] = (1, 5, 10)
DEFAULT_NDCG_KS: tuple[int, ...] = (5, 10)


def ndcg_at_k(rank: int, k: int) -> float:
    """Discounted gain for one target rank, 0 if the target is outside top-k."""

    if rank <= k:
        return 1.0 / math.log2(rank + 1)
    return 0.0


def evaluate_ranks(
    valid_ranks: Sequence[int],
    *,
    hr_ks: Sequence[int] = DEFAULT_HR_KS,
    ndcg_ks: Sequence[int] = DEFAULT_NDCG_KS,
) -> dict[str, float | None]:
    """Compute HR@k and NDCG@k over a list of 1-indexed target ranks.

    Ranks must already be filtered to valid (successfully parsed) trials.
    Returns ``None`` for every metric when ``valid_ranks`` is empty, rather
    than raising or silently reporting a 0.0 that would misrepresent an
    all-failed condition as a working-but-poor one.
    """

    valid_count = len(valid_ranks)
    metrics: dict[str, float | None] = {}
    for k in hr_ks:
        metrics[f"HR@{k}"] = (
            sum(rank <= k for rank in valid_ranks) / valid_count
            if valid_count
            else None
        )
    for k in ndcg_ks:
        metrics[f"NDCG@{k}"] = (
            sum(ndcg_at_k(rank, k) for rank in valid_ranks) / valid_count
            if valid_count
            else None
        )
    return metrics


def extract_valid_ranks(
    records: Sequence[dict[str, object]],
    *,
    rank_field: str = "target_rank",
    success_field: str = "parse_success",
) -> list[int]:
    """Pull valid (successfully parsed) target ranks out of trial records."""

    valid_ranks: list[int] = []
    for record in records:
        if record.get(success_field) is not True:
            continue
        rank = record.get(rank_field)
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            continue
        valid_ranks.append(rank)
    return valid_ranks
