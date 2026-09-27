"""Ranking-agreement metrics used to quantify prompt/context drift.

These functions operate on candidate *positions* (1..N, as presented to the
LLM) or *item IDs*, not on arbitrary values, so every ranking passed in must
be a permutation of the same underlying set.
"""

from __future__ import annotations

from collections.abc import Sequence

from scipy.stats import kendalltau


def kendall_tau(rank_a: Sequence[int], rank_b: Sequence[int]) -> float:
    """Return Kendall's tau-b between two rankings of the same items.

    ``rank_a``/``rank_b`` are sequences of item identifiers (or candidate
    positions) in ranked order, e.g. ``[3, 1, 2]`` means item 3 was ranked
    first. Both sequences must contain the same set of items. Returns 1.0 for
    identical order, -1.0 for fully reversed order, and 0.0 when either
    ranking has fewer than two items (correlation is undefined).
    """

    if set(rank_a) != set(rank_b):
        raise ValueError("rank_a and rank_b must rank the same set of items")
    if len(rank_a) != len(rank_b):
        raise ValueError("rank_a and rank_b must have the same length")
    if len(rank_a) < 2:
        return 0.0

    position_in_b = {item: index for index, item in enumerate(rank_b)}
    b_positions_in_a_order = [position_in_b[item] for item in rank_a]

    tau, _p_value = kendalltau(range(len(rank_a)), b_positions_in_a_order)
    if tau != tau:  # NaN guard (e.g. constant input after ties)
        return 0.0
    return float(tau)


def jaccard_at_k(ranked_a: Sequence[int], ranked_b: Sequence[int], k: int) -> float:
    """Return the Jaccard overlap of the top-``k`` items of two rankings."""

    if k < 1:
        raise ValueError("k must be at least 1")

    top_a = set(ranked_a[:k])
    top_b = set(ranked_b[:k])
    union = top_a | top_b
    if not union:
        return 0.0
    return len(top_a & top_b) / len(union)


def pairwise_agreement(
    member_rankings: Sequence[Sequence[int]],
    *,
    jaccard_k: Sequence[int] = (5, 10),
) -> dict[str, float | None]:
    """Summarize agreement across an ensemble's per-member rankings.

    Returns the mean pairwise Kendall's tau and the mean pairwise Jaccard@k
    (for every ``k`` in ``jaccard_k``) across all member pairs. This is the
    per-session drift score: low agreement means the ensemble members
    disagree about the ranking for this session. Returns ``None`` values when
    fewer than two member rankings are available (agreement is undefined).
    """

    n = len(member_rankings)
    result: dict[str, float | None] = {"mean_kendall_tau": None}
    for k in jaccard_k:
        result[f"mean_jaccard_at_{k}"] = None
    if n < 2:
        return result

    tau_values: list[float] = []
    jaccard_values: dict[int, list[float]] = {k: [] for k in jaccard_k}
    for i in range(n):
        for j in range(i + 1, n):
            tau_values.append(kendall_tau(member_rankings[i], member_rankings[j]))
            for k in jaccard_k:
                jaccard_values[k].append(
                    jaccard_at_k(member_rankings[i], member_rankings[j], k)
                )

    result["mean_kendall_tau"] = sum(tau_values) / len(tau_values)
    for k in jaccard_k:
        values = jaccard_values[k]
        result[f"mean_jaccard_at_{k}"] = sum(values) / len(values) if values else None
    return result
