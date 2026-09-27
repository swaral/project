"""Ensembling and reliability-weighted aggregation of multiple prompt-variant scores.

This is the core mitigation method: instead of trusting a single prompt
variant's ranking, score every candidate under several prompt/context
variants (an ensemble) and combine them. Three aggregation strategies are
implemented so they can be compared directly:

- ``aggregate_naive_mean``: a plain self-consistency baseline (unweighted
  average).
- ``compute_reliability_weights`` + ``aggregate_reliability_weighted``
  (Reliability-Weighted Rank Aggregation, RWRA): members are weighted, per
  trial, by how much they agree with the rest of the ensemble, so a variant
  that produced an outlier ranking for this specific session is downweighted
  without needing any labeled calibration data.
- ``long_tail_weight`` + ``aggregate_long_tail_weighted``: a session-level
  blend of the RWRA ensemble with the single baseline prompt, adapted from
  Luo et al.'s Llama4Rec (IEEE JSTSP 2026) long-tail-aware adaptive
  aggregation. Llama4Rec blends an LLM and a conventional recommender,
  weighted by how sparse a *user's* interaction history is; here there is
  only one model family (the LLM), so the same mechanism is repurposed to
  blend the *ensemble* with the *single prompt*, weighted by how short this
  *session's* prefix is. A short prefix gives every individual prompt little
  to reason over, so the ensemble consensus is trusted more there; a long,
  information-rich prefix may already be handled reliably by one prompt.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .stability import kendall_tau, pairwise_agreement


@dataclass(frozen=True)
class EnsembleMember:
    """One ensemble member's rendered-prompt identity and parsed outcome."""

    variant_id: str
    context_variant_id: str
    parse_success: bool
    position_scores: tuple[float, ...] = ()


@dataclass(frozen=True)
class AggregatedRanking:
    """One aggregation strategy's result for a single ensemble trial."""

    strategy: str
    position_scores: tuple[float, ...]
    ranked_positions: tuple[int, ...]
    ranked_item_ids: tuple[int, ...]
    target_rank: int | None


@dataclass(frozen=True)
class EnsembleResult:
    """Complete outcome of combining an ensemble of prompt-variant members."""

    valid_member_count: int
    member_weights: tuple[float, ...]
    naive: AggregatedRanking | None
    reliability_weighted: AggregatedRanking | None
    long_tail_weighted: AggregatedRanking | None
    long_tail_beta: float | None
    drift: dict[str, float | None]
    failure_reason: str | None


def aggregate_naive_mean(
    member_position_scores: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    """Elementwise mean of position scores across members."""

    if not member_position_scores:
        raise ValueError("member_position_scores cannot be empty")
    candidate_count = len(member_position_scores[0])
    if any(len(scores) != candidate_count for scores in member_position_scores):
        raise ValueError(
            "all members must score the same number of candidate positions"
        )

    return tuple(
        sum(scores[i] for scores in member_position_scores) / len(member_position_scores)
        for i in range(candidate_count)
    )


def compute_reliability_weights(
    member_rankings: Sequence[Sequence[int]],
) -> tuple[float, ...]:
    """Per-trial weight for each member based on agreement with the ensemble.

    Each member's weight is its mean Kendall's tau against every other
    member, shifted from [-1, 1] into [0, 2] so a member that disagrees with
    the consensus never receives a negative weight, then normalized to sum to
    1. With a single member, the weight is trivially 1.0.
    """

    n = len(member_rankings)
    if n == 0:
        raise ValueError("member_rankings cannot be empty")
    if n == 1:
        return (1.0,)

    raw_scores: list[float] = []
    for i in range(n):
        others_tau = [
            kendall_tau(member_rankings[i], member_rankings[j])
            for j in range(n)
            if j != i
        ]
        mean_tau = sum(others_tau) / len(others_tau)
        raw_scores.append(mean_tau + 1.0)

    total = sum(raw_scores)
    if total <= 0:
        return tuple(1.0 / n for _ in range(n))
    return tuple(score / total for score in raw_scores)


def aggregate_reliability_weighted(
    member_position_scores: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> tuple[float, ...]:
    """Weighted elementwise combination of position scores."""

    if not member_position_scores:
        raise ValueError("member_position_scores cannot be empty")
    if len(member_position_scores) != len(weights):
        raise ValueError(
            "member_position_scores and weights must have the same length"
        )
    candidate_count = len(member_position_scores[0])
    if any(len(scores) != candidate_count for scores in member_position_scores):
        raise ValueError(
            "all members must score the same number of candidate positions"
        )

    return tuple(
        sum(scores[i] * weight for scores, weight in zip(member_position_scores, weights))
        for i in range(candidate_count)
    )


def long_tail_coefficient(prefix_length: int) -> float:
    """log(N+1) coefficient for a session's prefix length (Llama4Rec Eq. 4).

    Higher values mean a longer, richer session history.
    """

    if prefix_length < 0:
        raise ValueError("prefix_length cannot be negative")
    return math.log(prefix_length + 1)


def long_tail_weight(
    prefix_length: int,
    *,
    min_prefix_length: int,
    max_prefix_length: int,
    beta1: float = 0.7,
    beta2: float = 0.3,
) -> float:
    """Session-level trust in the RWRA ensemble vs. the single baseline prompt.

    Adapted from Llama4Rec's adaptive-aggregation weight (Eq. 6): a session
    with a short prefix (relative to ``min_prefix_length``/
    ``max_prefix_length`` across the dataset) gets a weight close to
    ``beta1``; a session with a long prefix gets a weight approaching
    ``beta2``. ``beta2`` acts as a floor so the single prompt is never fully
    discarded even for very long sessions.
    """

    if max_prefix_length <= min_prefix_length:
        raise ValueError("max_prefix_length must be greater than min_prefix_length")
    if not 0 <= beta2 <= beta1 <= 1:
        raise ValueError("beta2 and beta1 must satisfy 0 <= beta2 <= beta1 <= 1")

    l_min = long_tail_coefficient(min_prefix_length)
    l_max = long_tail_coefficient(max_prefix_length)
    l_s = long_tail_coefficient(prefix_length)
    scaled = (l_max - l_s) / (l_max - l_min)
    return max(scaled, beta2) * beta1


def aggregate_long_tail_weighted(
    rwra_scores: Sequence[float],
    baseline_scores: Sequence[float],
    weight: float,
) -> tuple[float, ...]:
    """Blend RWRA-ensembled scores with the single baseline prompt's scores.

    ``weight`` (the session's long-tail-aware ensemble trust, from
    :func:`long_tail_weight`) is the trust placed in the RWRA ensemble;
    ``1 - weight`` is the trust placed in the single baseline prompt.
    """

    if len(rwra_scores) != len(baseline_scores):
        raise ValueError(
            "rwra_scores and baseline_scores must have the same length"
        )
    if not 0 <= weight <= 1:
        raise ValueError("weight must be between 0 and 1")

    return tuple(
        weight * r + (1 - weight) * b for r, b in zip(rwra_scores, baseline_scores)
    )


def _rank_from_scores(
    position_scores: Sequence[float], candidate_item_ids: Sequence[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Deterministically rank candidate positions by descending score.

    Mirrors ``parser.parse_ranking``'s tie-break rule (descending score, then
    ascending candidate position) so ensembled rankings stay comparable to
    single-prompt rankings.
    """

    ranked_positions = tuple(
        sorted(
            range(1, len(candidate_item_ids) + 1),
            key=lambda position: (-position_scores[position - 1], position),
        )
    )
    ranked_item_ids = tuple(
        candidate_item_ids[position - 1] for position in ranked_positions
    )
    return ranked_positions, ranked_item_ids


def _build_aggregated_ranking(
    strategy: str,
    position_scores: tuple[float, ...],
    candidate_item_ids: Sequence[int],
    target_item_id: int,
) -> AggregatedRanking:
    ranked_positions, ranked_item_ids = _rank_from_scores(
        position_scores, candidate_item_ids
    )
    target_rank = ranked_item_ids.index(target_item_id) + 1
    return AggregatedRanking(
        strategy=strategy,
        position_scores=position_scores,
        ranked_positions=ranked_positions,
        ranked_item_ids=ranked_item_ids,
        target_rank=target_rank,
    )


def combine_ensemble(
    members: Sequence[EnsembleMember],
    candidate_item_ids: Sequence[int],
    target_item_id: int,
    *,
    min_valid_members: int = 2,
    prefix_length: int | None = None,
    min_prefix_length: int | None = None,
    max_prefix_length: int | None = None,
    beta1: float = 0.7,
    beta2: float = 0.3,
) -> EnsembleResult:
    """Combine an ensemble of prompt-variant score members for one session.

    Members that failed to parse are excluded, never silently repaired. If
    fewer than ``min_valid_members`` members parsed successfully, all
    aggregation strategies are recorded as a structured failure rather than
    computed from a degenerate ensemble.

    ``members[0]`` is treated as the single baseline prompt for the
    long-tail-weighted blend. When ``prefix_length``/``min_prefix_length``/
    ``max_prefix_length`` are omitted, or ``members[0]`` itself failed to
    parse, ``long_tail_weighted`` is left ``None`` rather than guessed at.
    """

    if target_item_id not in candidate_item_ids:
        raise ValueError("target_item_id is not present in candidate_item_ids")

    valid_members = [member for member in members if member.parse_success]
    if len(valid_members) < min_valid_members:
        return EnsembleResult(
            valid_member_count=len(valid_members),
            member_weights=(),
            naive=None,
            reliability_weighted=None,
            long_tail_weighted=None,
            long_tail_beta=None,
            drift={
                "mean_kendall_tau": None,
                "mean_jaccard_at_5": None,
                "mean_jaccard_at_10": None,
            },
            failure_reason=(
                f"only {len(valid_members)} of {len(members)} members parsed "
                f"successfully; need at least {min_valid_members}"
            ),
        )

    member_position_scores = [member.position_scores for member in valid_members]
    member_rankings = [
        _rank_from_scores(scores, candidate_item_ids)[1]
        for scores in member_position_scores
    ]

    naive_scores = aggregate_naive_mean(member_position_scores)
    weights = compute_reliability_weights(member_rankings)
    rwra_scores = aggregate_reliability_weighted(member_position_scores, weights)
    drift = pairwise_agreement(member_rankings)

    long_tail_result: AggregatedRanking | None = None
    long_tail_beta: float | None = None
    baseline_member = members[0] if members else None
    if (
        baseline_member is not None
        and baseline_member.parse_success
        and prefix_length is not None
        and min_prefix_length is not None
        and max_prefix_length is not None
    ):
        long_tail_beta = long_tail_weight(
            prefix_length,
            min_prefix_length=min_prefix_length,
            max_prefix_length=max_prefix_length,
            beta1=beta1,
            beta2=beta2,
        )
        blended_scores = aggregate_long_tail_weighted(
            rwra_scores, baseline_member.position_scores, long_tail_beta
        )
        long_tail_result = _build_aggregated_ranking(
            "long_tail_weighted", blended_scores, candidate_item_ids, target_item_id
        )

    return EnsembleResult(
        valid_member_count=len(valid_members),
        member_weights=weights,
        naive=_build_aggregated_ranking(
            "naive_mean", naive_scores, candidate_item_ids, target_item_id
        ),
        reliability_weighted=_build_aggregated_ranking(
            "reliability_weighted", rwra_scores, candidate_item_ids, target_item_id
        ),
        long_tail_weighted=long_tail_result,
        long_tail_beta=long_tail_beta,
        drift=drift,
        failure_reason=None,
    )
