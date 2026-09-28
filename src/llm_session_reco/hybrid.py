"""Hybrid scoring: does the LLM add signal on top of non-LLM rankers?

Every component is z-scored within the session, then combined linearly.
Weights are chosen on validation sessions only, by exhaustive search over a
small grid that always includes zero, so a component that does not help on
validation is switched off rather than forced in.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence

import numpy as np

from .debiasing import zscore
from .metrics import tie_aware_metrics

Scores = Mapping[int, float]

DEFAULT_WEIGHT_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
DEFAULT_LLM_GRID: tuple[float, ...] = tuple(round(0.1 * k, 1) for k in range(21))


def blend(components: Sequence[Scores], weights: Sequence[float]) -> dict[int, float]:
    """Weighted sum of z-scored components (all keyed by the same items)."""

    items = components[0].keys()
    standardized = [zscore(c) for c in components]
    return {item: sum(w * z[item] for w, z in zip(weights, standardized)) for item in items}


def mean_rr(sessions: Sequence[tuple[Sequence[Scores], int]], weights: Sequence[float]) -> float:
    """Mean tie-aware reciprocal rank of a blend over (components, target) pairs."""

    return float(np.mean([tie_aware_metrics(blend(c, weights), t)["RR"] for c, t in sessions]))


def fit_weights(
    sessions: Sequence[tuple[Sequence[Scores], int]],
    grid: Sequence[float] = DEFAULT_WEIGHT_GRID,
) -> tuple[tuple[float, ...], float]:
    """Grid-search component weights maximizing validation MRR.

    All-zero weight vectors are skipped. Ties prefer fewer active components,
    then smaller weights, so the choice is deterministic.
    """

    n = len(sessions[0][0])
    best: tuple[float, tuple[float, ...]] | None = None
    best_key = None
    for weights in itertools.product(grid, repeat=n):
        if not any(weights):
            continue
        score = mean_rr(sessions, weights)
        key = (-score, sum(w > 0 for w in weights), sum(weights), weights)
        if best_key is None or key < best_key:
            best_key, best = key, (score, weights)
    return best[1], best[0]


def fit_llm_weight(
    sessions: Sequence[tuple[Scores, Scores, int]],
    grid: Sequence[float] = DEFAULT_LLM_GRID,
) -> tuple[float, float]:
    """Weight on the LLM added to a fixed base, maximizing validation MRR.

    ``sessions`` are (base scores, LLM scores, target). Weight 0 means the LLM
    is not used; ties prefer the smaller weight.
    """

    best_weight, best_score = 0.0, -1.0
    for weight in grid:
        score = mean_rr([((base, llm), target) for base, llm, target in sessions], (1.0, weight))
        if score > best_score + 1e-12:
            best_weight, best_score = weight, score
    return best_weight, best_score
