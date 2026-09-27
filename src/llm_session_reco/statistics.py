"""Paired statistical comparisons between two ranking conditions.

Used to compare single-prompt baseline vs. naive-ensemble vs. RWRA on the
same sessions: ``values_a``/``values_b`` must be paired per-session metric
values (e.g. reciprocal rank), aligned by index, restricted beforehand to
sessions valid under both conditions.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import wilcoxon


def paired_bootstrap_ci(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    n_boot: int = 2000,
    seed: int | None = None,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Bootstrap confidence interval for the paired mean difference (a - b).

    Resamples session pairs with replacement ``n_boot`` times and reports the
    percentile interval of the resampled mean difference.
    """

    if len(values_a) != len(values_b):
        raise ValueError("values_a and values_b must have the same length")
    if len(values_a) == 0:
        raise ValueError("values_a/values_b cannot be empty")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    diffs = a - b
    observed_diff = float(diffs.mean())

    rng = np.random.default_rng(seed)
    n = len(diffs)
    sample_indices = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[sample_indices].mean(axis=1)

    alpha = 1 - confidence
    ci_low = float(np.quantile(boot_means, alpha / 2))
    ci_high = float(np.quantile(boot_means, 1 - alpha / 2))

    return {
        "mean_difference": observed_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence": confidence,
        "n_boot": n_boot,
        "n_pairs": n,
    }


def wilcoxon_signed_rank(
    values_a: Sequence[float], values_b: Sequence[float]
) -> dict[str, float | None]:
    """Paired Wilcoxon signed-rank test between two conditions.

    Returns ``statistic=None``/``p_value=None`` when every pair is tied
    (a == b everywhere), since the test is undefined in that degenerate case
    rather than raising the underlying library's error.
    """

    if len(values_a) != len(values_b):
        raise ValueError("values_a and values_b must have the same length")
    if len(values_a) == 0:
        raise ValueError("values_a/values_b cannot be empty")

    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if np.all(a == b):
        return {"statistic": None, "p_value": None, "n_pairs": len(a)}

    statistic, p_value = wilcoxon(a, b)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_pairs": len(a),
    }
