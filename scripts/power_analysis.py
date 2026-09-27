"""Decide from a pilot whether a full ensemble run is worth doing.

Reads a pilot trials JSONL from scripts/run_ensemble.py, takes the paired
per-session reciprocal-rank differences for each headline comparison (the same
pairing rule evaluate_ensemble.py uses), and reports:

- the observed effect, its bootstrap CI, and the Wilcoxon p-value on the pilot;
- the standardized paired effect size d_z = mean(diff) / sd(diff);
- the achieved power at the planned full-run size, and the number of sessions
  needed for the target power.

Sample sizes use the normal approximation for a two-sided paired test, then
divide by a Wilcoxon asymptotic relative efficiency of 0.864 -- the worst case
over all continuous distributions -- so the estimate stays conservative for
the skewed, tie-heavy reciprocal-rank differences. A pilot effect is itself
noisy, so the verdict also reports the size needed at the pilot CI's
lower bound.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

# `python scripts/power_analysis.py` puts scripts/ (not the project root) on
# sys.path, so the sibling-script import below needs the root added explicitly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from llm_session_reco.statistics import paired_bootstrap_ci, wilcoxon_signed_rank  # noqa: E402
from scripts.evaluate_ensemble import (  # noqa: E402
    _aggregated_outcome,
    _baseline_outcome,
    _paired_reciprocal_ranks,
    _read_jsonl,
    _single_model_name,
)

#: Minimum Wilcoxon-vs-t asymptotic relative efficiency (Hodges-Lehmann bound).
WILCOXON_MIN_ARE = 0.864

COMPARISONS = {
    "reliability_weighted_vs_single_prompt_baseline": (
        lambda r: _aggregated_outcome(r, "reliability_weighted"),
        _baseline_outcome,
    ),
    "reliability_weighted_vs_naive_mean_ensemble": (
        lambda r: _aggregated_outcome(r, "reliability_weighted"),
        lambda r: _aggregated_outcome(r, "naive_mean"),
    ),
    "naive_mean_ensemble_vs_single_prompt_baseline": (
        lambda r: _aggregated_outcome(r, "naive_mean"),
        _baseline_outcome,
    ),
    "long_tail_weighted_vs_reliability_weighted": (
        lambda r: _aggregated_outcome(r, "long_tail_weighted"),
        lambda r: _aggregated_outcome(r, "reliability_weighted"),
    ),
}


def required_sessions(effect_size: float, *, alpha: float, power: float) -> int | None:
    """Sessions a two-sided paired Wilcoxon test needs to detect ``effect_size``."""

    if effect_size == 0 or not math.isfinite(effect_size):
        return None
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    n_normal = ((z_alpha + z_power) / abs(effect_size)) ** 2
    return math.ceil(n_normal / WILCOXON_MIN_ARE)


def achieved_power(effect_size: float, n: int, *, alpha: float) -> float:
    """Power of a two-sided paired Wilcoxon test with ``n`` sessions."""

    if n < 1:
        return 0.0
    z_alpha = norm.ppf(1 - alpha / 2)
    shift = abs(effect_size) * math.sqrt(n * WILCOXON_MIN_ARE)
    return float(norm.cdf(shift - z_alpha) + norm.cdf(-shift - z_alpha))


def analyse(
    records: list[dict[str, object]],
    *,
    planned_n: int,
    alpha: float = 0.05,
    power: float = 0.8,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, object]:
    results: dict[str, object] = {}
    for name, (fn_a, fn_b) in COMPARISONS.items():
        values_a, values_b = _paired_reciprocal_ranks(records, fn_a, fn_b)
        n = len(values_a)
        if n < 2:
            results[name] = {"n_pairs": n, "verdict": "too few paired sessions"}
            continue
        diffs = np.asarray(values_a) - np.asarray(values_b)
        sd = float(diffs.std(ddof=1))
        mean = float(diffs.mean())
        ci = paired_bootstrap_ci(values_a, values_b, n_boot=n_boot, seed=seed)
        d_z = mean / sd if sd > 0 else 0.0
        # Smallest plausible effect in the pilot's favoured direction.
        conservative_mean = ci["ci_low"] if mean > 0 else ci["ci_high"]
        conservative_d = (
            conservative_mean / sd
            if sd > 0 and conservative_mean * mean > 0
            else 0.0
        )
        needed = required_sessions(d_z, alpha=alpha, power=power)
        needed_conservative = required_sessions(conservative_d, alpha=alpha, power=power)
        planned_power = achieved_power(d_z, planned_n, alpha=alpha)
        if needed is None:
            verdict = "no effect in pilot; a full run would only confirm a null result"
        elif needed <= planned_n:
            verdict = (
                f"worth it: {planned_n} sessions give {planned_power:.0%} power "
                f"(need {needed})"
            )
        else:
            verdict = (
                f"underpowered: {planned_n} sessions give {planned_power:.0%} power; "
                f"need {needed}"
            )
        results[name] = {
            "n_pairs": n,
            "mean_rr_difference": mean,
            "sd_rr_difference": sd,
            "effect_size_dz": d_z,
            "sessions_improved": int((diffs > 0).sum()),
            "sessions_worsened": int((diffs < 0).sum()),
            "sessions_tied": int((diffs == 0).sum()),
            "pilot_bootstrap_ci": ci,
            "pilot_wilcoxon": wilcoxon_signed_rank(values_a, values_b),
            "sessions_needed": needed,
            "sessions_needed_at_ci_bound": needed_conservative,
            "power_at_planned_n": planned_power,
            "verdict": verdict,
        }
    return {
        "model": _single_model_name(records),
        "pilot_sessions": len(records),
        "planned_sessions": planned_n,
        "alpha": alpha,
        "target_power": power,
        "wilcoxon_are_assumed": WILCOXON_MIN_ARE,
        "comparisons": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="pilot trials JSONL")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--planned-sessions", type=int, default=3000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.8)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyse(
        _read_jsonl(args.input),
        planned_n=args.planned_sessions,
        alpha=args.alpha,
        power=args.power,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    output = args.output or args.input.with_name(
        args.input.stem.removesuffix("_trials") + "_power.json"
    )
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
