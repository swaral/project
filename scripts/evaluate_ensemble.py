"""Compare single-prompt baseline, naive-ensemble, RWRA, long-tail-weighted
RWRA, and the two non-LLM reference baselines on ensemble trials.

Reads the JSONL produced by scripts/run_ensemble.py and reports HR/NDCG for
four conditions plus paired significance tests between them, so the RWRA
mitigation method's effect -- and the Llama4Rec-style long-tail-aware blend
on top of it -- can be judged against a do-nothing baseline (the first
configured member alone) and a plain self-consistency ensemble. Also reports
metrics stratified by session prefix length (short/medium/long tertiles), to
test whether ensembling's benefit concentrates in long-tail (short-history)
sessions, as Option A's design hypothesizes.

The popularity and random rows (see src/llm_session_reco/baselines.py) are
reference floors, not competitors, and are reported together on purpose:
candidate pools here are built from training-only popularity, so popularity
scores far below random and quoting it alone would overstate the proposed
method's advantage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from llm_session_reco.baselines import (
    expected_random_metrics,
    load_item_popularity,
    popularity_ranking,
    random_ranking,
    rank_of_target,
)
from llm_session_reco.metrics import evaluate_ranks
from llm_session_reco.statistics import paired_bootstrap_ci, wilcoxon_signed_rank

OutcomeFn = Callable[[dict[str, object]], tuple[bool, "int | None"]]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(record)
    return records


def _record_model(record: dict[str, object]) -> str | None:
    model = record.get("model")
    if model is None and isinstance(record.get("client_config"), dict):
        model = record["client_config"].get("model")
    if not isinstance(model, str) or not model.strip():
        return None
    return model.strip()


def _single_model_name(records: list[dict[str, object]]) -> str | None:
    if not records:
        return None
    models = [_record_model(record) for record in records]
    if any(model is None for model in models):
        raise ValueError(
            "Every ensemble trial must record its model in 'model' or "
            "'client_config.model'"
        )
    distinct_models = sorted(set(models))
    if len(distinct_models) != 1:
        raise ValueError(
            "Mixed model results are not allowed in one evaluation file: "
            + ", ".join(distinct_models)
        )
    return distinct_models[0]


def _baseline_outcome(record: dict[str, object]) -> tuple[bool, int | None]:
    """The first configured ensemble member's individual outcome.

    This is the single-prompt-baseline comparison arm: what you would have
    gotten by trusting only the first prompt variant, with no ensembling.
    """

    member_trials = record.get("member_trials") or []
    if not member_trials:
        return False, None
    first = member_trials[0]
    target_rank = first.get("target_rank")
    if (
        first.get("parse_success") is True
        and isinstance(target_rank, int)
        and not isinstance(target_rank, bool)
        and target_rank >= 1
    ):
        return True, int(target_rank)
    return False, None


def _aggregated_outcome(record: dict[str, object], key: str) -> tuple[bool, int | None]:
    aggregated = record.get(key)
    if not isinstance(aggregated, dict):
        return False, None
    target_rank = aggregated.get("target_rank")
    if (
        isinstance(target_rank, int)
        and not isinstance(target_rank, bool)
        and target_rank >= 1
    ):
        return True, int(target_rank)
    return False, None


def _reference_candidates(
    record: dict[str, object],
) -> tuple[list[int], int, str] | None:
    """Pull the pool a non-LLM baseline needs, or None if the record lacks it."""

    candidate_item_ids = record.get("candidate_item_ids")
    target_item_id = record.get("target_item_id")
    if not isinstance(candidate_item_ids, list) or not candidate_item_ids:
        return None
    if isinstance(target_item_id, bool) or not isinstance(target_item_id, int):
        return None
    session_id = str(record.get("session_id") or record.get("trial_id") or "")
    if not session_id:
        return None
    return [int(item_id) for item_id in candidate_item_ids], int(target_item_id), session_id


def _popularity_outcome(
    record: dict[str, object], item_popularity: dict[int, int]
) -> tuple[bool, int | None]:
    pool = _reference_candidates(record)
    if pool is None:
        return False, None
    candidate_item_ids, target_item_id, _ = pool
    ranking = popularity_ranking(candidate_item_ids, item_popularity)
    return True, rank_of_target(ranking, target_item_id)


def _random_outcome(
    record: dict[str, object], random_seed: int
) -> tuple[bool, int | None]:
    pool = _reference_candidates(record)
    if pool is None:
        return False, None
    candidate_item_ids, target_item_id, session_id = pool
    ranking = random_ranking(candidate_item_ids, session_id=session_id, seed=random_seed)
    return True, rank_of_target(ranking, target_item_id)


def _condition_summary(
    records: list[dict[str, object]], outcome_fn: OutcomeFn
) -> dict[str, object]:
    valid_ranks: list[int] = []
    failures = 0
    for record in records:
        success, rank = outcome_fn(record)
        if success and rank is not None:
            valid_ranks.append(rank)
        else:
            failures += 1
    return {
        "total_trials": len(records),
        "valid_trials": len(valid_ranks),
        "failed_trials": failures,
        "metrics": evaluate_ranks(valid_ranks),
    }


def _paired_reciprocal_ranks(
    records: list[dict[str, object]], outcome_fn_a: OutcomeFn, outcome_fn_b: OutcomeFn
) -> tuple[list[float], list[float]]:
    """Reciprocal ranks for sessions where both conditions produced a valid rank."""

    values_a: list[float] = []
    values_b: list[float] = []
    for record in records:
        success_a, rank_a = outcome_fn_a(record)
        success_b, rank_b = outcome_fn_b(record)
        if not (success_a and success_b) or rank_a is None or rank_b is None:
            continue
        values_a.append(1.0 / rank_a)
        values_b.append(1.0 / rank_b)
    return values_a, values_b


def _drift_summary(records: list[dict[str, object]]) -> dict[str, object]:
    tau_values = [
        record["drift"]["mean_kendall_tau"]
        for record in records
        if isinstance(record.get("drift"), dict)
        and record["drift"].get("mean_kendall_tau") is not None
    ]
    jaccard5_values = [
        record["drift"]["mean_jaccard_at_5"]
        for record in records
        if isinstance(record.get("drift"), dict)
        and record["drift"].get("mean_jaccard_at_5") is not None
    ]
    return {
        "mean_kendall_tau": sum(tau_values) / len(tau_values) if tau_values else None,
        "mean_jaccard_at_5": (
            sum(jaccard5_values) / len(jaccard5_values) if jaccard5_values else None
        ),
        "sessions_with_drift_score": len(tau_values),
    }


def _prefix_length(record: dict[str, object]) -> int | None:
    prefix_item_ids = record.get("prefix_item_ids")
    if not isinstance(prefix_item_ids, list):
        return None
    return len(prefix_item_ids)


def _bucket_by_prefix_length(
    records: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """Split records into short/medium/long tertiles by session prefix length.

    Tertile boundaries are computed from the sorted prefix lengths present in
    ``records`` itself, so bucket sizes stay roughly equal regardless of the
    dataset's actual length distribution.
    """

    lengths = sorted(
        length for record in records if (length := _prefix_length(record)) is not None
    )
    buckets: dict[str, list[dict[str, object]]] = {
        "short_history": [],
        "medium_history": [],
        "long_history": [],
    }
    if not lengths:
        return buckets

    n = len(lengths)
    low_cutoff = lengths[max(0, n // 3 - 1)]
    high_cutoff = lengths[max(0, (2 * n) // 3 - 1)]

    for record in records:
        length = _prefix_length(record)
        if length is None:
            continue
        if length <= low_cutoff:
            buckets["short_history"].append(record)
        elif length <= high_cutoff:
            buckets["medium_history"].append(record)
        else:
            buckets["long_history"].append(record)
    return buckets


def evaluate_ensemble_records(
    records: list[dict[str, object]],
    *,
    n_boot: int = 2000,
    seed: int | None = 0,
    item_popularity: dict[int, int] | None = None,
    random_seed: int = 0,
    include_reference_baselines: bool = True,
) -> dict[str, object]:
    model_name = _single_model_name(records)
    outcome_fns: dict[str, OutcomeFn] = {
        "single_prompt_baseline": _baseline_outcome,
        "naive_mean_ensemble": lambda r: _aggregated_outcome(r, "naive_mean"),
        "reliability_weighted": lambda r: _aggregated_outcome(r, "reliability_weighted"),
        "long_tail_weighted": lambda r: _aggregated_outcome(r, "long_tail_weighted"),
    }

    # Reference floors are added only when the trial records actually carry the
    # candidate pool they need, so a records file without pools reports four
    # conditions rather than two silently-empty extra rows.
    pool_sizes = [
        len(pool[0])
        for record in records
        if (pool := _reference_candidates(record)) is not None
    ] if include_reference_baselines else []
    pools_available = bool(pool_sizes)
    reference_baselines: dict[str, object] = {
        "computed": include_reference_baselines,
        "random_analytic_metrics": (
            expected_random_metrics(pool_sizes) if pool_sizes else None
        ),
        "popularity_available": bool(pools_available and item_popularity is not None),
        "random_available": bool(pools_available),
        "random_seed": random_seed,
        "popularity_items": len(item_popularity) if item_popularity else 0,
        "note": (
            "Reference baselines were skipped for this model; reuse the values "
            "from the first model's metrics file."
            if not include_reference_baselines
            else
            "Candidate pools are built from training-only popularity, so the "
            "popularity row is adversarial by construction and falls below the "
            "random floor; read the two together."
        ),
    }
    if include_reference_baselines and pools_available:
        outcome_fns["random_baseline"] = lambda r: _random_outcome(r, random_seed)
        if item_popularity is not None:
            outcome_fns["popularity_baseline"] = lambda r: _popularity_outcome(
                r, item_popularity
            )
    conditions = {
        name: _condition_summary(records, fn) for name, fn in outcome_fns.items()
    }

    comparisons: dict[str, object] = {}
    for name_a, name_b in (
        ("reliability_weighted", "single_prompt_baseline"),
        ("reliability_weighted", "naive_mean_ensemble"),
        ("naive_mean_ensemble", "single_prompt_baseline"),
        ("long_tail_weighted", "single_prompt_baseline"),
        ("long_tail_weighted", "reliability_weighted"),
        ("reliability_weighted", "random_baseline"),
        ("reliability_weighted", "popularity_baseline"),
    ):
        if name_a not in outcome_fns or name_b not in outcome_fns:
            continue
        values_a, values_b = _paired_reciprocal_ranks(
            records, outcome_fns[name_a], outcome_fns[name_b]
        )
        comparison_key = f"{name_a}_vs_{name_b}"
        if len(values_a) < 2:
            comparisons[comparison_key] = {
                "n_pairs": len(values_a),
                "metric": "reciprocal_rank",
                "bootstrap_ci": None,
                "wilcoxon": None,
            }
            continue
        comparisons[comparison_key] = {
            "n_pairs": len(values_a),
            "metric": "reciprocal_rank",
            "bootstrap_ci": paired_bootstrap_ci(
                values_a, values_b, n_boot=n_boot, seed=seed
            ),
            "wilcoxon": wilcoxon_signed_rank(values_a, values_b),
        }

    stratified_by_history_length = {
        bucket_name: {
            "session_count": len(bucket_records),
            "conditions": {
                name: _condition_summary(bucket_records, fn)
                for name, fn in outcome_fns.items()
            },
        }
        for bucket_name, bucket_records in _bucket_by_prefix_length(records).items()
    }

    return {
        "schema_version": "ensemble_evaluation_v3",
        "model": model_name,
        "total_trials": len(records),
        "reference_baselines": reference_baselines,
        "conditions": conditions,
        "comparisons": comparisons,
        "stratified_by_history_length": stratified_by_history_length,
        "drift": _drift_summary(records),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/ml1m_3b_ensemble_trials.jsonl"),
        help="trials JSONL from run_ensemble.py (one model per file)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="metrics JSON path; defaults beside the input with an _metrics suffix",
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-reference-baselines",
        action="store_true",
        help="skip model-independent random/popularity baselines when reusing the "
        "values computed for another model",
    )
    parser.add_argument(
        "--popularity-file",
        type=Path,
        default=None,
        help=(
            "Training-only item popularity JSON from "
            "scripts/build_item_popularity.py. Omit to skip the popularity row."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Seed for the per-session random reference ranking.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _read_jsonl(args.input)
    item_popularity = (
        load_item_popularity(args.popularity_file) if args.popularity_file else None
    )
    summary = evaluate_ensemble_records(
        records,
        n_boot=args.n_boot,
        seed=args.seed,
        item_popularity=item_popularity,
        random_seed=args.random_seed,
        include_reference_baselines=not args.skip_reference_baselines,
    )
    summary["input"] = str(args.input)
    summary["popularity_file"] = (
        str(args.popularity_file) if args.popularity_file else None
    )

    output = args.output
    if output is None:
        stem = args.input.stem
        if stem.endswith("_trials"):
            stem = f"{stem[:-len('_trials')]}_metrics"
        else:
            stem = f"{stem}_metrics"
        output = args.input.with_name(f"{stem}.json")
    summary["output"] = str(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
