"""Paired comparison of Qwen2.5-3B and Qwen2.5-7B ensemble trials."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

# `python scripts/compare_models.py` puts scripts/ (not the project root) on
# sys.path, so the sibling-script import below needs the root added explicitly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from llm_session_reco.metrics import evaluate_ranks  # noqa: E402
from llm_session_reco.statistics import paired_bootstrap_ci, wilcoxon_signed_rank  # noqa: E402
from scripts.evaluate_ensemble import _bucket_by_prefix_length, _single_model_name  # noqa: E402

MODEL_3B = "qwen2.5:3b-instruct"
MODEL_7B = "qwen2.5:7b-instruct"
ModelRecord = dict[str, Any]


def _read_jsonl(path: Path) -> list[ModelRecord]:
    records: list[ModelRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(record)
    return records


def _session_id(record: ModelRecord) -> str:
    value = record.get("session_id") or record.get("trial_id")
    if value is None:
        raise ValueError("Every model trial must include session_id or trial_id")
    return str(value)


def _session_map(records: list[ModelRecord], model_label: str) -> dict[str, ModelRecord]:
    mapped: dict[str, ModelRecord] = {}
    for record in records:
        session_id = _session_id(record)
        if session_id in mapped:
            raise ValueError(f"Duplicate {model_label} trial for session {session_id}")
        mapped[session_id] = record
    return mapped


def _rank(record: ModelRecord, key: str) -> int | None:
    if key == "single_prompt_baseline":
        member_trials = record.get("member_trials")
        if not isinstance(member_trials, list) or not member_trials:
            return None
        first = member_trials[0]
        if not isinstance(first, dict) or first.get("parse_success") is not True:
            return None
        rank = first.get("target_rank")
    else:
        result = record.get(key)
        if not isinstance(result, dict):
            return None
        rank = result.get("target_rank")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        return None
    return rank


def _has_paired_ranks(record: ModelRecord) -> bool:
    """Both numbers the headline comparisons need exist for this model."""

    return (
        _rank(record, "single_prompt_baseline") is not None
        and _rank(record, "reliability_weighted") is not None
    )


def _is_complete_success(record: ModelRecord) -> bool:
    """Every prompt answer parsed and RWRA succeeded (complete-case sensitivity)."""

    trials = record.get("member_trials")
    return (
        isinstance(trials, list)
        and bool(trials)
        and all(isinstance(trial, dict) and trial.get("parse_success") is True for trial in trials)
        and record.get("failure_reason") is None
        and _has_paired_ranks(record)
    )


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _drift_tau(record: ModelRecord) -> float | None:
    drift = record.get("drift")
    if not isinstance(drift, dict):
        return None
    return _finite_float(drift.get("mean_kendall_tau"))


def _broken_answer_rate(record: ModelRecord) -> tuple[int, int] | None:
    trials = record.get("member_trials")
    if not isinstance(trials, list) or not trials:
        return None
    broken = sum(
        not isinstance(trial, dict) or trial.get("parse_success") is not True
        for trial in trials
    )
    return int(broken), len(trials)


def _validate_shared_session_data(
    session_id: str, record_3b: ModelRecord, record_7b: ModelRecord
) -> None:
    for field in ("prefix_item_ids", "candidate_item_ids", "target_item_id"):
        if field not in record_3b or field not in record_7b:
            raise ValueError(f"Session {session_id} is missing shared field {field!r}")
        if record_3b[field] != record_7b[field]:
            raise ValueError(
                f"The 3B and 7B trials use different {field} for session {session_id}"
            )
    # Identical treatment also means the same prompt members. Real trial records
    # always carry "members"; only hand-built records may omit it from both.
    if ("members" in record_3b) != ("members" in record_7b) or (
        record_3b.get("members") != record_7b.get("members")
    ):
        raise ValueError(
            f"The 3B and 7B trials used different prompt members for session {session_id}"
        )


def _paired_test(
    values_3b: list[float],
    values_7b: list[float],
    *,
    metric: str,
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    if len(values_3b) != len(values_7b):
        raise ValueError("Paired model measurements must have equal lengths")
    result: dict[str, object] = {
        "metric": metric,
        "n_pairs": len(values_3b),
        "mean_3b": mean(values_3b) if values_3b else None,
        "mean_7b": mean(values_7b) if values_7b else None,
        "mean_difference_3b_minus_7b": (
            mean(a - b for a, b in zip(values_3b, values_7b))
            if values_3b
            else None
        ),
        "bootstrap_ci": None,
        "wilcoxon": None,
    }
    if len(values_3b) >= 2:
        result["bootstrap_ci"] = paired_bootstrap_ci(
            values_3b, values_7b, n_boot=n_boot, seed=seed
        )
        result["wilcoxon"] = wilcoxon_signed_rank(values_3b, values_7b)
    return result


def _reciprocal_rank(rank: int | None) -> float | None:
    return None if rank is None else 1.0 / rank


def _condition_summary(records: list[ModelRecord], condition: str) -> dict[str, object]:
    ranks = [rank for record in records if (rank := _rank(record, condition)) is not None]
    return {"valid_sessions": len(ranks), "metrics": evaluate_ranks(ranks)}


def _mean_rwra_gain(records: list[ModelRecord]) -> float | None:
    gains: list[float] = []
    for record in records:
        baseline = _reciprocal_rank(_rank(record, "single_prompt_baseline"))
        rwra = _reciprocal_rank(_rank(record, "reliability_weighted"))
        if baseline is not None and rwra is not None:
            gains.append(rwra - baseline)
    return mean(gains) if gains else None


def _model_summary(records: list[ModelRecord]) -> dict[str, object]:
    return {
        "session_count": len(records),
        "single_prompt_baseline": _condition_summary(records, "single_prompt_baseline"),
        "reliability_weighted": _condition_summary(records, "reliability_weighted"),
        "mean_rwra_gain_reciprocal_rank": _mean_rwra_gain(records),
    }


def _paired_rwra_gains(
    records_3b: dict[str, ModelRecord],
    records_7b: dict[str, ModelRecord],
    session_ids: list[str],
) -> tuple[list[float], list[float]]:
    gains_3b: list[float] = []
    gains_7b: list[float] = []
    for session_id in session_ids:
        r3 = records_3b[session_id]
        r7 = records_7b[session_id]
        single_3b = _reciprocal_rank(_rank(r3, "single_prompt_baseline"))
        rwra_3b = _reciprocal_rank(_rank(r3, "reliability_weighted"))
        single_7b = _reciprocal_rank(_rank(r7, "single_prompt_baseline"))
        rwra_7b = _reciprocal_rank(_rank(r7, "reliability_weighted"))
        if None not in (single_3b, rwra_3b, single_7b, rwra_7b):
            gains_3b.append(float(rwra_3b) - float(single_3b))
            gains_7b.append(float(rwra_7b) - float(single_7b))
    return gains_3b, gains_7b


def _headline_comparisons(
    trials_3b: dict[str, ModelRecord],
    trials_7b: dict[str, ModelRecord],
    session_ids: list[str],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    """The three RWRA comparisons, over sessions where every input rank exists."""

    gain_3b, gain_7b = _paired_rwra_gains(trials_3b, trials_7b, session_ids)
    rwra_3b = [float(1 / _rank(trials_3b[sid], "reliability_weighted")) for sid in session_ids]
    rwra_7b = [float(1 / _rank(trials_7b[sid], "reliability_weighted")) for sid in session_ids]
    single_7b = [
        float(1 / _rank(trials_7b[sid], "single_prompt_baseline")) for sid in session_ids
    ]
    return {
        "rwra_gain_reciprocal_rank_3b_vs_7b": _paired_test(
            gain_3b, gain_7b, metric="change_in_reciprocal_rank", n_boot=n_boot, seed=seed
        ),
        "rwra_reciprocal_rank_3b_vs_7b": _paired_test(
            rwra_3b, rwra_7b, metric="reciprocal_rank", n_boot=n_boot, seed=seed
        ),
        "3b_rwra_vs_7b_single_prompt": _paired_test(
            rwra_3b, single_7b, metric="reciprocal_rank", n_boot=n_boot, seed=seed
        ),
    }


def compare_model_records(
    records_3b: list[ModelRecord],
    records_7b: list[ModelRecord],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    expected_3b: str = MODEL_3B,
    expected_7b: str = MODEL_7B,
) -> dict[str, object]:
    model_3b = _single_model_name(records_3b)
    model_7b = _single_model_name(records_7b)
    if model_3b != expected_3b:
        raise ValueError(f"Expected {expected_3b} results, found {model_3b!r}")
    if model_7b != expected_7b:
        raise ValueError(f"Expected {expected_7b} results, found {model_7b!r}")

    trials_3b = _session_map(records_3b, "3B")
    trials_7b = _session_map(records_7b, "7B")
    matched_ids = sorted(trials_3b.keys() & trials_7b.keys())
    for session_id in matched_ids:
        _validate_shared_session_data(session_id, trials_3b[session_id], trials_7b[session_id])

    # Primary sample: every matched session where both models produced the
    # single-prompt and RWRA ranks being compared -- the same availability rule
    # evaluate_ensemble.py uses. It keeps sessions where RWRA succeeded despite a
    # broken prompt answer; requiring all four answers to parse would drop the
    # sessions hardest for the weaker model and favour it.
    analysis_ids = [
        sid
        for sid in matched_ids
        if _has_paired_ranks(trials_3b[sid]) and _has_paired_ranks(trials_7b[sid])
    ]
    # Sensitivity sample: all four answers parsed for both models.
    complete_ids = [
        sid
        for sid in analysis_ids
        if _is_complete_success(trials_3b[sid]) and _is_complete_success(trials_7b[sid])
    ]
    analysis_3b = [trials_3b[sid] for sid in analysis_ids]
    analysis_7b = [trials_7b[sid] for sid in analysis_ids]

    drift_ids = [
        sid
        for sid in matched_ids
        if _drift_tau(trials_3b[sid]) is not None and _drift_tau(trials_7b[sid]) is not None
    ]
    drift_3b = [float(_drift_tau(trials_3b[sid])) for sid in drift_ids]
    drift_7b = [float(_drift_tau(trials_7b[sid])) for sid in drift_ids]

    broken_ids = [
        sid
        for sid in matched_ids
        if _broken_answer_rate(trials_3b[sid]) is not None
        and _broken_answer_rate(trials_7b[sid]) is not None
    ]
    broken_rates_3b: list[float] = []
    broken_rates_7b: list[float] = []
    broken_counts: dict[str, dict[str, int]] = {
        "3b": {"broken_answers": 0, "attempted_answers": 0},
        "7b": {"broken_answers": 0, "attempted_answers": 0},
    }
    for session_id in matched_ids:
        for label, record in (("3b", trials_3b[session_id]), ("7b", trials_7b[session_id])):
            rate_data = _broken_answer_rate(record)
            if rate_data is not None:
                broken, attempted = rate_data
                broken_counts[label]["broken_answers"] += broken
                broken_counts[label]["attempted_answers"] += attempted
    for session_id in broken_ids:
        broken_3b, total_3b = _broken_answer_rate(trials_3b[session_id]) or (0, 0)
        broken_7b, total_7b = _broken_answer_rate(trials_7b[session_id]) or (0, 0)
        broken_rates_3b.append(broken_3b / total_3b)
        broken_rates_7b.append(broken_7b / total_7b)

    def overall_broken_rate(label: str) -> float | None:
        total = broken_counts[label]["attempted_answers"]
        return broken_counts[label]["broken_answers"] / total if total else None

    comparisons = {
        **_headline_comparisons(trials_3b, trials_7b, analysis_ids, n_boot=n_boot, seed=seed),
        "drift_mean_kendall_tau_3b_vs_7b": _paired_test(
            drift_3b, drift_7b, metric="mean_kendall_tau", n_boot=n_boot, seed=seed
        ),
        "broken_answer_rate_3b_vs_7b": _paired_test(
            broken_rates_3b,
            broken_rates_7b,
            metric="per_session_broken_answer_rate",
            n_boot=n_boot,
            seed=seed,
        ),
    }

    history_groups: dict[str, object] = {}
    for group, group_records in _bucket_by_prefix_length(analysis_3b).items():
        group_ids = [_session_id(record) for record in group_records]
        history_groups[group] = {
            "session_count": len(group_ids),
            "conditions": {
                "3b": _model_summary([trials_3b[sid] for sid in group_ids]),
                "7b": _model_summary([trials_7b[sid] for sid in group_ids]),
            },
            "comparisons": _headline_comparisons(
                trials_3b, trials_7b, group_ids, n_boot=n_boot, seed=seed
            ),
        }

    return {
        "schema_version": "qwen_model_comparison_v2",
        "models": {"3b": model_3b, "7b": model_7b},
        "session_alignment": {
            "3b_trials": len(trials_3b),
            "7b_trials": len(trials_7b),
            "matched_sessions": len(matched_ids),
            "only_in_3b": len(trials_3b.keys() - trials_7b.keys()),
            "only_in_7b": len(trials_7b.keys() - trials_3b.keys()),
            "analysis_sessions": len(analysis_ids),
            "complete_success_sessions": len(complete_ids),
        },
        "overall": {
            "conditions": {
                "3b": _model_summary(analysis_3b),
                "7b": _model_summary(analysis_7b),
            },
            "comparisons": comparisons,
            "drift": {
                "3b_mean_kendall_tau": mean(drift_3b) if drift_3b else None,
                "7b_mean_kendall_tau": mean(drift_7b) if drift_7b else None,
                "sessions_with_both_scores": len(drift_ids),
            },
            "broken_answers": {
                "3b": {
                    **broken_counts["3b"],
                    "rate": overall_broken_rate("3b"),
                },
                "7b": {
                    **broken_counts["7b"],
                    "rate": overall_broken_rate("7b"),
                },
                "sessions_with_both_rates": len(broken_ids),
            },
        },
        "complete_case_sensitivity": {
            "session_count": len(complete_ids),
            "comparisons": _headline_comparisons(
                trials_3b, trials_7b, complete_ids, n_boot=n_boot, seed=seed
            ),
        },
        "history_groups": history_groups,
        "analysis_notes": {
            "analysis_sample": (
                "Performance and history-group comparisons use every matched session "
                "where both models produced the single-prompt and RWRA ranks being "
                "compared, including sessions where RWRA succeeded despite a broken "
                "prompt answer. This matches evaluate_ensemble.py's availability rule."
            ),
            "complete_case_sensitivity": (
                "Repeats the headline comparisons on the stricter cohort where all "
                "prompt answers parsed for both models. If it disagrees with the main "
                "result, the difference is driven by sessions with broken answers."
            ),
            "failure_sample": (
                "Broken-answer rates use all matched sessions with recorded member trials, "
                "including sessions excluded from the analysis sample."
            ),
            "statistics": (
                "Paired bootstrap confidence intervals and paired Wilcoxon tests compare "
                "per-session measurements; differences are 3B minus 7B."
            ),
        },
    }


def _default_output_path(input_3b: Path) -> Path:
    """Name the comparison after the dataset so the two domains never overwrite."""

    name = input_3b.name
    dataset = name.split("_3b_", 1)[0] if "_3b_" in name else None
    filename = f"{dataset}_model_comparison.json" if dataset else "model_comparison.json"
    return input_3b.with_name(filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-3b",
        type=Path,
        default=Path("data/processed/ml1m_3b_ensemble_trials.jsonl"),
    )
    parser.add_argument(
        "--input-7b",
        type=Path,
        default=Path("data/processed/ml1m_7b_ensemble_trials.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="defaults to <dataset>_model_comparison.json beside --input-3b",
    )
    parser.add_argument(
        "--model-3b",
        default=MODEL_3B,
        help="model name recorded in the smaller model's trials",
    )
    parser.add_argument(
        "--model-7b",
        default=MODEL_7B,
        help="model name recorded in the larger model's trials",
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or _default_output_path(args.input_3b)
    summary = compare_model_records(
        _read_jsonl(args.input_3b),
        _read_jsonl(args.input_7b),
        n_boot=args.n_boot,
        seed=args.seed,
        expected_3b=args.model_3b,
        expected_7b=args.model_7b,
    )
    summary["inputs"] = {"3b": str(args.input_3b), "7b": str(args.input_7b)}
    summary["output"] = str(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
