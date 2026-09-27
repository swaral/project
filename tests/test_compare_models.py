import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.compare_models import _default_output_path, compare_model_records

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _record(
    model,
    session_id,
    *,
    baseline_rank,
    rwra_rank,
    prefix_length=4,
    member_successes=None,
    members=None,
):
    successes = member_successes or [True, True, True, True]
    member_trials = [
        {
            "parse_success": success,
            "target_rank": baseline_rank if index == 0 and success else (1 if success else None),
        }
        for index, success in enumerate(successes)
    ]
    record = {
        "session_id": session_id,
        "client_config": {"model": model},
        "prefix_item_ids": list(range(prefix_length)),
        "candidate_item_ids": [10, 20, 30, 40],
        "target_item_id": 40,
        "member_trials": member_trials,
        # Mirrors run_ensemble.py: the ensemble only fails when RWRA could not be
        # computed; a broken answer alongside a successful RWRA is not a failure.
        "failure_reason": None if rwra_rank is not None else "member failure",
        "reliability_weighted": (
            {"target_rank": rwra_rank} if rwra_rank is not None else None
        ),
        "drift": {"mean_kendall_tau": 0.5},
    }
    if members is not None:
        record["members"] = members
    return record


def test_sessions_are_matched_by_id_and_unmatched_rows_are_reported():
    records_3b = [
        _record("qwen2.5:3b-instruct", "shared", baseline_rank=4, rwra_rank=2),
        _record("qwen2.5:3b-instruct", "only-3b", baseline_rank=3, rwra_rank=1),
    ]
    records_7b = [
        _record("qwen2.5:7b-instruct", "shared", baseline_rank=2, rwra_rank=1),
        _record("qwen2.5:7b-instruct", "only-7b", baseline_rank=1, rwra_rank=1),
    ]

    summary = compare_model_records(records_3b, records_7b, n_boot=100, seed=0)

    alignment = summary["session_alignment"]
    assert alignment["matched_sessions"] == 1
    assert alignment["only_in_3b"] == 1
    assert alignment["only_in_7b"] == 1
    assert alignment["complete_success_sessions"] == 1
    assert summary["overall"]["comparisons"]["rwra_gain_reciprocal_rank_3b_vs_7b"]["n_pairs"] == 1


def test_sessions_with_a_failure_in_either_model_are_excluded_from_quality_pairs():
    records_3b = [
        _record("qwen2.5:3b-instruct", "ok", baseline_rank=4, rwra_rank=1),
        _record(
            "qwen2.5:3b-instruct",
            "failed",
            baseline_rank=5,
            rwra_rank=None,
            member_successes=[True, False, True, True],
        ),
    ]
    records_7b = [
        _record("qwen2.5:7b-instruct", "ok", baseline_rank=2, rwra_rank=1),
        _record("qwen2.5:7b-instruct", "failed", baseline_rank=2, rwra_rank=1),
    ]

    summary = compare_model_records(records_3b, records_7b, n_boot=100, seed=0)

    assert summary["session_alignment"]["matched_sessions"] == 2
    assert summary["session_alignment"]["complete_success_sessions"] == 1
    assert (
        summary["overall"]["comparisons"]["rwra_gain_reciprocal_rank_3b_vs_7b"][
            "n_pairs"
        ]
        == 1
    )
    # Failure-rate analysis retains that failed session instead of hiding it.
    assert summary["overall"]["comparisons"]["broken_answer_rate_3b_vs_7b"]["n_pairs"] == 2
    assert summary["overall"]["broken_answers"]["3b"]["broken_answers"] == 1


def test_paired_rwra_gain_difference_is_correct_on_a_small_example():
    records_3b = [
        _record("qwen2.5:3b-instruct", "u1", baseline_rank=4, rwra_rank=1),
        _record("qwen2.5:3b-instruct", "u2", baseline_rank=10, rwra_rank=2),
    ]
    records_7b = [
        _record("qwen2.5:7b-instruct", "u1", baseline_rank=2, rwra_rank=1),
        _record("qwen2.5:7b-instruct", "u2", baseline_rank=5, rwra_rank=4),
    ]

    summary = compare_model_records(records_3b, records_7b, n_boot=500, seed=0)

    gain_comparison = summary["overall"]["comparisons"][
        "rwra_gain_reciprocal_rank_3b_vs_7b"
    ]
    assert gain_comparison["n_pairs"] == 2
    assert gain_comparison["mean_3b"] == pytest.approx(0.575)
    assert gain_comparison["mean_7b"] == pytest.approx(0.275)
    assert gain_comparison["mean_difference_3b_minus_7b"] == pytest.approx(0.3)
    assert gain_comparison["bootstrap_ci"]["mean_difference"] == pytest.approx(0.3)
    assert gain_comparison["wilcoxon"]["n_pairs"] == 2


def test_a_broken_answer_with_successful_rwra_stays_in_the_main_comparison():
    """Dropping these sessions would hide the weaker model's hardest cases."""

    records_3b = [
        _record("qwen2.5:3b-instruct", "clean", baseline_rank=4, rwra_rank=1),
        _record(
            "qwen2.5:3b-instruct",
            "one-broken",
            baseline_rank=5,
            rwra_rank=2,
            member_successes=[True, False, True, True],
        ),
    ]
    records_7b = [
        _record("qwen2.5:7b-instruct", "clean", baseline_rank=2, rwra_rank=1),
        _record("qwen2.5:7b-instruct", "one-broken", baseline_rank=2, rwra_rank=1),
    ]

    summary = compare_model_records(records_3b, records_7b, n_boot=100, seed=0)

    assert summary["session_alignment"]["analysis_sessions"] == 2
    assert summary["session_alignment"]["complete_success_sessions"] == 1
    main = summary["overall"]["comparisons"]["rwra_gain_reciprocal_rank_3b_vs_7b"]
    strict = summary["complete_case_sensitivity"]["comparisons"][
        "rwra_gain_reciprocal_rank_3b_vs_7b"
    ]
    assert main["n_pairs"] == 2
    assert strict["n_pairs"] == 1
    assert summary["overall"]["conditions"]["3b"]["session_count"] == 2


def test_models_that_used_different_prompts_are_rejected():
    four = [{"variant_id": f"v{i}", "context_variant_id": "c"} for i in range(4)]
    three = four[:3]
    records_3b = [
        _record("qwen2.5:3b-instruct", "u1", baseline_rank=2, rwra_rank=1, members=four)
    ]
    records_7b = [
        _record("qwen2.5:7b-instruct", "u1", baseline_rank=2, rwra_rank=1, members=three)
    ]

    with pytest.raises(ValueError, match="different prompt members"):
        compare_model_records(records_3b, records_7b, n_boot=100, seed=0)


def test_expected_model_names_can_be_changed_for_other_providers():
    records_3b = [_record("provider/qwen-3b", "u1", baseline_rank=2, rwra_rank=1)]
    records_7b = [_record("provider/qwen-7b", "u1", baseline_rank=2, rwra_rank=1)]

    with pytest.raises(ValueError, match="Expected qwen2.5:3b-instruct"):
        compare_model_records(records_3b, records_7b, n_boot=100, seed=0)

    summary = compare_model_records(
        records_3b,
        records_7b,
        n_boot=100,
        seed=0,
        expected_3b="provider/qwen-3b",
        expected_7b="provider/qwen-7b",
    )
    assert summary["models"] == {"3b": "provider/qwen-3b", "7b": "provider/qwen-7b"}


def test_default_output_is_named_per_dataset_so_domains_do_not_overwrite():
    ml1m = _default_output_path(Path("data/processed/ml1m_3b_ensemble_trials.jsonl"))
    amazon = _default_output_path(
        Path("data/processed/amazon_games_3b_ensemble_trials.jsonl")
    )

    assert ml1m.name == "ml1m_model_comparison.json"
    assert amazon.name == "amazon_games_model_comparison.json"


def test_script_runs_as_a_file_not_only_under_pytest():
    """`python scripts/compare_models.py` must resolve its sibling-script import."""

    # src/ on PYTHONPATH mirrors pytest's own pythonpath setting, so this checks
    # the script's sibling import without depending on how the package is installed.
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "scripts/compare_models.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "--input-3b" in result.stdout
