from scripts.evaluate_ensemble import evaluate_ensemble_records


def _record(
    *,
    baseline_rank,
    naive_rank,
    rwra_rank,
    long_tail_rank=None,
    prefix_length=None,
    kendall_tau=1.0,
    jaccard5=1.0,
    failure_reason=None,
):
    record = {
        "client_config": {"model": "qwen2.5:3b-instruct"},
        "member_trials": [
            {
                "parse_success": baseline_rank is not None,
                "target_rank": baseline_rank,
            }
        ],
        "naive_mean": (
            {"target_rank": naive_rank} if naive_rank is not None else None
        ),
        "reliability_weighted": (
            {"target_rank": rwra_rank} if rwra_rank is not None else None
        ),
        "long_tail_weighted": (
            {"target_rank": long_tail_rank} if long_tail_rank is not None else None
        ),
        "failure_reason": failure_reason,
        "drift": {"mean_kendall_tau": kendall_tau, "mean_jaccard_at_5": jaccard5},
    }
    if prefix_length is not None:
        record["prefix_item_ids"] = list(range(prefix_length))
    return record


def test_condition_summaries_use_the_right_outcome_per_condition():
    records = [
        _record(baseline_rank=5, naive_rank=2, rwra_rank=1),
        _record(baseline_rank=10, naive_rank=8, rwra_rank=3),
    ]
    summary = evaluate_ensemble_records(records, n_boot=200, seed=0)

    assert summary["conditions"]["single_prompt_baseline"]["valid_trials"] == 2
    assert summary["conditions"]["single_prompt_baseline"]["metrics"]["HR@5"] == 0.5
    assert summary["conditions"]["reliability_weighted"]["metrics"]["HR@5"] == 1.0
    assert summary["conditions"]["naive_mean_ensemble"]["valid_trials"] == 2


def test_failed_ensemble_is_excluded_from_aggregated_conditions_but_not_baseline():
    records = [
        _record(
            baseline_rank=4,
            naive_rank=None,
            rwra_rank=None,
            failure_reason="only 1 of 4 members parsed successfully",
        ),
    ]
    summary = evaluate_ensemble_records(records, n_boot=200, seed=0)

    assert summary["conditions"]["single_prompt_baseline"]["valid_trials"] == 1
    assert summary["conditions"]["naive_mean_ensemble"]["valid_trials"] == 0
    assert summary["conditions"]["reliability_weighted"]["valid_trials"] == 0


def test_comparisons_only_include_sessions_valid_in_both_conditions():
    records = [
        _record(baseline_rank=1, naive_rank=1, rwra_rank=1),
        _record(baseline_rank=None, naive_rank=None, rwra_rank=2),  # baseline missing
    ]
    summary = evaluate_ensemble_records(records, n_boot=200, seed=0)

    comparison = summary["comparisons"]["reliability_weighted_vs_single_prompt_baseline"]
    assert comparison["n_pairs"] == 1
    # fewer than 2 pairs -> no bootstrap/wilcoxon computed
    assert comparison["bootstrap_ci"] is None
    assert comparison["wilcoxon"] is None


def test_comparison_runs_statistics_with_enough_pairs():
    records = [
        _record(baseline_rank=5, naive_rank=3, rwra_rank=1),
        _record(baseline_rank=6, naive_rank=4, rwra_rank=2),
        _record(baseline_rank=7, naive_rank=5, rwra_rank=1),
    ]
    summary = evaluate_ensemble_records(records, n_boot=200, seed=0)

    comparison = summary["comparisons"]["reliability_weighted_vs_single_prompt_baseline"]
    assert comparison["n_pairs"] == 3
    assert comparison["bootstrap_ci"] is not None
    assert comparison["bootstrap_ci"]["mean_difference"] > 0
    assert comparison["wilcoxon"] is not None


def test_drift_summary_averages_only_defined_scores():
    records = [
        _record(baseline_rank=1, naive_rank=1, rwra_rank=1, kendall_tau=1.0, jaccard5=1.0),
        _record(baseline_rank=1, naive_rank=1, rwra_rank=1, kendall_tau=0.0, jaccard5=0.5),
        _record(
            baseline_rank=None,
            naive_rank=None,
            rwra_rank=None,
            kendall_tau=None,
            jaccard5=None,
            failure_reason="only 0 of 4 members parsed successfully",
        ),
    ]
    summary = evaluate_ensemble_records(records, n_boot=200, seed=0)

    assert summary["drift"]["sessions_with_drift_score"] == 2
    assert summary["drift"]["mean_kendall_tau"] == 0.5
    assert summary["drift"]["mean_jaccard_at_5"] == 0.75


def test_long_tail_weighted_condition_and_comparisons_are_reported():
    records = [
        _record(baseline_rank=5, naive_rank=3, rwra_rank=2, long_tail_rank=1),
        _record(baseline_rank=6, naive_rank=4, rwra_rank=3, long_tail_rank=1),
        _record(baseline_rank=7, naive_rank=5, rwra_rank=1, long_tail_rank=2),
    ]
    summary = evaluate_ensemble_records(records, n_boot=200, seed=0)

    assert summary["conditions"]["long_tail_weighted"]["valid_trials"] == 3
    assert summary["conditions"]["long_tail_weighted"]["metrics"]["HR@1"] == 2 / 3

    lt_vs_baseline = summary["comparisons"]["long_tail_weighted_vs_single_prompt_baseline"]
    assert lt_vs_baseline["n_pairs"] == 3
    assert lt_vs_baseline["bootstrap_ci"]["mean_difference"] > 0

    lt_vs_rwra = summary["comparisons"]["long_tail_weighted_vs_reliability_weighted"]
    assert lt_vs_rwra["n_pairs"] == 3


def test_records_without_long_tail_field_are_excluded_from_that_condition():
    records = [_record(baseline_rank=1, naive_rank=1, rwra_rank=1)]  # no long_tail_rank
    summary = evaluate_ensemble_records(records, n_boot=200, seed=0)

    assert summary["conditions"]["long_tail_weighted"]["valid_trials"] == 0
    assert summary["conditions"]["long_tail_weighted"]["failed_trials"] == 1


def test_metrics_record_the_model_name():
    summary = evaluate_ensemble_records(
        [_record(baseline_rank=1, naive_rank=1, rwra_rank=1)], n_boot=200, seed=0
    )

    assert summary["model"] == "qwen2.5:3b-instruct"


def test_evaluator_rejects_a_file_that_mixes_models():
    records = [
        _record(baseline_rank=1, naive_rank=1, rwra_rank=1),
        _record(baseline_rank=2, naive_rank=2, rwra_rank=2),
    ]
    records[1]["client_config"]["model"] = "qwen2.5:7b-instruct"

    import pytest

    with pytest.raises(ValueError, match="Mixed model results"):
        evaluate_ensemble_records(records, n_boot=200, seed=0)


def test_model_independent_reference_baselines_can_be_skipped_on_second_run():
    summary = evaluate_ensemble_records(
        [
            _record_with_pool(
                rwra_rank=1,
                candidate_item_ids=[10, 20, 30],
                target_item_id=30,
                session_id="user-1",
            )
        ],
        n_boot=200,
        seed=0,
        include_reference_baselines=False,
    )

    assert summary["reference_baselines"]["computed"] is False
    assert "random_baseline" not in summary["conditions"]
    assert "popularity_baseline" not in summary["conditions"]


def test_stratified_by_history_length_splits_into_three_roughly_equal_buckets():
    records = [
        _record(baseline_rank=1, naive_rank=1, rwra_rank=1, prefix_length=length)
        for length in (3, 4, 5, 20, 21, 22, 45, 46, 47)
    ]
    summary = evaluate_ensemble_records(records, n_boot=200, seed=0)
    strata = summary["stratified_by_history_length"]

    assert strata["short_history"]["session_count"] == 3
    assert strata["medium_history"]["session_count"] == 3
    assert strata["long_history"]["session_count"] == 3
    assert strata["short_history"]["conditions"]["single_prompt_baseline"]["valid_trials"] == 3


def test_stratification_ignores_records_missing_prefix_length():
    records = [_record(baseline_rank=1, naive_rank=1, rwra_rank=1)]  # no prefix_length
    summary = evaluate_ensemble_records(records, n_boot=200, seed=0)
    strata = summary["stratified_by_history_length"]

    assert (
        strata["short_history"]["session_count"]
        + strata["medium_history"]["session_count"]
        + strata["long_history"]["session_count"]
        == 0
    )


def _record_with_pool(*, rwra_rank, candidate_item_ids, target_item_id, session_id):
    record = _record(baseline_rank=rwra_rank, naive_rank=rwra_rank, rwra_rank=rwra_rank)
    record["session_id"] = session_id
    record["candidate_item_ids"] = list(candidate_item_ids)
    record["target_item_id"] = target_item_id
    return record


def test_reference_rows_are_absent_when_records_carry_no_candidate_pool():
    records = [_record(baseline_rank=5, naive_rank=2, rwra_rank=1)]
    summary = evaluate_ensemble_records(records, n_boot=200, seed=0)

    assert "random_baseline" not in summary["conditions"]
    assert "popularity_baseline" not in summary["conditions"]
    assert summary["reference_baselines"]["random_available"] is False


def test_random_row_appears_with_pools_and_popularity_row_needs_the_table():
    records = [
        _record_with_pool(
            rwra_rank=1,
            candidate_item_ids=[10, 20, 30, 40],
            target_item_id=40,
            session_id="user-1",
        )
    ]

    without_popularity = evaluate_ensemble_records(records, n_boot=200, seed=0)
    assert without_popularity["conditions"]["random_baseline"]["valid_trials"] == 1
    assert "popularity_baseline" not in without_popularity["conditions"]
    assert without_popularity["reference_baselines"]["popularity_available"] is False

    with_popularity = evaluate_ensemble_records(
        records,
        n_boot=200,
        seed=0,
        item_popularity={10: 9, 20: 8, 30: 7, 40: 1},
    )
    popularity_row = with_popularity["conditions"]["popularity_baseline"]
    assert popularity_row["valid_trials"] == 1
    # The target is the least popular item in the pool, so it ranks 4th of 4.
    assert popularity_row["metrics"]["HR@1"] == 0.0
    assert with_popularity["reference_baselines"]["popularity_available"] is True
    assert with_popularity["schema_version"] == "ensemble_evaluation_v3"


def test_reference_comparisons_are_reported_only_for_available_rows():
    records = [
        _record_with_pool(
            rwra_rank=1,
            candidate_item_ids=[10, 20, 30, 40],
            target_item_id=40,
            session_id=f"user-{index}",
        )
        for index in range(1, 4)
    ]
    summary = evaluate_ensemble_records(
        records, n_boot=200, seed=0, item_popularity={10: 9, 20: 8, 30: 7, 40: 1}
    )

    assert "reliability_weighted_vs_random_baseline" in summary["comparisons"]
    assert "reliability_weighted_vs_popularity_baseline" in summary["comparisons"]

    without_popularity = evaluate_ensemble_records(records, n_boot=200, seed=0)
    assert (
        "reliability_weighted_vs_popularity_baseline"
        not in without_popularity["comparisons"]
    )
