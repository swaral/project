from scripts.evaluate_baseline import evaluate_records


def test_evaluate_records_calculates_hr_and_ndcg_on_valid_trials_only():
    summary = evaluate_records(
        [
            {
                "request_success": True,
                "parse_success": True,
                "target_rank": 1,
            },
            {
                "request_success": True,
                "parse_success": True,
                "target_rank": 6,
            },
            {
                "request_success": True,
                "parse_success": False,
                "target_rank": None,
            },
        ]
    )

    assert summary["total_trials"] == 3
    assert summary["valid_trials"] == 2
    assert summary["parse_failures"] == 1
    assert summary["request_failures"] == 0
    assert summary["metrics"]["HR@1"] == 0.5
    assert summary["metrics"]["HR@5"] == 0.5
    assert summary["metrics"]["HR@10"] == 1.0
    assert summary["metrics"]["NDCG@5"] == 0.5


def test_evaluate_records_returns_null_metrics_without_valid_trials():
    summary = evaluate_records(
        [
            {
                "request_success": False,
                "parse_success": False,
                "target_rank": None,
            }
        ]
    )

    assert summary["valid_trials"] == 0
    assert summary["request_failures"] == 1
    assert summary["metrics"]["HR@1"] is None
