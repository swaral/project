from scripts.compare_variants import compare_metrics


def test_compare_metrics_identifies_best_variant_per_metric():
    comparison = compare_metrics(
        {
            "baseline": {
                "total_trials": 5,
                "valid_trials": 5,
                "parse_failures": 0,
                "request_failures": 0,
                "metrics": {
                    "HR@1": 0.2,
                    "HR@5": 0.2,
                    "HR@10": 0.4,
                    "NDCG@5": 0.2,
                    "NDCG@10": 0.2,
                },
            },
            "direct": {
                "total_trials": 5,
                "valid_trials": 5,
                "parse_failures": 0,
                "request_failures": 0,
                "metrics": {
                    "HR@1": 0.4,
                    "HR@5": 0.4,
                    "HR@10": 0.6,
                    "NDCG@5": 0.4,
                    "NDCG@10": 0.46,
                },
            },
        }
    )

    assert len(comparison["variants"]) == 2
    assert comparison["best_by_metric"]["HR@1"] == ["direct"]
    assert comparison["best_by_metric"]["NDCG@10"] == ["direct"]
