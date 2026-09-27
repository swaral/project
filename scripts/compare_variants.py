"""Compare Step 5 metrics across prompt variants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_METRIC_FILES: dict[str, Path] = {
    "baseline_scores_v1": Path("data/processed/ml1m_baseline_metrics.json"),
    "wording_direct_v1": Path("data/processed/ml1m_wording_direct_metrics.json"),
    "wording_preference_v1": Path(
        "data/processed/ml1m_wording_preference_metrics.json"
    ),
    "wording_detailed_v1": Path(
        "data/processed/ml1m_wording_detailed_metrics.json"
    ),
}

METRIC_NAMES = ("HR@1", "HR@5", "HR@10", "NDCG@5", "NDCG@10")


def _read_metrics(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    if not isinstance(record, dict):
        raise ValueError(f"{path} is not a JSON object")
    return record


def _parse_metric_argument(value: str) -> tuple[str, Path]:
    variant_id, separator, path_text = value.partition("=")
    if not separator or not variant_id or not path_text:
        raise ValueError("metric inputs must use VARIANT_ID=PATH format")
    return variant_id, Path(path_text)


def compare_metrics(metrics_by_variant: dict[str, dict[str, object]]) -> dict[str, object]:
    """Create a JSON-serializable comparison summary."""

    rows: list[dict[str, object]] = []
    for variant_id, record in metrics_by_variant.items():
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"Metrics are missing for variant {variant_id}")
        row: dict[str, object] = {
            "variant_id": variant_id,
            "total_trials": record.get("total_trials"),
            "valid_trials": record.get("valid_trials"),
            "parse_failures": record.get("parse_failures"),
            "request_failures": record.get("request_failures"),
        }
        for metric_name in METRIC_NAMES:
            row[metric_name] = metrics.get(metric_name)
        rows.append(row)

    best_by_metric: dict[str, list[str]] = {}
    for metric_name in METRIC_NAMES:
        numeric_rows = [
            row
            for row in rows
            if isinstance(row.get(metric_name), (int, float))
        ]
        if not numeric_rows:
            best_by_metric[metric_name] = []
            continue
        best_value = max(float(row[metric_name]) for row in numeric_rows)
        best_by_metric[metric_name] = [
            str(row["variant_id"])
            for row in numeric_rows
            if float(row[metric_name]) == best_value
        ]

    return {
        "schema_version": "step7_variant_comparison_v1",
        "scope": "pilot_or_full_run_as_indicated_by_input_files",
        "variants": rows,
        "best_by_metric": best_by_metric,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "variant_id",
        "total_trials",
        "valid_trials",
        "parse_failures",
        "request_failures",
        *METRIC_NAMES,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        action="append",
        help="optional VARIANT_ID=PATH input; defaults to the four project metric files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/ml1m_variant_comparison.json"),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("data/processed/ml1m_variant_comparison.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric_files = DEFAULT_METRIC_FILES
    if args.metric:
        metric_files = dict(_parse_metric_argument(value) for value in args.metric)

    metrics_by_variant = {
        variant_id: _read_metrics(path)
        for variant_id, path in metric_files.items()
    }
    comparison = compare_metrics(metrics_by_variant)
    comparison["metric_files"] = {
        variant_id: str(path) for variant_id, path in metric_files.items()
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.csv_output, comparison["variants"])
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
