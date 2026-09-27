"""Calculate ranking metrics from Step 4 baseline trial records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_session_reco.metrics import evaluate_ranks, extract_valid_ranks


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


def evaluate_records(records: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate HR and NDCG over valid parsed trials only."""

    valid_ranks = extract_valid_ranks(records)

    return {
        "schema_version": "step5_metrics_v1",
        "total_trials": len(records),
        "valid_trials": len(valid_ranks),
        "parse_failures": sum(
            record.get("parse_success") is False for record in records
        ),
        "request_failures": sum(
            record.get("request_success") is False for record in records
        ),
        "metrics": evaluate_ranks(valid_ranks),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/ml1m_baseline_trials.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/ml1m_baseline_metrics.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_records(_read_jsonl(args.input))
    summary["input"] = str(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
