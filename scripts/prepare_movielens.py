"""Download MovieLens-1M and create chronological session examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_session_reco.movielens import (
    build_candidate_pools,
    build_leave_one_out_examples,
    build_leave_one_out_training_ratings,
    download_movielens_1m,
    load_movies,
    load_ratings,
    write_candidate_pools_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/ml1m_leave_one_out.jsonl"),
    )
    parser.add_argument(
        "--candidate-output",
        type=Path,
        default=Path("data/processed/ml1m_candidate_pools.jsonl"),
    )
    parser.add_argument("--min-history-length", type=int, default=3)
    parser.add_argument("--max-history-length", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = download_movielens_1m(args.data_root, force=args.force_download)
    ratings = load_ratings(dataset_dir)
    movies = load_movies(dataset_dir)
    examples = build_leave_one_out_examples(
        ratings,
        min_history_length=args.min_history_length,
        max_history_length=args.max_history_length,
    )
    training_ratings = build_leave_one_out_training_ratings(ratings)
    candidate_pools = build_candidate_pools(training_ratings, examples)
    write_jsonl(examples, args.output)
    write_candidate_pools_jsonl(candidate_pools, args.candidate_output)

    summary = {
        "dataset": "MovieLens-1M",
        "dataset_dir": str(dataset_dir),
        "ratings": int(len(ratings)),
        "users": int(ratings["user_id"].nunique()),
        "items": int(ratings["item_id"].nunique()),
        "movies_metadata_rows": int(len(movies)),
        "session_examples": len(examples),
        "training_ratings": int(len(training_ratings)),
        "candidate_pools": len(candidate_pools),
        "candidate_pool_size": 20,
        "candidate_policy": "training-only popularity; prefix items excluded",
        "target_position_policy": "balanced positions 0-19 in example order",
        "min_history_length": args.min_history_length,
        "max_history_length": args.max_history_length,
        "target_policy": "final chronological interaction per user",
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
