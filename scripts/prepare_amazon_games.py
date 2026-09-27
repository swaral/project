"""Download Amazon Video Games (2023) and create chronological session examples.

Mirrors scripts/prepare_movielens.py, but for the second (cross-domain) data
source. See src/llm_session_reco/amazon_games.py for the dataset source and
the string-ID -> integer-ID remapping this script relies on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_session_reco.amazon_games import download_amazon_games, load_items, load_ratings
from llm_session_reco.session_dataset import (
    build_candidate_pools,
    build_leave_one_out_examples,
    build_leave_one_out_training_ratings,
    write_candidate_pools_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/amazon_games_leave_one_out.jsonl"),
    )
    parser.add_argument(
        "--candidate-output",
        type=Path,
        default=Path("data/processed/amazon_games_candidate_pools.jsonl"),
    )
    parser.add_argument(
        "--id-map-output",
        type=Path,
        default=Path("data/processed/amazon_games_id_map.json"),
    )
    parser.add_argument(
        "--items-output",
        type=Path,
        default=Path("data/processed/amazon_games_items.jsonl"),
    )
    parser.add_argument("--min-history-length", type=int, default=3)
    parser.add_argument("--max-history-length", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = download_amazon_games(args.data_root, force=args.force_download)
    ratings = load_ratings(dataset_dir, id_map_output=args.id_map_output)
    id_maps = json.loads(args.id_map_output.read_text(encoding="utf-8"))
    items = load_items(dataset_dir, id_maps["item_id_map"])

    args.items_output.parent.mkdir(parents=True, exist_ok=True)
    with args.items_output.open("w", encoding="utf-8") as handle:
        for record in items.to_dict("records"):
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")

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
        "dataset": "Amazon-Video-Games-2023 (5core, rating_only)",
        "dataset_dir": str(dataset_dir),
        "ratings": int(len(ratings)),
        "users": int(ratings["user_id"].nunique()),
        "items": int(ratings["item_id"].nunique()),
        "items_with_metadata": int(len(items)),
        "session_examples": len(examples),
        "training_ratings": int(len(training_ratings)),
        "candidate_pools": len(candidate_pools),
        "candidate_pool_size": 20,
        "candidate_policy": "training-only popularity; prefix items excluded",
        "target_position_policy": "balanced positions 0-19 in example order",
        "min_history_length": args.min_history_length,
        "max_history_length": args.max_history_length,
        "target_policy": "final chronological interaction per user",
        "id_remapping": "Amazon reviewer_id/ASIN strings mapped to dense sorted integers; "
        f"see {args.id_map_output}",
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
