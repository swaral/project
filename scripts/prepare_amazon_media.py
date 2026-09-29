"""Download Amazon Books or CDs & Vinyl (2023), build the user subset and session examples.

    python scripts/prepare_amazon_media.py --domain amazon_books
    python scripts/prepare_amazon_media.py --domain amazon_music

Steps: download the ratings CSV into data/raw/<subdir>/ (skipped when
present), stream the metadata once into a compact cache (the Books file is
~14 GB and is never stored), build the deterministic user subset (skipped
when present unless --rebuild-subset), then write the same processed files
as the other Amazon domains. The Kaggle job reruns this script from scratch;
the subset's SHA-256 in the summary must match the local run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_session_reco.amazon_media import (
    CATEGORIES,
    build_subset,
    download_ratings,
    load_items,
    load_ratings,
)
from llm_session_reco.session_dataset import (
    build_candidate_pools,
    build_leave_one_out_examples,
    build_leave_one_out_training_ratings,
    write_candidate_pools_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=sorted(CATEGORIES), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--user-fraction", type=float, default=None,
        help="share of users kept; defaults to the domain's frozen value (Addendum v12)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-history-length", type=int, default=3)
    parser.add_argument("--rebuild-subset", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    category = CATEGORIES[args.domain]
    prefix = args.domain
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    subset_summary_path = out / f"{prefix}_subset.summary.json"

    raw_dir = download_ratings(args.domain, args.data_root, force=args.force_download)
    if args.rebuild_subset or not (raw_dir / category.subset_ratings).is_file():
        subset_summary = build_subset(
            args.domain, raw_dir, user_fraction=args.user_fraction, seed=args.seed
        )
        subset_summary_path.write_text(json.dumps(subset_summary, indent=2) + "\n", encoding="utf-8")
    print(subset_summary_path.read_text(encoding="utf-8") if subset_summary_path.is_file() else "")

    id_map_path = out / f"{prefix}_id_map.json"
    ratings = load_ratings(args.domain, raw_dir, id_map_output=id_map_path)
    items = load_items(
        args.domain, raw_dir, json.loads(id_map_path.read_text(encoding="utf-8"))["item_id_map"]
    )
    with (out / f"{prefix}_items.jsonl").open("w", encoding="utf-8") as handle:
        for record in items.to_dict("records"):
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    examples = build_leave_one_out_examples(ratings, min_history_length=args.min_history_length)
    training_ratings = build_leave_one_out_training_ratings(ratings)
    candidate_pools = build_candidate_pools(training_ratings, examples)
    write_jsonl(examples, out / f"{prefix}_leave_one_out.jsonl")
    write_candidate_pools_jsonl(candidate_pools, out / f"{prefix}_candidate_pools.jsonl")

    summary = {
        "dataset": f"Amazon-Reviews-2023 {category.hf_name} (5core rating_only, hashed user subset, re-5-cored)",
        "ratings": int(len(ratings)),
        "users": int(ratings["user_id"].nunique()),
        "items": int(ratings["item_id"].nunique()),
        "items_with_metadata": int(len(items)),
        "session_examples": len(examples),
        "min_history_length": args.min_history_length,
        "target_policy": "final chronological interaction per user",
        "subset_summary": str(subset_summary_path),
    }
    (out / f"{prefix}_leave_one_out.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
