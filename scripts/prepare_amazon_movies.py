"""Download Amazon Movies & TV (2023), build the user subset and session examples.

Run this once on a local machine (the raw metadata is ~1.3 GB):

    python scripts/prepare_amazon_movies.py              # combined (amazon_movies)
    python scripts/prepare_amazon_movies.py --media movie  # movies only (amazon_film)
    python scripts/prepare_amazon_movies.py --media tv     # TV only (amazon_tv)

Steps: download the raw files into data/raw/amazon-movies/ (skipped when
present), build the deterministic user subset (skipped when present unless
--rebuild-subset), then write the same processed files as the games domain.
The Kaggle job reruns this script from scratch; the subset's SHA-256 in the
summary must match the local run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_session_reco.amazon_movies import (
    build_subset,
    download_amazon_movies,
    load_items,
    load_ratings,
    subset_file_names,
)
from llm_session_reco.session_dataset import (
    build_candidate_pools,
    build_leave_one_out_examples,
    build_leave_one_out_training_ratings,
    write_candidate_pools_jsonl,
    write_jsonl,
)

MEDIA = {
    "all": (None, "amazon_movies", "Amazon-Movies-and-TV-2023"),
    "movie": ("movie", "amazon_film", "Amazon-Movies-and-TV-2023, movies only"),
    "tv": ("tv", "amazon_tv", "Amazon-Movies-and-TV-2023, TV only"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--user-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-history-length", type=int, default=3)
    parser.add_argument(
        "--media", choices=sorted(MEDIA), default="all",
        help="all = combined Movies & TV; movie / tv = one half of the Addendum v10 split",
    )
    parser.add_argument("--rebuild-subset", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    media, prefix, dataset_name = MEDIA[args.media]
    subset_summary_path = out / f"{prefix}_subset.summary.json"

    raw_dir = download_amazon_movies(args.data_root, force=args.force_download)
    if args.rebuild_subset or not (raw_dir / subset_file_names(media)[0]).is_file():
        subset_summary = build_subset(
            raw_dir, user_fraction=args.user_fraction, seed=args.seed, media=media
        )
        subset_summary_path.write_text(json.dumps(subset_summary, indent=2) + "\n", encoding="utf-8")
    print(subset_summary_path.read_text(encoding="utf-8") if subset_summary_path.is_file() else "")

    id_map_path = out / f"{prefix}_id_map.json"
    ratings = load_ratings(raw_dir, id_map_output=id_map_path, media=media)
    items = load_items(
        raw_dir, json.loads(id_map_path.read_text(encoding="utf-8"))["item_id_map"], media=media
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
        "dataset": f"{dataset_name} (5core rating_only, hashed user subset, re-5-cored)",
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
