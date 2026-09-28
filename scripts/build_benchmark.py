"""Build the repaired benchmark for one domain.

Writes, under data/processed/:
- {prefix}_clean_examples.jsonl: positive, uniquely-last targets with a
  validation/test split;
- {prefix}_pool_popularity_matched.jsonl: the clean test pools;
- {prefix}_pool_retrieval.jsonl: item-KNN retrieval pools (training data only);
- {prefix}_pool_top_popular.jsonl: the original top-19-popular design on the
  same examples, kept only as a shortcut diagnostic;
- {prefix}_benchmark_summary.json.

Run scripts/prepare_movielens.py or scripts/prepare_amazon_games.py first so
the raw data (and the Amazon item file) exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_session_reco.baselines import build_item_popularity
from llm_session_reco.benchmark import (
    BenchmarkPool,
    ItemKNNRetriever,
    build_clean_examples,
    build_popularity_matched_pools,
    build_retrieval_pools,
    popularity_midrank,
    write_records,
)
from llm_session_reco.domains import DOMAINS, load_domain
from llm_session_reco.session_dataset import (
    build_candidate_pools,
    build_leave_one_out_training_ratings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--positive-threshold", type=float, default=4.0)
    parser.add_argument("--validation-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool-size", type=int, default=20)
    parser.add_argument("--retriever-history-window", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_domain(args.domain, args.data_root, args.output_dir)
    examples, dropped = build_clean_examples(
        data.ratings,
        positive_threshold=args.positive_threshold,
        validation_fraction=args.validation_fraction,
        split_seed=args.seed,
    )
    # Every user's final interaction is held out of training, whether or not
    # that user became an example, so no evaluated target is ever seen.
    training = build_leave_one_out_training_ratings(data.ratings)
    popularity = build_item_popularity(training)
    catalog = sorted(set(int(i) for i in data.ratings["item_id"].unique()) & set(data.item_titles))
    examples = [e for e in examples if e.target_item_id in data.item_titles]

    retriever = ItemKNNRetriever(
        training, catalog, popularity, history_window=args.retriever_history_window
    )
    matched = build_popularity_matched_pools(
        examples, popularity, catalog, pool_size=args.pool_size, seed=args.seed
    )
    retrieval = build_retrieval_pools(
        examples, retriever, popularity, pool_size=args.pool_size, seed=args.seed
    )
    by_session = {e.session_id: e for e in examples}
    top_popular = [
        BenchmarkPool(
            session_id=pool.session_id,
            split=by_session[pool.session_id].split,
            pool_type="top_popular",
            target_item_id=pool.target_item_id,
            candidate_item_ids=pool.candidate_item_ids,
            target_position=pool.target_position,
            target_popularity_rank=popularity_midrank(
                pool.candidate_item_ids, pool.target_item_id, popularity
            ),
        )
        for pool in build_candidate_pools(training, examples, candidate_pool_size=args.pool_size)
    ]

    out = args.output_dir
    prefix = data.file_prefix
    write_records(examples, out / f"{prefix}_clean_examples.jsonl")
    write_records(matched, out / f"{prefix}_pool_popularity_matched.jsonl")
    write_records(retrieval, out / f"{prefix}_pool_retrieval.jsonl")
    write_records(top_popular, out / f"{prefix}_pool_top_popular.jsonl")
    splits = {s: sum(e.split == s for e in examples) for s in ("validation", "test")}
    summary = {
        "domain": args.domain,
        "examples": len(examples),
        "splits": splits,
        "dropped_users": dropped,
        "target_policy": (
            f"final interaction per user, rated >= {args.positive_threshold}; users whose final "
            "timestamp equals the previous interaction's are excluded (order unknown)"
        ),
        "training_policy": "every user's final interaction removed (leave-one-out frame)",
        "catalog_items": len(catalog),
        "pools": {
            "popularity_matched": "19 negatives from the target's popularity neighbourhood; "
            "number less popular than the target drawn uniformly 0..19; seeded order",
            "retrieval": f"item-KNN cosine over last {args.retriever_history_window} history items, "
            "training data only; target inserted in place of the 20th item when not retrieved",
            "top_popular": "original design (19 most popular non-history items); diagnostic only",
        },
        "retrieval_recall_at_pool_size": sum(p.target_retrieved for p in retrieval) / len(retrieval),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
    }
    (out / f"{prefix}_benchmark_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
