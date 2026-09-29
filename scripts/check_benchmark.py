"""Shortcut checks and non-LLM baselines for every benchmark pool of a domain.

Run after scripts/build_benchmark.py and before spending GPU time. For each
pool type it reports, on test-split sessions:

- how the target's popularity rank inside its pool is distributed (uniform
  on a fair pool) and whether popularity, inverse popularity or newest-first
  (recency) beats random.
  On popularity-matched pools a flag means a construction shortcut. On the
  retrieval subset it reflects real popularity bias in what the retriever
  finds, so popularity is then a baseline to beat, not an artifact;
- tie-aware MRR / HR@1 / HR@5 / NDCG@10 for every reference ranker;
- for retrieval pools: the retriever's recall; reranking quality on the
  sessions where the retriever found the target (no insertion); end-to-end
  scores, where an unretrieved target is a miss; and, as a diagnostic only,
  scores with the missed target inserted. Insertion is not used for headline
  numbers: the retriever favours popular items, so an inserted target is
  usually the least popular candidate and inverse popularity finds it.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from scipy.stats import chisquare

from llm_session_reco.baselines import build_item_popularity
from llm_session_reco.benchmark import ItemKNNRetriever
from llm_session_reco.domains import DOMAINS, load_domain
from llm_session_reco.metrics import tie_aware_metrics
from llm_session_reco.reference_rankers import build_reference_rankers
from llm_session_reco.session_dataset import build_leave_one_out_training_ratings

POOL_TYPES = ("top_popular", "random", "popularity_matched", "attribute_matched", "retrieval")
METRICS = ("RR", "HR@1", "HR@5", "NDCG@10")
MISS = {metric: 0.0 for metric in METRICS}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(values: list[dict[str, float]], seed: int = 0) -> dict[str, object]:
    out: dict[str, object] = {"n": len(values)}
    for metric in METRICS:
        column = np.array([v[metric] for v in values])
        idx = np.random.default_rng(seed).integers(0, len(column), size=(1000, len(column)))
        boot = column[idx].mean(axis=1)
        out[metric] = round(float(column.mean()), 4)
        if metric == "RR":
            out["RR_ci"] = [round(float(np.quantile(boot, 0.025)), 4), round(float(np.quantile(boot, 0.975)), 4)]
    return out


def retrieval_order_scores(pool: dict) -> dict[int, float]:
    """The retriever's own order; an inserted (unretrieved) target ranks last."""

    order = pool["retrieved_item_ids"]
    return {
        item: -float(order.index(item) if item in order else len(order))
        for item in pool["candidate_item_ids"]
    }


def shortcut_flag(result: dict, random_rr: float, margin: float = 0.02) -> bool:
    """True when the whole MRR interval is more than ``margin`` from random."""

    low, high = result["RR_ci"]
    return low > random_rr + margin or high < random_rr - margin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", default="test", choices=("test", "validation"))
    parser.add_argument("--max-sessions", type=int, default=3000, help="seeded sample per pool")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_domain(args.domain, args.data_root, args.processed_dir)
    training = build_leave_one_out_training_ratings(data.ratings)
    popularity = build_item_popularity(training)
    catalog = sorted(set(int(i) for i in data.ratings["item_id"].unique()) & set(data.item_titles))
    retriever = ItemKNNRetriever(training, catalog, popularity)
    rankers = build_reference_rankers(training, popularity, data.item_titles, data.item_genres, retriever)
    prefix = data.file_prefix
    examples = {
        e["session_id"]: e for e in read_jsonl(args.processed_dir / f"{prefix}_clean_examples.jsonl")
    }

    report: dict[str, object] = {"domain": args.domain, "split": args.split, "pools": {}}
    for pool_type in POOL_TYPES:
        pools = [p for p in read_jsonl(args.processed_dir / f"{prefix}_pool_{pool_type}.jsonl") if p["split"] == args.split]
        pools.sort(key=lambda p: p["session_id"])
        if len(pools) > args.max_sessions:
            pools = sorted(random.Random(args.seed).sample(pools, args.max_sessions), key=lambda p: p["session_id"])
        pool_size = len(pools[0]["candidate_item_ids"])
        random_rr = sum(1 / k for k in range(1, pool_size + 1)) / pool_size

        # Mid-ranks (ties averaged) binned into quintiles; uniform on a fair pool.
        ranks = np.array([p["target_popularity_rank"] for p in pools])
        quintiles = np.minimum(((ranks - 1) / pool_size * 5).astype(int), 4)
        entry: dict[str, object] = {
            "sessions": len(pools),
            "random_RR": round(random_rr, 4),
            "target_least_popular_share": round(float(np.mean(ranks == 1)), 4),
            "target_popularity_midrank_mean": round(float(ranks.mean()), 2),
            "target_popularity_midrank_expected": (pool_size + 1) / 2,
            "target_popularity_quintile_shares": [round(float(x), 3) for x in np.bincount(quintiles, minlength=5) / len(ranks)],
            "target_popularity_uniformity_p": float(chisquare(np.bincount(quintiles, minlength=5)).pvalue),
            "rankers": {},
        }
        names = list(rankers) + (["retrieval_order"] if pool_type == "retrieval" else [])
        for name in names:
            conditional, end_to_end, retrieved_only = [], [], []
            for pool in pools:
                history = examples[pool["session_id"]]["prefix_item_ids"]
                if name == "retrieval_order":
                    scores = retrieval_order_scores(pool)
                else:
                    scores = rankers[name](history, pool["candidate_item_ids"])
                metrics = tie_aware_metrics(scores, pool["target_item_id"])
                conditional.append(metrics)
                if pool_type == "retrieval":
                    end_to_end.append(metrics if pool["target_retrieved"] else MISS)
                    if pool["target_retrieved"]:
                        retrieved_only.append(metrics)
            if pool_type == "retrieval":
                result = {
                    "retrieved_subset": summarize(retrieved_only, args.seed),
                    "end_to_end": summarize(end_to_end, args.seed),
                    "inserted_diagnostic": summarize(conditional, args.seed),
                }
            else:
                result = {"all_sessions": summarize(conditional, args.seed)}
            entry["rankers"][name] = result

        if pool_type == "retrieval":
            entry["retrieval_recall_at_pool_size"] = round(float(np.mean([p["target_retrieved"] for p in pools])), 4)
        headline = "retrieved_subset" if pool_type == "retrieval" else "all_sessions"
        entry["headline_metrics"] = headline
        entry["shortcut_flags"] = {
            name: shortcut_flag(entry["rankers"][name][headline], random_rr)
            for name in ("popularity", "inverse_popularity", "newest_first")
        }
        if pool_type == "retrieval":
            subset = [p for p in pools if p["target_retrieved"]]
            entry["retrieved_subset_target_least_popular_share"] = round(
                float(np.mean([p["target_popularity_rank"] == 1 for p in subset])), 4
            )
            entry["inserted_diagnostic_shortcut_flags"] = {
                name: shortcut_flag(entry["rankers"][name]["inserted_diagnostic"], random_rr)
                for name in ("popularity", "inverse_popularity")
            }
        report["pools"][pool_type] = entry

    output = args.output or Path("results/benchmark_checks") / f"{args.domain}_{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))

    print(f"\n== {args.domain} ({args.split} split) -> {output}")
    for pool_type, entry in report["pools"].items():
        print(f"\n[{pool_type}] sessions={entry['sessions']} random MRR={entry['random_RR']} "
              f"target strictly least popular={entry['target_least_popular_share']:.0%} "
              f"popularity mid-rank mean={entry['target_popularity_midrank_mean']} "
              f"(expected {entry['target_popularity_midrank_expected']}), quintiles={entry['target_popularity_quintile_shares']} "
              f"shortcut flags={entry['shortcut_flags']}"
              + (f" retrieval recall@20={entry['retrieval_recall_at_pool_size']:.1%}" if "retrieval_recall_at_pool_size" in entry else ""))
        if "retrieved_subset_target_least_popular_share" in entry:
            print(f"   retrieved subset: n={entry['rankers']['random']['retrieved_subset']['n']}, "
                  f"target least popular={entry['retrieved_subset_target_least_popular_share']:.0%}; "
                  f"inserted-diagnostic shortcut flags={entry['inserted_diagnostic_shortcut_flags']}")
        for name, result in entry["rankers"].items():
            c = result[entry["headline_metrics"]]
            line = f"   {name:24s} MRR {c['RR']:.3f} {c['RR_ci']}  HR@1 {c['HR@1']:.3f}  HR@5 {c['HR@5']:.3f}  NDCG@10 {c['NDCG@10']:.3f}"
            if "end_to_end" in result:
                e, i = result["end_to_end"], result["inserted_diagnostic"]
                line += f"  | end-to-end MRR {e['RR']:.3f} NDCG@10 {e['NDCG@10']:.3f} | inserted MRR {i['RR']:.3f}"
            print(line)


if __name__ == "__main__":
    main()
