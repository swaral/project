"""Does the LLM add anything to non-LLM rankers? Validation-fitted hybrids.

Inputs: shuffled ensemble trials on a benchmark pool (run_ensemble.py
--shuffle-candidates), the pool file, and the domain (to rebuild the
training-only reference rankers). All weights are fitted on validation
sessions; all scores are tie-aware and reported on test sessions.

Rankers compared on identical candidates:

- llm: the debiased ensemble (z-score, validation slot priors, mean);
- item_knn, sequential_transitions, genre_overlap, title_overlap (and
  retrieval_order on retrieval pools);
- nonllm_blend: z-scored combination of those rankers, weights fitted on
  validation - the strongest non-LLM system available here;
- hybrid: nonllm_blend plus the LLM with a validation-fitted weight
  (weight 0 means validation said the LLM does not help);
- item_knn+llm and sequential+llm: the same with a single base ranker.

Primary, pre-registered comparison: hybrid vs nonllm_blend (paired sign-flip
test on mean reciprocal rank, bootstrap CI). Secondary (Holm-corrected
together): item_knn+llm vs item_knn, sequential+llm vs sequential.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from llm_session_reco.baselines import build_item_popularity
from llm_session_reco.benchmark import ItemKNNRetriever
from llm_session_reco.debiasing import calibrated_scores, fit_slot_priors, mean_scores, member_answers
from llm_session_reco.domains import DOMAINS, load_domain
from llm_session_reco.hybrid import blend, fit_llm_weight, fit_weights
from llm_session_reco.metrics import tie_aware_metrics
from llm_session_reco.reference_rankers import build_reference_rankers
from llm_session_reco.session_dataset import build_leave_one_out_training_ratings
from llm_session_reco.statistics import holm_adjust, paired_bootstrap_ci, paired_sign_flip_test

METRICS = ("RR", "HR@1", "HR@5", "NDCG@10")
BASE_RANKERS = ("item_knn", "sequential_transitions", "genre_overlap", "title_overlap")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def retrieval_order(pool: dict) -> dict[int, float]:
    order = pool["retrieved_item_ids"]
    return {i: -float(order.index(i) if i in order else len(order)) for i in pool["candidate_item_ids"]}


def compare(a: list[float], b: list[float], seed: int) -> dict:
    ci = paired_bootstrap_ci(a, b, n_boot=2000, seed=seed)
    test = paired_sign_flip_test(a, b, seed=seed)
    return {"mean_difference": ci["mean_difference"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            "p_value": test["p_value"], "n_pairs": ci["n_pairs"]}


def evaluate(trials: list[dict], pools: list[dict], rankers: dict, examples: dict, *, seed: int = 0) -> dict:
    pool_by_session = {p["session_id"]: p for p in pools}
    records = [r for r in trials if r["session_id"] in pool_by_session]
    answers = {r["session_id"]: member_answers(r) for r in records}
    members = sorted({m for a in answers.values() for m in a})
    records = [r for r in records if set(answers[r["session_id"]]) == set(members)]
    split = {r["session_id"]: pool_by_session[r["session_id"]]["split"] for r in records}
    validation = [r for r in records if split[r["session_id"]] == "validation"]
    test = [r for r in records if split[r["session_id"]] == "test"]
    priors = fit_slot_priors(a for r in validation for a in answers[r["session_id"]].values())

    pool_type = pools[0]["pool_type"]
    base_names = list(BASE_RANKERS) + (["retrieval_order"] if pool_type == "retrieval" else [])

    def components(record: dict) -> dict[str, dict[int, float]]:
        pool = pool_by_session[record["session_id"]]
        history = examples[record["session_id"]]["prefix_item_ids"]
        out = {
            name: (retrieval_order(pool) if name == "retrieval_order"
                   else rankers[name](history, pool["candidate_item_ids"]))
            for name in base_names
        }
        out["llm"] = mean_scores([calibrated_scores(a, priors) for a in answers[record["session_id"]].values()])
        return out

    val = [(components(r), r["target_item_id"]) for r in validation]
    tst = [(components(r), r["target_item_id"]) for r in test]

    blend_weights, blend_val = fit_weights([([c[n] for n in base_names], t) for c, t in val])
    fitted: dict[str, object] = {"nonllm_blend_weights": dict(zip(base_names, blend_weights)),
                                 "nonllm_blend_validation_mrr": blend_val}

    def nonllm(c: dict) -> dict[int, float]:
        return blend([c[n] for n in base_names], blend_weights)

    llm_weights = {}
    for name, base_fn in (("hybrid", nonllm),
                          ("item_knn+llm", lambda c: c["item_knn"]),
                          ("sequential+llm", lambda c: c["sequential_transitions"])):
        weight, _ = fit_llm_weight([(base_fn(c), c["llm"], t) for c, t in val])
        llm_weights[name] = (base_fn, weight)
    fitted["llm_weight"] = {name: w for name, (_, w) in llm_weights.items()}

    methods: dict[str, list[dict[str, float]]] = {}
    for c, t in tst:
        scores = {name: c[name] for name in base_names}
        scores["llm"] = c["llm"]
        scores["nonllm_blend"] = nonllm(c)
        for name, (base_fn, weight) in llm_weights.items():
            scores[name] = blend([base_fn(c), c["llm"]], (1.0, weight))
        for name, s in scores.items():
            methods.setdefault(name, []).append(tie_aware_metrics(s, t))

    rr = {name: [m["RR"] for m in rows] for name, rows in methods.items()}
    primary = {"hybrid_vs_nonllm_blend": compare(rr["hybrid"], rr["nonllm_blend"], seed)}
    primary["hybrid_vs_nonllm_blend"]["p_holm"] = primary["hybrid_vs_nonllm_blend"]["p_value"]
    secondary = {
        "item_knn+llm_vs_item_knn": compare(rr["item_knn+llm"], rr["item_knn"], seed),
        "sequential+llm_vs_sequential": compare(rr["sequential+llm"], rr["sequential_transitions"], seed),
        "llm_vs_nonllm_blend": compare(rr["llm"], rr["nonllm_blend"], seed),
    }
    for key, p in holm_adjust({k: v["p_value"] for k, v in secondary.items()}).items():
        secondary[key]["p_holm"] = p

    summary = {
        "pool_type": pool_type,
        "members": members,
        "validation_sessions": len(validation),
        "test_sessions": len(test),
        "fitted_on_validation": fitted,
        "methods": {name: {m: float(np.mean([row[m] for row in rows])) for m in METRICS}
                    for name, rows in methods.items()},
        "primary_comparisons": primary,
        "secondary_comparisons": secondary,
    }
    if pool_type == "retrieval":
        recall = float(np.mean([bool(p["target_retrieved"]) for p in pools if p["split"] == "test"]))
        summary["retrieval_recall_test"] = recall
        summary["end_to_end_methods"] = {
            name: {m: recall * v for m, v in metrics.items()} for name, metrics in summary["methods"].items()
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_domain(args.domain, args.data_root, args.processed_dir)
    training = build_leave_one_out_training_ratings(data.ratings)
    popularity = build_item_popularity(training)
    catalog = sorted(set(int(i) for i in data.ratings["item_id"].unique()) & set(data.item_titles))
    rankers = build_reference_rankers(
        training, popularity, data.item_titles, data.item_genres,
        ItemKNNRetriever(training, catalog, popularity),
    )
    examples = {e["session_id"]: e for e in read_jsonl(args.processed_dir / f"{data.file_prefix}_clean_examples.jsonl")}
    summary = evaluate(read_jsonl(args.trials), read_jsonl(args.pools), rankers, examples, seed=args.seed)
    output = args.output or args.trials.with_name(args.trials.stem.removesuffix("_trials") + "_hybrid.json")
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("fitted_on_validation", "primary_comparisons")}, indent=2))


if __name__ == "__main__":
    main()
