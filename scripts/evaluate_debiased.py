"""Evaluate the debiased ensemble with validation-fitted choices, on test sessions.

Inputs are trials from ``run_ensemble.py --shuffle-candidates`` on a
benchmark pool, plus that pool file (for the validation/test split and, on
retrieval pools, the retriever's recall). Everything data-driven is chosen on
validation sessions only:

- the slot priors used for calibration;
- the preselected single prompt (the member with the best validation MRR).

On test sessions it reports tie-aware MRR / HR@1 / HR@5 / NDCG@10 for:
preselected single prompt, naive mean of raw scores, mean of z-scores, the
debiased ensemble (z-scores minus slot priors, then mean) and the recorded
RWRA ranking. Primary, pre-registered comparisons (Holm-corrected together):
debiased vs preselected single prompt, and debiased vs naive mean. The test
is a paired sign-flip permutation test on the mean, with a bootstrap CI.

For retrieval pools the trials should cover only sessions where the retriever
found the target; end-to-end numbers are recall x reranking quality, so an
unretrieved target counts as a miss.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from llm_session_reco.debiasing import (
    calibrated_scores,
    fit_slot_priors,
    mean_scores,
    member_answers,
    zscore,
)
from llm_session_reco.metrics import tie_aware_metrics
from llm_session_reco.statistics import holm_adjust, paired_bootstrap_ci, paired_sign_flip_test

METRICS = ("RR", "HR@1", "HR@5", "NDCG@10")
PRIMARY = (("debiased_ensemble", "single_prompt_preselected"), ("debiased_ensemble", "naive_mean"))
SECONDARY = (
    ("zscore_mean", "naive_mean"),
    ("naive_mean", "single_prompt_preselected"),
    ("rwra_recorded", "naive_mean"),
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compare(values: dict[str, list[float]], a: str, b: str, *, n_boot: int, seed: int) -> dict:
    ci = paired_bootstrap_ci(values[a], values[b], n_boot=n_boot, seed=seed)
    test = paired_sign_flip_test(values[a], values[b], seed=seed)
    return {
        "metric": "reciprocal_rank",
        "mean_difference": ci["mean_difference"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "p_value": test["p_value"],
        "n_pairs": ci["n_pairs"],
    }


def evaluate(trials: list[dict], pools: list[dict], *, n_boot: int = 2000, seed: int = 0) -> dict:
    pool_by_session = {p["session_id"]: p for p in pools}
    records = [r for r in trials if r["session_id"] in pool_by_session]
    answers = {r["session_id"]: member_answers(r) for r in records}
    members = sorted({m for a in answers.values() for m in a})

    validation = [r for r in records if pool_by_session[r["session_id"]]["split"] == "validation"]
    test = [r for r in records if pool_by_session[r["session_id"]]["split"] == "test"]
    if not validation or not test:
        raise ValueError("need both validation and test sessions in the trials")

    priors = fit_slot_priors(a for r in validation for a in answers[r["session_id"]].values())
    validation_mrr = {
        m: float(np.mean([
            tie_aware_metrics(answers[r["session_id"]][m].item_scores, r["target_item_id"])["RR"]
            for r in validation if m in answers[r["session_id"]]
        ]))
        for m in members
    }
    preselected = max(members, key=lambda m: (validation_mrr[m], m))

    complete = [r for r in test if set(answers[r["session_id"]]) == set(members)]
    per_method: dict[str, list[dict[str, float]]] = {name: [] for name in (
        "single_prompt_preselected", "naive_mean", "zscore_mean", "debiased_ensemble", "rwra_recorded",
    )}
    for record in complete:
        session_answers = answers[record["session_id"]]
        target = record["target_item_id"]
        ordered = [session_answers[m] for m in members]
        scores = {
            "single_prompt_preselected": session_answers[preselected].item_scores,
            "naive_mean": mean_scores([a.item_scores for a in ordered]),
            "zscore_mean": mean_scores([zscore(a.item_scores) for a in ordered]),
            "debiased_ensemble": mean_scores([calibrated_scores(a, priors) for a in ordered]),
        }
        rwra = record.get("reliability_weighted") or {}
        if rwra.get("position_scores"):
            scores["rwra_recorded"] = dict(zip(record["candidate_item_ids"], rwra["position_scores"]))
        for name, item_scores in scores.items():
            per_method[name].append(tie_aware_metrics(item_scores, target))
    per_method = {k: v for k, v in per_method.items() if len(v) == len(complete)}

    rr = {name: [m["RR"] for m in rows] for name, rows in per_method.items()}
    primary = {f"{a}_vs_{b}": compare(rr, a, b, n_boot=n_boot, seed=seed) for a, b in PRIMARY}
    secondary = {
        f"{a}_vs_{b}": compare(rr, a, b, n_boot=n_boot, seed=seed)
        for a, b in SECONDARY if a in rr and b in rr
    }
    for family in (primary, secondary):
        adjusted = holm_adjust({k: v["p_value"] for k, v in family.items()})
        for key, value in family.items():
            value["p_holm"] = adjusted[key]

    pool_type = pools[0]["pool_type"]
    summary = {
        "pool_type": pool_type,
        "members": members,
        "validation_sessions": len(validation),
        "test_sessions": len(test),
        "test_sessions_all_members_parsed": len(complete),
        "preselected_single_prompt": preselected,
        "validation_mrr_by_member": validation_mrr,
        "slot_prior_spread_by_member": {m: float(np.ptp(list(p.values()))) for m, p in priors.items()},
        "methods": {
            name: {metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS}
            for name, rows in per_method.items()
        },
        "primary_comparisons": primary,
        "secondary_comparisons": secondary,
    }
    if pool_type == "retrieval":
        test_pools = [p for p in pools if p["split"] == "test"]
        recall = float(np.mean([bool(p["target_retrieved"]) for p in test_pools]))
        summary["retrieval_recall_test"] = recall
        summary["end_to_end_methods"] = {
            name: {metric: recall * value for metric, value in metrics.items()}
            for name, metrics in summary["methods"].items()
        }
        summary["note"] = (
            "methods = reranking quality on test sessions where the retriever found the target; "
            "end_to_end_methods = recall x that quality (unretrieved targets are misses)"
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate(read_jsonl(args.trials), read_jsonl(args.pools), n_boot=args.n_boot, seed=args.seed)
    output = args.output or args.trials.with_name(args.trials.stem.removesuffix("_trials") + "_debiased.json")
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
