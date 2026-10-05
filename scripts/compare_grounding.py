"""Compare the grounded strategy prompts with title only (Addendum v20).

Experiment 3 scores the eight Experiment 1 strategy prompts again, with the
co-purchase context (``context_collab_v2``) in place of the title-only
context. Each prompt sees the same sessions, candidates and candidate order
as in Experiment 1 (``run_ensemble.py --shuffle-key-context
context_title_v1``), so only the item context changes. On the test split,
for sessions where all sixteen members parsed:

Primary (Holm across the three):

1. agreement: each session's mean pairwise tau-b between the eight members
   on raw scores, grounded against title only (higher means less drift);
2. RWRA (as recorded) grounded against RWRA title only;
3. RWRA grounded against item-KNN alone, the ranker whose signal the
   grounding context carries.

Secondary (Holm across the two): the debiased ensemble grounded against
title only and against item-KNN. Per prompt (Holm across the eight): each
prompt grounded against the same prompt with title only.

Accuracy comparisons use reciprocal rank; all use the paired sign-flip test
and a paired bootstrap 95% CI. Each run's debiased ensemble is fitted on its
own validation sessions, as in evaluate_debiased.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from llm_session_reco.debiasing import (  # noqa: E402
    MemberAnswer,
    calibrated_scores,
    fit_slot_priors,
    mean_scores,
    member_answers,
)
from llm_session_reco.metrics import tie_aware_metrics  # noqa: E402
from llm_session_reco.statistics import holm_adjust, paired_bootstrap_ci, paired_sign_flip_test  # noqa: E402
from scripts.evaluate_debiased import read_jsonl  # noqa: E402

TITLE_CONTEXT = "context_title_v1"
GROUNDING_CONTEXT = "context_collab_v2"

Ranker = Callable[[Sequence[int], Sequence[int]], dict[int, float]]


def _by_prompt(record: dict, context: str) -> dict[str, MemberAnswer]:
    return {
        member.split(":")[0]: answer
        for member, answer in member_answers(record).items()
        if member.split(":")[1] == context
    }


def _agreement(answers: dict[str, MemberAnswer], pool: Sequence[int]) -> float:
    """Mean pairwise tau-b on raw scores, as evaluate_debiased.py reports it."""

    vectors = [[answers[p].item_scores[i] for i in pool] for p in sorted(answers)]
    taus = []
    for a in range(len(vectors)):
        for b in range(a + 1, len(vectors)):
            tau = kendalltau(vectors[a], vectors[b]).statistic
            taus.append(0.0 if tau != tau else float(tau))
    return float(np.mean(taus))


def _compare(values: dict[str, list[float]], a: str, b: str, metric: str, *, n_boot: int, seed: int) -> dict:
    ci = paired_bootstrap_ci(values[a], values[b], n_boot=n_boot, seed=seed)
    test = paired_sign_flip_test(values[a], values[b], seed=seed)
    return {
        "metric": metric,
        "mean_difference": ci["mean_difference"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "p_value": test["p_value"],
        "n_pairs": ci["n_pairs"],
    }


def _holm(family: dict) -> dict:
    for key, p in holm_adjust({k: v["p_value"] for k, v in family.items()}).items():
        family[key]["p_holm"] = p
    return family


def _priors(records: list[dict], split: dict[str, str], context: str) -> dict:
    return fit_slot_priors(
        a for r in records if split[r["session_id"]] == "validation"
        for a in _by_prompt(r, context).values()
    )


def evaluate(title_trials: list[dict], grounded_trials: list[dict], pools: list[dict],
             item_knn: Ranker, *, n_boot: int = 2000, seed: int = 0) -> dict:
    split = {p["session_id"]: p["split"] for p in pools}
    title = {r["session_id"]: r for r in title_trials if r["session_id"] in split}
    grounded = [r for r in grounded_trials if r["session_id"] in split]
    for record in grounded:
        other = title.get(record["session_id"])
        if other is None:
            raise ValueError(f"session {record['session_id']} has no title-only trial")
        if (other["candidate_item_ids"] != record["candidate_item_ids"]
                or other["target_item_id"] != record["target_item_id"]):
            raise ValueError(f"session {record['session_id']} was scored on different candidates")

    prompts = sorted({p for r in grounded for p in _by_prompt(r, GROUNDING_CONTEXT)})
    if not prompts:
        raise ValueError(f"no {GROUNDING_CONTEXT} members in the grounded trials")
    title_priors = _priors(list(title.values()), split, TITLE_CONTEXT)
    grounded_priors = _priors(grounded, split, GROUNDING_CONTEXT)

    values: dict[str, list[float]] = {name: [] for name in (
        "agreement_title_only", "agreement_grounded", "rwra_title_only", "rwra_grounded",
        "debiased_title_only", "debiased_grounded", "item_knn",
        *(f"{p}:title_only" for p in prompts), *(f"{p}:grounded" for p in prompts),
    )}
    order_mismatches = 0
    for record in grounded:
        session = record["session_id"]
        if split[session] != "test":
            continue
        g = _by_prompt(record, GROUNDING_CONTEXT)
        t = _by_prompt(title[session], TITLE_CONTEXT)
        rwra_g = (record.get("reliability_weighted") or {}).get("position_scores")
        rwra_t = (title[session].get("reliability_weighted") or {}).get("position_scores")
        if set(g) != set(prompts) or set(t) != set(prompts) or not rwra_g or not rwra_t:
            continue
        # The design holds the candidate order fixed; count any session where it was not.
        order_mismatches += any(g[p].slot_of_item != t[p].slot_of_item for p in prompts)
        pool, target = list(record["candidate_item_ids"]), record["target_item_id"]

        def rr(scores: dict[int, float]) -> float:
            return tie_aware_metrics(scores, target)["RR"]

        values["agreement_title_only"].append(_agreement(t, pool))
        values["agreement_grounded"].append(_agreement(g, pool))
        values["rwra_title_only"].append(rr(dict(zip(pool, rwra_t))))
        values["rwra_grounded"].append(rr(dict(zip(pool, rwra_g))))
        values["debiased_title_only"].append(
            rr(mean_scores([calibrated_scores(t[p], title_priors) for p in prompts])))
        values["debiased_grounded"].append(
            rr(mean_scores([calibrated_scores(g[p], grounded_priors) for p in prompts])))
        values["item_knn"].append(rr(item_knn(record["prefix_item_ids"], pool)))
        for p in prompts:
            values[f"{p}:title_only"].append(rr(t[p].item_scores))
            values[f"{p}:grounded"].append(rr(g[p].item_scores))

    def cmp(a: str, b: str, metric: str = "reciprocal_rank") -> dict:
        return _compare(values, a, b, metric, n_boot=n_boot, seed=seed)

    primary = _holm({
        "agreement_grounded_vs_title_only": cmp("agreement_grounded", "agreement_title_only", "tau_b"),
        "rwra_grounded_vs_title_only": cmp("rwra_grounded", "rwra_title_only"),
        "rwra_grounded_vs_item_knn": cmp("rwra_grounded", "item_knn"),
    })
    secondary = _holm({
        "debiased_grounded_vs_title_only": cmp("debiased_grounded", "debiased_title_only"),
        "debiased_grounded_vs_item_knn": cmp("debiased_grounded", "item_knn"),
    })
    per_prompt = _holm({f"{p}_grounded_vs_title_only": cmp(f"{p}:grounded", f"{p}:title_only")
                        for p in prompts})
    return {
        "prompts": prompts,
        "test_sessions_compared": len(values["item_knn"]),
        "sessions_with_a_different_candidate_order": order_mismatches,
        "mean": {name: float(np.mean(v)) for name, v in values.items()},
        "primary_comparisons": primary,
        "secondary_comparisons": secondary,
        "per_prompt_comparisons": per_prompt,
        "note": "pre-registered (Addendum v20); mean of agreement_* is tau-b, the rest MRR",
    }


def _item_knn_ranker(domain: str, data_root: Path, processed_dir: Path) -> Ranker:
    from llm_session_reco.baselines import build_item_popularity
    from llm_session_reco.benchmark import ItemKNNRetriever
    from llm_session_reco.domains import load_domain
    from llm_session_reco.reference_rankers import build_reference_rankers
    from llm_session_reco.session_dataset import build_leave_one_out_training_ratings

    data = load_domain(domain, data_root, processed_dir)
    training = build_leave_one_out_training_ratings(data.ratings)
    popularity = build_item_popularity(training)
    catalog = sorted(set(int(i) for i in data.ratings["item_id"].unique()) & set(data.item_titles))
    rankers = build_reference_rankers(
        training, popularity, data.item_titles, data.item_genres,
        ItemKNNRetriever(training, catalog, popularity),
    )
    return rankers["item_knn"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--title-trials", type=Path, required=True, help="Experiment 1 trials JSONL")
    parser.add_argument("--grounded-trials", type=Path, required=True, help="Experiment 3 trials JSONL")
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--domain", required=True, help="rebuilds item-KNN from training data only")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    summary = evaluate(
        read_jsonl(args.title_trials), read_jsonl(args.grounded_trials), read_jsonl(args.pools),
        _item_knn_ranker(args.domain, args.data_root, args.processed_dir),
        n_boot=args.n_boot, seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {args.output} ({summary['test_sessions_compared']} test sessions)")


if __name__ == "__main__":
    main()
