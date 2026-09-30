"""Compare the Experiment 2b semantic contexts with title only (Addendum v19).

Experiment 1 includes the member ``baseline_scores_v1:context_title_v1``: the
same scoring prompt as the Experiment 2b members, with the title as the only
item context. Where both experiments scored the same sessions with the same
candidates, this reports on the test split, as exploratory comparisons:

1. each context alone against title only (Holm across the contexts);
2. RWRA (as recorded) and the debiased ensemble against title only (Holm
   across the two).

The debiased ensemble is fitted on the validation sessions exactly as in
evaluate_debiased.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from llm_session_reco.debiasing import calibrated_scores, fit_slot_priors, mean_scores, member_answers  # noqa: E402
from llm_session_reco.metrics import tie_aware_metrics  # noqa: E402
from llm_session_reco.statistics import holm_adjust  # noqa: E402
from scripts.evaluate_debiased import compare, read_jsonl  # noqa: E402

TITLE_MEMBER = "baseline_scores_v1:context_title_v1"


def _holm(family: dict) -> dict:
    adjusted = holm_adjust({k: v["p_value"] for k, v in family.items()})
    for key, value in family.items():
        value["p_holm"] = adjusted[key]
    return family


def evaluate(title_trials: list[dict], context_trials: list[dict], pools: list[dict], *,
             n_boot: int = 2000, seed: int = 0) -> dict:
    split = {p["session_id"]: p["split"] for p in pools}
    title = {r["session_id"]: r for r in title_trials}
    records = [r for r in context_trials if r["session_id"] in split]
    for record in records:
        other = title.get(record["session_id"])
        if other is None:
            raise ValueError(f"session {record['session_id']} has no title-only trial")
        if (other["candidate_item_ids"] != record["candidate_item_ids"]
                or other["target_item_id"] != record["target_item_id"]):
            raise ValueError(f"session {record['session_id']} was scored on different candidates")
    answers = {r["session_id"]: member_answers(r) for r in records}
    contexts = sorted({m for a in answers.values() for m in a})
    title_answer = {s: member_answers(title[s]).get(TITLE_MEMBER) for s in answers}

    validation = [r for r in records if split[r["session_id"]] == "validation"]
    priors = fit_slot_priors(a for r in validation for a in answers[r["session_id"]].values())
    complete = [
        r for r in records
        if split[r["session_id"]] == "test" and set(answers[r["session_id"]]) == set(contexts)
        and title_answer[r["session_id"]] is not None
        and (r.get("reliability_weighted") or {}).get("position_scores")
    ]
    rr: dict[str, list[float]] = {"title_only": [], "rwra_recorded": [], "debiased_ensemble": [],
                                  **{c: [] for c in contexts}}
    for record in complete:
        session, target = record["session_id"], record["target_item_id"]
        rr["title_only"].append(tie_aware_metrics(title_answer[session].item_scores, target)["RR"])
        for context in contexts:
            rr[context].append(tie_aware_metrics(answers[session][context].item_scores, target)["RR"])
        rwra = dict(zip(record["candidate_item_ids"], record["reliability_weighted"]["position_scores"]))
        rr["rwra_recorded"].append(tie_aware_metrics(rwra, target)["RR"])
        debiased = mean_scores([calibrated_scores(answers[session][c], priors) for c in contexts])
        rr["debiased_ensemble"].append(tie_aware_metrics(debiased, target)["RR"])

    contexts_vs_title = _holm({f"{c}_vs_title_only": compare(rr, c, "title_only", n_boot=n_boot, seed=seed)
                               for c in contexts})
    ensembles_vs_title = _holm({f"{e}_vs_title_only": compare(rr, e, "title_only", n_boot=n_boot, seed=seed)
                                for e in ("rwra_recorded", "debiased_ensemble")})
    return {
        "title_member": TITLE_MEMBER,
        "contexts": contexts,
        "test_sessions_compared": len(complete),
        "mrr": {name: float(np.mean(values)) for name, values in rr.items()},
        "contexts_vs_title_only": contexts_vs_title,
        "ensembles_vs_title_only": ensembles_vs_title,
        "note": "exploratory (Addendum v19)",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--title-trials", type=Path, required=True, help="Experiment 1 trials JSONL")
    parser.add_argument("--context-trials", type=Path, required=True, help="Experiment 2b trials JSONL")
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    summary = evaluate(read_jsonl(args.title_trials), read_jsonl(args.context_trials), read_jsonl(args.pools),
                       n_boot=args.n_boot, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {args.output} ({summary['test_sessions_compared']} test sessions)")


if __name__ == "__main__":
    main()
