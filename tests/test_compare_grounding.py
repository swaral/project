import random

import pytest

from scripts.compare_grounding import GROUNDING_CONTEXT, TITLE_CONTEXT, evaluate

PROMPTS = ("baseline_scores_v1", "next_step_v2", "skip_risk_v2")


def _trial(variant, context, presented, target, signal, rng):
    return {"variant_id": variant, "context_variant_id": context, "parse_success": True,
            "presented_candidate_item_ids": presented,
            "position_scores": [50 + rng.gauss(0, 5) + (signal if item == target else 0) for item in presented]}


def _data(sessions=300, seed=5, grounded_signal=20.0):
    rng = random.Random(seed)
    title, grounded, pools = [], [], []
    for i in range(sessions):
        pool = list(range(100, 120))
        target = pool[rng.randrange(20)]
        base = {"session_id": f"user-{i}", "candidate_item_ids": pool, "target_item_id": target,
                "prefix_item_ids": [1, 2, 3]}
        orders = {p: rng.sample(pool, len(pool)) for p in PROMPTS}
        # Title-only members see nothing; grounded members see the target.
        title.append({**base, "member_trials": [_trial(p, TITLE_CONTEXT, orders[p], target, 0.0, rng)
                                                for p in PROMPTS],
                      "reliability_weighted": {"position_scores": [rng.random() for _ in pool]}})
        grounded.append({**base, "member_trials": [_trial(p, GROUNDING_CONTEXT, orders[p], target,
                                                          grounded_signal, rng) for p in PROMPTS],
                         "reliability_weighted": {"position_scores": [float(item == target) for item in pool]}})
        pools.append({"session_id": base["session_id"], "split": "validation" if i < 100 else "test"})
    return title, grounded, pools


def _blind_knn(prefix, candidates):
    return {item: 0.0 for item in candidates}


def test_grounding_that_sees_the_target_raises_agreement_and_accuracy():
    title, grounded, pools = _data()
    summary = evaluate(title, grounded, pools, _blind_knn, n_boot=200)
    assert summary["test_sessions_compared"] == 200
    assert summary["sessions_with_a_different_candidate_order"] == 0
    primary = summary["primary_comparisons"]
    assert set(primary) == {"agreement_grounded_vs_title_only", "rwra_grounded_vs_title_only",
                            "rwra_grounded_vs_item_knn"}
    assert primary["agreement_grounded_vs_title_only"]["mean_difference"] > 0
    assert primary["agreement_grounded_vs_title_only"]["p_holm"] < 0.01
    assert summary["mean"]["rwra_grounded"] == 1.0
    assert primary["rwra_grounded_vs_item_knn"]["p_holm"] < 0.01
    assert set(summary["per_prompt_comparisons"]) == {f"{p}_grounded_vs_title_only" for p in PROMPTS}
    assert all(c["mean_difference"] > 0.3 for c in summary["per_prompt_comparisons"].values())


def test_item_knn_that_finds_the_target_is_not_beaten():
    title, grounded, pools = _data(grounded_signal=0.0)
    # A stand-in item-KNN that always ranks the target first (read from the prefix).
    for record in grounded:
        record["prefix_item_ids"] = [record["target_item_id"]]
    summary = evaluate(title, grounded, pools,
                       lambda prefix, candidates: {i: float(i == prefix[0]) for i in candidates}, n_boot=200)
    assert summary["mean"]["item_knn"] == 1.0
    assert summary["secondary_comparisons"]["debiased_grounded_vs_item_knn"]["mean_difference"] < 0


def test_a_changed_candidate_order_is_counted():
    title, grounded, pools = _data(sessions=150)
    trial = grounded[120]["member_trials"][0]
    trial["presented_candidate_item_ids"] = list(reversed(trial["presented_candidate_item_ids"]))
    assert evaluate(title, grounded, pools, _blind_knn, n_boot=50)["sessions_with_a_different_candidate_order"] == 1


def test_sessions_scored_on_different_candidates_are_rejected():
    title, grounded, pools = _data(sessions=120)
    title[5]["candidate_item_ids"] = list(reversed(title[5]["candidate_item_ids"]))
    with pytest.raises(ValueError, match="different candidates"):
        evaluate(title, grounded, pools, _blind_knn, n_boot=50)
