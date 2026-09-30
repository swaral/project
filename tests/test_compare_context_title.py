import random

import pytest

from scripts.compare_context_title import TITLE_MEMBER, evaluate


def _trial(variant, context, pool, target, signal, rng):
    presented = pool[:]
    rng.shuffle(presented)
    return {"variant_id": variant, "context_variant_id": context, "parse_success": True,
            "presented_candidate_item_ids": presented,
            "position_scores": [50 + rng.gauss(0, 5) + (signal if item == target else 0) for item in presented]}


def _data(sessions=300, seed=3):
    rng = random.Random(seed)
    title, contexts, pools = [], [], []
    for i in range(sessions):
        pool = list(range(100, 120))
        target = pool[rng.randrange(20)]
        base = {"session_id": f"user-{i}", "candidate_item_ids": pool, "target_item_id": target}
        variant, context = TITLE_MEMBER.split(":")
        title.append({**base, "member_trials": [_trial(variant, context, pool, target, 0.0, rng)]})
        # One context sees the target clearly, the other not at all.
        trials = [_trial("baseline_scores_v1", "context_content_v2", pool, target, 20.0, rng),
                  _trial("baseline_scores_v1", "context_format_v2", pool, target, 0.0, rng)]
        contexts.append({**base, "member_trials": trials,
                         "reliability_weighted": {"position_scores": [float(item == target) for item in pool]}})
        pools.append({"session_id": base["session_id"], "split": "validation" if i < 100 else "test"})
    return title, contexts, pools


def test_context_that_sees_the_target_beats_title_only():
    title, contexts, pools = _data()
    summary = evaluate(title, contexts, pools, n_boot=200)
    assert summary["test_sessions_compared"] == 200
    content = summary["contexts_vs_title_only"]["baseline_scores_v1:context_content_v2_vs_title_only"]
    blind = summary["contexts_vs_title_only"]["baseline_scores_v1:context_format_v2_vs_title_only"]
    assert content["mean_difference"] > 0.3 and content["p_holm"] < 0.01
    assert blind["p_holm"] > 0.05
    assert summary["mrr"]["rwra_recorded"] == 1.0
    assert set(summary["ensembles_vs_title_only"]) == {"rwra_recorded_vs_title_only",
                                                       "debiased_ensemble_vs_title_only"}


def test_sessions_scored_on_different_candidates_are_rejected():
    title, contexts, pools = _data(sessions=120)
    title[5]["candidate_item_ids"] = list(reversed(title[5]["candidate_item_ids"]))
    with pytest.raises(ValueError, match="different candidates"):
        evaluate(title, contexts, pools, n_boot=50)
