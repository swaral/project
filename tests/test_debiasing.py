"""Tests for the debiased ensemble and its validation-fitted evaluation."""

from __future__ import annotations

import random

import numpy as np

from llm_session_reco.debiasing import (
    MemberAnswer,
    calibrated_scores,
    fit_slot_priors,
    mean_scores,
    member_answers,
    zscore,
)
from llm_session_reco.metrics import tie_aware_metrics
from llm_session_reco.statistics import holm_adjust, paired_sign_flip_test
from scripts.evaluate_debiased import evaluate

MEMBERS = ("baseline_scores_v1", "wording_direct_v1", "wording_preference_v1", "wording_detailed_v1")


def test_tie_aware_metrics_average_over_tied_ranks():
    metrics = tie_aware_metrics({1: 5.0, 2: 5.0, 3: 1.0}, 1)
    assert metrics["RR"] == 0.75
    assert metrics["HR@1"] == 0.5
    constant = tie_aware_metrics({i: 0.0 for i in range(20)}, 0)
    assert abs(constant["RR"] - sum(1 / k for k in range(1, 21)) / 20) < 1e-12


def test_zscore_handles_constant_answers():
    assert zscore({1: 3.0, 2: 3.0}) == {1: 0.0, 2: 0.0}
    z = zscore({1: 1.0, 2: 3.0})
    assert z[2] > 0 > z[1]


def test_calibration_removes_a_learned_slot_preference():
    # A member that adds +2 to whatever sits in slot 0.
    rng = random.Random(0)
    validation = []
    for _ in range(400):
        items = list(range(20))
        rng.shuffle(items)
        scores = {item: rng.gauss(0, 1) + (2.0 if slot == 0 else 0.0) for slot, item in enumerate(items)}
        validation.append(MemberAnswer("m", scores, {item: slot for slot, item in enumerate(items)}))
    priors = fit_slot_priors(validation)
    assert priors["m"][0] > 1.0 and abs(priors["m"][5]) < 0.3

    # Item 1 is genuinely better; item 0 only wins raw because it sits in slot 0.
    raw = {item: -1.0 + 2.0 * (item - 2) / 17 for item in range(2, 20)}  # spread over [-1, 1]
    raw[0], raw[1] = 0.3 + 2.0, 1.2  # item 0: weak item plus the slot-0 boost
    answer = MemberAnswer("m", raw, {item: item for item in range(20)})
    assert max(raw, key=raw.get) == 0
    calibrated = calibrated_scores(answer, priors)
    assert max(calibrated, key=calibrated.get) == 1


def test_sign_flip_test_and_holm():
    a = [0.5] * 50 + [0.0] * 50
    b = [0.0] * 100
    assert paired_sign_flip_test(a, b)["p_value"] < 0.001
    assert paired_sign_flip_test(b, b)["p_value"] == 1.0
    assert holm_adjust({"x": 0.01, "y": 0.04, "z": 0.5}) == {"x": 0.03, "y": 0.08, "z": 0.5}


def _record(session_id, rng, *, bias):
    """Members see the target's quality weakly; each prefers slot 0 by ``bias``."""

    pool = list(range(100, 120))
    target = pool[rng.randrange(20)]
    trials = []
    for member in MEMBERS:
        presented = pool[:]
        rng.shuffle(presented)
        scores = [
            50 + rng.gauss(0, 5) + (4 if item == target else 0) + (bias if slot == 0 else 0)
            for slot, item in enumerate(presented)
        ]
        trials.append({
            "variant_id": member, "context_variant_id": "context_title_v1", "parse_success": True,
            "presented_candidate_item_ids": presented, "position_scores": scores,
        })
    return {"session_id": session_id, "candidate_item_ids": pool, "target_item_id": target, "member_trials": trials}


def test_evaluate_debiased_uses_validation_only_and_beats_naive_under_slot_bias():
    rng = random.Random(1)
    trials = [_record(f"user-{i}", rng, bias=25.0) for i in range(600)]
    pools = [
        {"session_id": r["session_id"], "split": "validation" if i < 200 else "test", "pool_type": "popularity_matched"}
        for i, r in enumerate(trials)
    ]
    summary = evaluate(trials, pools, n_boot=200)

    assert summary["validation_sessions"] == 200 and summary["test_sessions"] == 400
    assert summary["preselected_single_prompt"] in {f"{m}:context_title_v1" for m in MEMBERS}
    methods = summary["methods"]
    assert methods["debiased_ensemble"]["RR"] > methods["naive_mean"]["RR"]
    primary = summary["primary_comparisons"]["debiased_ensemble_vs_naive_mean"]
    assert primary["mean_difference"] > 0 and primary["p_holm"] < 0.05


def test_member_answers_map_presented_scores_to_items():
    record = {
        "candidate_item_ids": [1, 2, 3],
        "member_trials": [
            {"variant_id": "v", "context_variant_id": "c", "parse_success": True,
             "presented_candidate_item_ids": [3, 1, 2], "position_scores": [9, 1, 5]},
            {"variant_id": "w", "context_variant_id": "c", "parse_success": False},
        ],
    }
    answers = member_answers(record)
    assert list(answers) == ["v:c"]
    assert answers["v:c"].item_scores == {3: 9.0, 1: 1.0, 2: 5.0}
    assert answers["v:c"].slot_of_item[3] == 0
    assert mean_scores([{1: 1.0, 2: 3.0}, {1: 3.0, 2: 1.0}]) == {1: 2.0, 2: 2.0}
    assert np.isclose(sum(zscore({1: 1.0, 2: 2.0, 3: 6.0}).values()), 0.0)


def test_evaluate_debiased_can_exclude_a_member():
    rng = random.Random(2)
    trials = []
    for i in range(400):
        pool = list(range(100, 120))
        target = pool[rng.randrange(20)]
        record = {"session_id": f"user-{i}", "candidate_item_ids": pool, "target_item_id": target,
                  "member_trials": []}
        # One member sees the target clearly; the other three barely do.
        for context, signal in (("context_collab_v2", 15.0), ("context_content_v2", 1.0),
                                ("context_crowd_v2", 1.0), ("context_personal_v2", 1.0)):
            presented = pool[:]
            rng.shuffle(presented)
            record["member_trials"].append({
                "variant_id": "baseline_scores_v1", "context_variant_id": context, "parse_success": True,
                "presented_candidate_item_ids": presented,
                "position_scores": [50 + rng.gauss(0, 5) + (signal if item == target else 0) for item in presented],
            })
        trials.append(record)
    pools = [{"session_id": r["session_id"], "split": "validation" if i < 150 else "test",
              "pool_type": "popularity_matched"} for i, r in enumerate(trials)]

    full = evaluate(trials, pools, n_boot=100)
    without = evaluate(trials, pools, n_boot=100, exclude_members=["context_collab_v2"])

    assert full["preselected_single_prompt"] == "baseline_scores_v1:context_collab_v2"
    assert without["excluded_members"] == ["baseline_scores_v1:context_collab_v2"]
    assert "baseline_scores_v1:context_collab_v2" not in without["members"] + [without["preselected_single_prompt"]]
    # RWRA is recomputed from the remaining members, never read from the record.
    assert "rwra_recomputed" in without["methods"] and "rwra_recorded" not in without["methods"]
    assert without["methods"]["debiased_ensemble"]["RR"] < full["methods"]["debiased_ensemble"]["RR"]
    try:
        evaluate(trials, pools, n_boot=100, exclude_members=["context_colab_v2"])
    except ValueError as error:
        assert "matched no member" in str(error)
    else:
        raise AssertionError("a misspelled member must not be silently ignored")
