"""Tests for validation-fitted hybrid scoring."""

from llm_session_reco.hybrid import blend, fit_llm_weight, fit_weights


def test_blend_z_scores_components_before_weighting():
    # Different scales: without z-scoring the large-scale component would dominate.
    small = {1: 0.1, 2: 0.2, 3: 0.0}
    large = {1: 300.0, 2: 100.0, 3: 200.0}
    combined = blend([small, large], (1.0, 1.0))
    assert max(combined, key=combined.get) in (1, 2)


def test_fit_llm_weight_switches_off_an_unhelpful_llm():
    base = {1: 3.0, 2: 2.0, 3: 1.0}
    anti = {1: 1.0, 2: 2.0, 3: 3.0}  # ranks the target last
    sessions = [(base, anti, 1)] * 5
    weight, score = fit_llm_weight(sessions)
    assert weight == 0.0 and score == 1.0


def test_fit_llm_weight_uses_a_helpful_llm():
    base = {1: 1.0, 2: 2.0, 3: 0.0}
    helpful = {1: 5.0, 2: 0.0, 3: 0.0}
    weight, score = fit_llm_weight([(base, helpful, 1)] * 5)
    assert weight > 0 and score == 1.0


def test_fit_weights_prefers_the_useful_component():
    good = {1: 2.0, 2: 1.0, 3: 0.0}
    bad = {1: 0.0, 2: 1.0, 3: 2.0}
    weights, score = fit_weights([([good, bad], 1)] * 4)
    assert score == 1.0 and weights[0] > 0 and weights[1] == 0.0
