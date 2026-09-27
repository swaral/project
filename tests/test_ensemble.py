import pytest

from llm_session_reco.ensemble import (
    EnsembleMember,
    aggregate_long_tail_weighted,
    aggregate_naive_mean,
    aggregate_reliability_weighted,
    combine_ensemble,
    compute_reliability_weights,
    long_tail_coefficient,
    long_tail_weight,
)


def test_aggregate_naive_mean_is_elementwise_average():
    scores = aggregate_naive_mean([(10.0, 20.0), (30.0, 40.0)])
    assert scores == pytest.approx((20.0, 30.0))


def test_aggregate_naive_mean_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same number of candidate positions"):
        aggregate_naive_mean([(1.0, 2.0), (1.0, 2.0, 3.0)])


def test_reliability_weights_are_equal_for_a_single_member():
    assert compute_reliability_weights([[1, 2, 3]]) == (1.0,)


def test_reliability_weights_are_equal_for_agreeing_members():
    weights = compute_reliability_weights([[1, 2, 3], [1, 2, 3]])
    assert weights[0] == pytest.approx(weights[1])
    assert sum(weights) == pytest.approx(1.0)


def test_reliability_weights_downweight_a_divergent_member():
    # A and B agree; C is the exact reverse of both.
    weights = compute_reliability_weights([[1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]])
    weight_a, weight_b, weight_c = weights
    assert weight_a == pytest.approx(weight_b)
    assert weight_c < weight_a
    assert weight_c == pytest.approx(0.0)
    assert sum(weights) == pytest.approx(1.0)


def test_aggregate_reliability_weighted_matches_naive_when_weights_are_uniform():
    scores = [(10.0, 20.0), (30.0, 40.0)]
    uniform = aggregate_reliability_weighted(scores, (0.5, 0.5))
    assert uniform == pytest.approx(aggregate_naive_mean(scores))


def test_aggregate_reliability_weighted_rejects_mismatched_weight_count():
    with pytest.raises(ValueError, match="same length"):
        aggregate_reliability_weighted([(1.0,), (2.0,)], (1.0,))


CANDIDATES = (10, 20, 30, 40)


def _member(variant_id: str, scores: tuple[float, ...], *, parse_success: bool = True) -> EnsembleMember:
    return EnsembleMember(
        variant_id=variant_id,
        context_variant_id="context_title_v1",
        parse_success=parse_success,
        position_scores=scores,
    )


def test_combine_ensemble_reports_failure_below_min_valid_members():
    members = [
        _member("a", (100.0, 50.0, 25.0, 10.0)),
        _member("b", (), parse_success=False),
    ]
    result = combine_ensemble(members, CANDIDATES, target_item_id=10)
    assert result.valid_member_count == 1
    assert result.naive is None
    assert result.reliability_weighted is None
    assert result.failure_reason is not None
    assert result.drift["mean_kendall_tau"] is None


def test_combine_ensemble_excludes_parse_failures_but_still_combines_the_rest():
    members = [
        _member("a", (100.0, 50.0, 25.0, 10.0)),
        _member("b", (100.0, 50.0, 25.0, 10.0)),
        _member("broken", (), parse_success=False),
    ]
    result = combine_ensemble(members, CANDIDATES, target_item_id=10)
    assert result.valid_member_count == 2
    assert result.naive is not None
    assert result.naive.target_rank == 1


def test_rwra_ignores_a_divergent_member_more_than_naive_mean_does():
    # a and b agree the target (item 10) is clearly best; c is the reverse.
    members = [
        _member("a", (100.0, 50.0, 25.0, 10.0)),
        _member("b", (100.0, 50.0, 25.0, 10.0)),
        _member("c", (10.0, 25.0, 50.0, 100.0)),
    ]
    result = combine_ensemble(members, CANDIDATES, target_item_id=10)

    assert result.valid_member_count == 3
    # Naive mean is diluted by the outlier: item 20 (naive score ~41.7)
    # nearly overtakes item 40 (naive score 40).
    assert result.naive.position_scores == pytest.approx(
        (70.0, 41.666666, 33.333333, 40.0), rel=1e-4
    )
    # RWRA fully discounts the divergent member (weight ~0), reproducing the
    # agreeing members' scores exactly.
    assert result.reliability_weighted.position_scores == pytest.approx(
        (100.0, 50.0, 25.0, 10.0)
    )
    assert result.naive.target_rank == 1
    assert result.reliability_weighted.target_rank == 1
    assert result.member_weights[2] == pytest.approx(0.0)


def test_combine_ensemble_rejects_missing_target():
    members = [
        _member("a", (100.0, 50.0, 25.0, 10.0)),
        _member("b", (100.0, 50.0, 25.0, 10.0)),
    ]
    with pytest.raises(ValueError, match="not present in candidate_item_ids"):
        combine_ensemble(members, CANDIDATES, target_item_id=999)


# --- Option A: long-tail-aware weighting (adapted from Llama4Rec) ---


def test_long_tail_coefficient_is_log_n_plus_one():
    assert long_tail_coefficient(0) == pytest.approx(0.0)
    import math

    assert long_tail_coefficient(9) == pytest.approx(math.log(10))


def test_long_tail_coefficient_rejects_negative_length():
    with pytest.raises(ValueError, match="cannot be negative"):
        long_tail_coefficient(-1)


def test_long_tail_weight_is_beta1_at_the_shortest_prefix():
    weight = long_tail_weight(1, min_prefix_length=1, max_prefix_length=49, beta1=0.7, beta2=0.3)
    assert weight == pytest.approx(0.7)


def test_long_tail_weight_floors_at_beta2_times_beta1_for_the_longest_prefix():
    weight = long_tail_weight(49, min_prefix_length=1, max_prefix_length=49, beta1=0.7, beta2=0.3)
    assert weight == pytest.approx(0.3 * 0.7)


def test_long_tail_weight_decreases_as_prefix_grows():
    short = long_tail_weight(2, min_prefix_length=1, max_prefix_length=49)
    long = long_tail_weight(30, min_prefix_length=1, max_prefix_length=49)
    assert short > long


def test_long_tail_weight_rejects_invalid_bounds():
    with pytest.raises(ValueError, match="max_prefix_length must be greater"):
        long_tail_weight(5, min_prefix_length=10, max_prefix_length=10)


def test_long_tail_weight_rejects_invalid_betas():
    with pytest.raises(ValueError, match="beta2 and beta1"):
        long_tail_weight(5, min_prefix_length=1, max_prefix_length=10, beta1=0.3, beta2=0.7)


def test_aggregate_long_tail_weighted_blends_by_weight():
    blended = aggregate_long_tail_weighted((100.0, 0.0), (0.0, 100.0), weight=0.7)
    assert blended == pytest.approx((70.0, 30.0))


def test_aggregate_long_tail_weighted_rejects_out_of_range_weight():
    with pytest.raises(ValueError, match="between 0 and 1"):
        aggregate_long_tail_weighted((1.0,), (1.0,), weight=1.5)


def test_combine_ensemble_omits_long_tail_weighted_without_prefix_args():
    members = [
        _member("a", (100.0, 50.0, 25.0, 10.0)),
        _member("b", (100.0, 50.0, 25.0, 10.0)),
    ]
    result = combine_ensemble(members, CANDIDATES, target_item_id=10)
    assert result.long_tail_weighted is None
    assert result.long_tail_beta is None


def test_combine_ensemble_omits_long_tail_weighted_when_baseline_member_failed():
    members = [
        _member("broken_baseline", (), parse_success=False),
        _member("b", (100.0, 50.0, 25.0, 10.0)),
        _member("c", (100.0, 50.0, 25.0, 10.0)),
    ]
    result = combine_ensemble(
        members,
        CANDIDATES,
        target_item_id=10,
        prefix_length=3,
        min_prefix_length=1,
        max_prefix_length=49,
    )
    assert result.valid_member_count == 2  # RWRA/naive still computed from b, c
    assert result.naive is not None
    assert result.long_tail_weighted is None
    assert result.long_tail_beta is None


def test_combine_ensemble_wires_long_tail_weighting_consistently_with_the_pure_functions():
    members = [
        _member("baseline", (10.0, 100.0, 50.0, 25.0)),  # favors item 20, not target
        _member("b", (100.0, 50.0, 25.0, 10.0)),
        _member("c", (100.0, 50.0, 25.0, 10.0)),
    ]
    result = combine_ensemble(
        members,
        CANDIDATES,
        target_item_id=10,
        prefix_length=3,
        min_prefix_length=1,
        max_prefix_length=49,
        beta1=0.7,
        beta2=0.3,
    )

    expected_beta = long_tail_weight(3, min_prefix_length=1, max_prefix_length=49, beta1=0.7, beta2=0.3)
    expected_scores = aggregate_long_tail_weighted(
        result.reliability_weighted.position_scores,
        members[0].position_scores,
        expected_beta,
    )

    assert result.long_tail_beta == pytest.approx(expected_beta)
    assert result.long_tail_weighted is not None
    assert result.long_tail_weighted.position_scores == pytest.approx(expected_scores)
