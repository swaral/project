import pytest

from llm_session_reco.stability import jaccard_at_k, kendall_tau, pairwise_agreement


def test_identical_rankings_have_tau_one():
    assert kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)


def test_reversed_rankings_have_tau_negative_one():
    assert kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_tau_requires_matching_item_sets():
    with pytest.raises(ValueError, match="same set of items"):
        kendall_tau([1, 2, 3], [1, 2, 4])


def test_tau_requires_matching_length():
    with pytest.raises(ValueError, match="same length"):
        kendall_tau([1, 2, 3], [1, 2, 3, 3])


def test_single_item_ranking_has_undefined_tau_as_zero():
    assert kendall_tau([1], [1]) == 0.0


def test_jaccard_at_k_identical_top_k_is_one():
    assert jaccard_at_k([10, 20, 30], [10, 20, 99], 2) == pytest.approx(1.0)


def test_jaccard_at_k_disjoint_top_k_is_zero():
    assert jaccard_at_k([10, 20], [30, 40], 2) == pytest.approx(0.0)


def test_jaccard_at_k_partial_overlap():
    # top-2 of a: {10, 20}; top-2 of b: {20, 30}; intersection=1, union=3
    assert jaccard_at_k([10, 20, 99], [20, 30, 99], 2) == pytest.approx(1 / 3)


def test_jaccard_rejects_non_positive_k():
    with pytest.raises(ValueError, match="at least 1"):
        jaccard_at_k([1, 2], [1, 2], 0)


def test_pairwise_agreement_with_fewer_than_two_members_is_undefined():
    result = pairwise_agreement([[1, 2, 3]])
    assert result["mean_kendall_tau"] is None
    assert result["mean_jaccard_at_5"] is None


def test_pairwise_agreement_across_identical_members_is_perfect():
    members = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    result = pairwise_agreement(members, jaccard_k=(2,))
    assert result["mean_kendall_tau"] == pytest.approx(1.0)
    assert result["mean_jaccard_at_2"] == pytest.approx(1.0)


def test_pairwise_agreement_detects_a_divergent_member():
    # Two members agree; a third is fully reversed relative to them.
    members = [[1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]]
    agreeing_only = pairwise_agreement(members[:2])
    with_divergent = pairwise_agreement(members)
    assert agreeing_only["mean_kendall_tau"] == pytest.approx(1.0)
    assert with_divergent["mean_kendall_tau"] < agreeing_only["mean_kendall_tau"]
