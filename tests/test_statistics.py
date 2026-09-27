import pytest

from llm_session_reco.statistics import paired_bootstrap_ci, wilcoxon_signed_rank


def test_bootstrap_ci_is_centered_at_zero_for_identical_inputs():
    values = [0.5, 0.2, 0.8, 0.1, 0.9]
    result = paired_bootstrap_ci(values, values, seed=42, n_boot=500)

    assert result["mean_difference"] == pytest.approx(0.0)
    assert result["ci_low"] == pytest.approx(0.0)
    assert result["ci_high"] == pytest.approx(0.0)
    assert result["n_pairs"] == 5


def test_bootstrap_ci_detects_a_consistent_positive_shift():
    values_a = [1.0, 1.0, 1.0, 1.0, 1.0]
    values_b = [0.0, 0.0, 0.0, 0.0, 0.0]
    result = paired_bootstrap_ci(values_a, values_b, seed=1, n_boot=500)

    assert result["mean_difference"] == pytest.approx(1.0)
    assert result["ci_low"] > 0.0


def test_bootstrap_ci_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_ci([1.0, 2.0], [1.0])


def test_bootstrap_ci_rejects_empty_input():
    with pytest.raises(ValueError, match="cannot be empty"):
        paired_bootstrap_ci([], [])


def test_wilcoxon_all_equal_pairs_is_undefined():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = wilcoxon_signed_rank(values, values)

    assert result["statistic"] is None
    assert result["p_value"] is None
    assert result["n_pairs"] == 5


def test_wilcoxon_detects_a_consistent_shift():
    values_a = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    values_b = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]
    result = wilcoxon_signed_rank(values_a, values_b)

    assert result["p_value"] is not None
    assert result["p_value"] < 0.05


def test_wilcoxon_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        wilcoxon_signed_rank([1.0, 2.0], [1.0])
