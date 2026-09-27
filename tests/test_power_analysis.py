import math

from scripts.power_analysis import achieved_power, analyse, required_sessions


def _record(session_id, baseline_rank, rwra_rank):
    return {
        "session_id": session_id,
        "client_config": {"model": "qwen2.5:3b-instruct"},
        "member_trials": [{"parse_success": True, "target_rank": baseline_rank}],
        "naive_mean": {"target_rank": baseline_rank},
        "reliability_weighted": {"target_rank": rwra_rank},
        "long_tail_weighted": {"target_rank": rwra_rank},
    }


def test_required_sessions_matches_textbook_value_after_are_correction():
    # d = 0.5, alpha 0.05, power 0.8: normal approximation gives 31.4 sessions.
    assert required_sessions(0.5, alpha=0.05, power=0.8) == math.ceil(31.4 / 0.864)


def test_power_at_required_sessions_reaches_target():
    needed = required_sessions(0.2, alpha=0.05, power=0.8)
    assert achieved_power(0.2, needed, alpha=0.05) >= 0.8
    assert achieved_power(0.2, needed - 10, alpha=0.05) < 0.8


def test_no_effect_reports_null_verdict():
    records = [_record(str(i), 3, 3) for i in range(5)] + [
        _record("up", 4, 2),
        _record("down", 2, 4),
    ]
    result = analyse(records, planned_n=3000)["comparisons"]
    comparison = result["reliability_weighted_vs_single_prompt_baseline"]
    assert comparison["sessions_needed"] is None
    assert comparison["verdict"].startswith("no effect")


def test_clear_improvement_is_worth_a_full_run():
    records = [_record(str(i), 5, 1 if i % 2 else 5) for i in range(40)]
    comparison = analyse(records, planned_n=3000)["comparisons"][
        "reliability_weighted_vs_single_prompt_baseline"
    ]
    assert comparison["sessions_improved"] == 20
    assert comparison["verdict"].startswith("worth it")
