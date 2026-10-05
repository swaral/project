"""Tests for per-member candidate shuffling in scripts/run_ensemble.py."""

from __future__ import annotations

from scripts.run_ensemble import presented_candidate_order, scores_in_pool_order

POOL = list(range(100, 120))


def _order(variant_id, *, session_id="user-1", seed=0):
    return presented_candidate_order(
        POOL,
        session_id=session_id,
        variant_id=variant_id,
        context_variant_id="context_title_v1",
        shuffle_seed=seed,
    )


def test_without_a_seed_the_pool_order_is_kept():
    assert (
        presented_candidate_order(
            POOL,
            session_id="user-1",
            variant_id="baseline_scores_v1",
            context_variant_id="context_title_v1",
            shuffle_seed=None,
        )
        == POOL
    )


def test_shuffle_is_a_reproducible_permutation():
    order = _order("baseline_scores_v1")
    assert sorted(order) == POOL
    assert order != POOL
    assert order == _order("baseline_scores_v1")


def test_each_member_and_session_gets_its_own_order():
    assert _order("baseline_scores_v1") != _order("wording_direct_v1")
    assert _order("baseline_scores_v1") != _order("baseline_scores_v1", session_id="user-2")
    assert _order("baseline_scores_v1") != _order("baseline_scores_v1", seed=1)


def test_scores_map_back_to_pool_positions():
    presented = _order("wording_detailed_v1")
    # Score each presented slot by its item ID so the mapping is checkable.
    presented_scores = tuple(float(item_id) for item_id in presented)
    assert scores_in_pool_order(presented, presented_scores, POOL) == tuple(
        float(item_id) for item_id in POOL
    )


def test_shuffle_key_context_reuses_that_contexts_order():
    grounded = presented_candidate_order(
        POOL,
        session_id="user-1",
        variant_id="next_step_v2",
        context_variant_id="context_collab_v2",
        shuffle_seed=0,
        shuffle_context_id="context_title_v1",
    )
    assert grounded == _order("next_step_v2")
    assert grounded != presented_candidate_order(
        POOL,
        session_id="user-1",
        variant_id="next_step_v2",
        context_variant_id="context_collab_v2",
        shuffle_seed=0,
    )
