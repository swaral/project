"""Debiased prompt-ensemble aggregation: shuffle, normalize, calibrate.

The pilots showed that the ensemble's gains came from removing list-position
bias, not from RWRA's agreement weights (which stay within about 0.25 +/- 0.03,
i.e. a plain mean). This module aggregates members in three steps:

1. Shuffle: each member saw its own seeded permutation of the pool
   (``run_ensemble.py --shuffle-candidates``), so a slot preference falls on
   a different item under each member.
2. Normalize: each member's scores are z-scored within the session, because
   members use different score scales and a raw mean lets the widest-spread
   member dominate.
3. Calibrate: each member's remaining preference for particular presented
   slots (its mean z-score per slot) is estimated on validation sessions only
   and subtracted before averaging.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MemberAnswer:
    """One parsed member answer, keyed by item and by presented slot."""

    member: str
    item_scores: dict[int, float]
    slot_of_item: dict[int, int]


def member_answers(record: Mapping[str, object]) -> dict[str, MemberAnswer]:
    """Parsed member answers of one ensemble trial record, keyed by member."""

    pool = list(record["candidate_item_ids"])
    answers: dict[str, MemberAnswer] = {}
    for trial in record.get("member_trials") or []:
        if not trial.get("parse_success"):
            continue
        presented = list(trial.get("presented_candidate_item_ids") or pool)
        member = f"{trial['variant_id']}:{trial['context_variant_id']}"
        answers[member] = MemberAnswer(
            member=member,
            item_scores=dict(zip(presented, (float(s) for s in trial["position_scores"]))),
            slot_of_item={item: slot for slot, item in enumerate(presented)},
        )
    return answers


def zscore(scores: Mapping[int, float]) -> dict[int, float]:
    """Within-session standardization; a constant answer maps to all zeros."""

    values = np.fromiter(scores.values(), dtype=float)
    spread = values.std()
    if spread == 0:
        return {item: 0.0 for item in scores}
    mean = values.mean()
    return {item: (score - mean) / spread for item, score in scores.items()}


def fit_slot_priors(answers: Iterable[MemberAnswer]) -> dict[str, dict[int, float]]:
    """Mean z-score each member gives to each presented slot.

    Fit on validation sessions only. With shuffled pools the target is equally
    likely to sit in any slot, so a non-zero slot mean measures position bias,
    not item quality.
    """

    sums: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for answer in answers:
        for item, value in zscore(answer.item_scores).items():
            slot = answer.slot_of_item[item]
            sums[answer.member][slot] += value
            counts[answer.member][slot] += 1
    return {
        member: {slot: sums[member][slot] / counts[member][slot] for slot in sums[member]}
        for member in sums
    }


def calibrated_scores(answer: MemberAnswer, priors: Mapping[str, Mapping[int, float]]) -> dict[int, float]:
    """z-scored answer with the member's validation slot prior removed."""

    prior = priors.get(answer.member, {})
    return {
        item: value - prior.get(answer.slot_of_item[item], 0.0)
        for item, value in zscore(answer.item_scores).items()
    }


def mean_scores(per_member: Sequence[Mapping[int, float]]) -> dict[int, float]:
    """Unweighted mean across members of item-keyed scores."""

    if not per_member:
        raise ValueError("need at least one member")
    items = per_member[0].keys()
    return {item: sum(scores[item] for scores in per_member) / len(per_member) for item in items}
