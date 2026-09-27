"""Dataset-agnostic leave-one-out session and candidate-pool construction.

These functions only depend on a ``user_id``/``item_id``/``timestamp`` interaction
table; they contain no MovieLens-specific logic, so any domain loader (MovieLens,
Amazon Games, ...) can reuse them unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SessionExample:
    """One leave-one-out next-item recommendation example."""

    user_id: int
    session_id: str
    prefix_item_ids: tuple[int, ...]
    target_item_id: int
    target_timestamp: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class CandidatePool:
    """A fixed candidate list paired with one session target."""

    session_id: str
    target_item_id: int
    candidate_item_ids: tuple[int, ...]
    target_position: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def build_leave_one_out_training_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Remove each user's final chronological interaction for training use.

    The removed interactions are the targets used by
    :func:`build_leave_one_out_examples`. Keeping them out of the popularity
    calculation prevents the target from influencing negative sampling.
    """

    required_columns = {"user_id", "item_id", "timestamp"}
    missing = required_columns.difference(ratings.columns)
    if missing:
        raise ValueError(f"ratings is missing required columns: {sorted(missing)}")

    ordered = ratings.sort_values(
        ["user_id", "timestamp", "item_id"],
        kind="mergesort",
    )
    final_indices = ordered.groupby("user_id", sort=False).tail(1).index
    return ordered.drop(index=final_indices).reset_index(drop=True)


def build_leave_one_out_examples(
    ratings: pd.DataFrame,
    *,
    min_history_length: int = 3,
    max_history_length: int | None = None,
) -> list[SessionExample]:
    """Create one chronological next-item example per eligible user.

    All interactions before a user's final interaction form the prefix; the
    final interaction is the held-out target. Timestamps are sorted
    chronologically, with ``item_id`` as a deterministic tie-breaker.
    """

    required_columns = {"user_id", "item_id", "timestamp"}
    missing = required_columns.difference(ratings.columns)
    if missing:
        raise ValueError(f"ratings is missing required columns: {sorted(missing)}")
    if min_history_length < 1:
        raise ValueError("min_history_length must be at least 1")
    if max_history_length is not None and max_history_length < 1:
        raise ValueError("max_history_length must be at least 1 when provided")

    ordered = ratings.sort_values(
        ["user_id", "timestamp", "item_id"],
        kind="mergesort",
    )
    examples: list[SessionExample] = []

    for user_id, user_rows in ordered.groupby("user_id", sort=True):
        if len(user_rows) <= min_history_length:
            continue

        prefix = user_rows.iloc[:-1]["item_id"].astype(int).tolist()
        if max_history_length is not None:
            prefix = prefix[-max_history_length:]
        target = user_rows.iloc[-1]
        target_item_id = int(target["item_id"])
        if target_item_id in prefix:
            raise ValueError(
                f"Target item {target_item_id} appears in the prefix for user {int(user_id)}"
            )

        examples.append(
            SessionExample(
                user_id=int(user_id),
                session_id=f"user-{int(user_id)}",
                prefix_item_ids=tuple(prefix),
                target_item_id=target_item_id,
                target_timestamp=int(target["timestamp"]),
            )
        )

    return examples


def _item_popularity(training_ratings: pd.DataFrame) -> list[int]:
    """Return item IDs ordered by descending training interaction count."""

    if "item_id" not in training_ratings.columns:
        raise ValueError("training_ratings is missing required column: item_id")

    popularity = (
        training_ratings.groupby("item_id", as_index=False)
        .size()
        .rename(columns={"size": "interaction_count"})
        .sort_values(
            ["interaction_count", "item_id"],
            ascending=[False, True],
            kind="mergesort",
        )
    )
    return popularity["item_id"].astype(int).tolist()


def build_candidate_pools(
    training_ratings: pd.DataFrame,
    examples: list[SessionExample],
    *,
    candidate_pool_size: int = 20,
    balance_target_positions: bool = True,
) -> list[CandidatePool]:
    """Build deterministic candidate pools from training-only popularity.

    The pool contains the held-out target plus the most popular items that are
    neither the target nor already present in the session prefix. Candidate
    order is stable because popularity ties are broken by ``item_id``. When
    ``balance_target_positions`` is enabled, the target is inserted at
    positions 0, 1, ..., 19 repeatedly across the ordered examples.
    """

    if candidate_pool_size < 2:
        raise ValueError("candidate_pool_size must be at least 2")

    popularity = _item_popularity(training_ratings)
    pools: list[CandidatePool] = []

    for example_index, example in enumerate(examples):
        prefix_items = set(example.prefix_item_ids)
        negative_items: list[int] = []
        for item_id in popularity:
            if item_id == example.target_item_id or item_id in prefix_items:
                continue
            negative_items.append(item_id)
            if len(negative_items) == candidate_pool_size - 1:
                break

        if len(negative_items) != candidate_pool_size - 1:
            raise ValueError(
                f"Not enough eligible negative items for {example.session_id}; "
                f"needed {candidate_pool_size - 1}, found {len(negative_items)}"
            )

        target_position = (
            example_index % candidate_pool_size if balance_target_positions else 0
        )
        candidates = (
            negative_items[:target_position]
            + [example.target_item_id]
            + negative_items[target_position:]
        )
        pools.append(
            CandidatePool(
                session_id=example.session_id,
                target_item_id=example.target_item_id,
                candidate_item_ids=tuple(candidates),
                target_position=target_position,
            )
        )

    return pools


def _write_jsonl(records: list[object], output_path: str | Path) -> None:
    """Write objects exposing ``to_dict`` as JSON lines."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True))
            handle.write("\n")


def write_jsonl(examples: list[SessionExample], output_path: str | Path) -> None:
    """Write session examples as one reproducible JSON object per line."""

    _write_jsonl(examples, output_path)


def write_candidate_pools_jsonl(
    pools: list[CandidatePool], output_path: str | Path
) -> None:
    """Write candidate pools as one reproducible JSON object per line."""

    _write_jsonl(pools, output_path)
