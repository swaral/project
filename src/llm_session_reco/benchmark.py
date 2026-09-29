"""Clean benchmark: positive targets, popularity-matched pools, retrieval pools.

The pilots exposed two validity problems in the original benchmark:

- The original pools use the 19 most popular items as negatives, so the target
  is the least popular candidate in ~95% of pools and a "pick the least
  popular item" rule reaches HR@1 ~0.95. A model's score there says more about
  its popularity preference than about personalization.
  :func:`build_popularity_matched_pools` draws negatives from the target's own
  popularity neighbourhood and fixes how many are less popular than the target
  (uniformly at random), so the target's popularity rank inside its pool is
  uninformative by construction.
- Targets could be low ratings, and on MovieLens 42% of final ratings share a
  timestamp with the previous one, so the "next" item was picked by item ID.
  :func:`build_clean_examples` keeps only positive final interactions that are
  uniquely last in time.

:class:`ItemKNNRetriever` is a fixed, non-LLM retriever fitted on training
interactions only. :func:`build_retrieval_pools` records whether it found the
target, so rerankers can be scored both assuming retrieval worked (target
inserted) and end to end (an unretrieved target is a miss).
"""

from __future__ import annotations

import bisect
import json
import random
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .session_dataset import SessionExample

SPLITS = ("validation", "test")


@dataclass(frozen=True)
class BenchmarkExample(SessionExample):
    """A leave-one-out example with a positive, uniquely-last target."""

    target_rating: float = 0.0
    split: str = "test"


@dataclass(frozen=True)
class BenchmarkPool:
    """A candidate list for one example, plus how it was built."""

    session_id: str
    split: str
    pool_type: str
    target_item_id: int
    candidate_item_ids: tuple[int, ...]
    target_position: int
    target_popularity_rank: float
    # Retrieval pools only: the retriever's own top-k (end-to-end candidates),
    # whether it contained the target, and the target's full retrieval rank.
    retrieved_item_ids: tuple[int, ...] | None = None
    target_retrieved: bool | None = None
    target_retrieval_rank: int | None = None
    # Attribute-matched pools only: False when the target had no usable
    # attribute (or too few same-attribute items) and plain popularity
    # matching was used instead.
    attribute_matched: bool | None = None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        if self.pool_type != "retrieval":
            for key in ("retrieved_item_ids", "target_retrieved", "target_retrieval_rank"):
                data.pop(key)
        if self.attribute_matched is None:
            data.pop("attribute_matched")
        return data


def assign_split(session_id: str, *, validation_fraction: float, seed: int) -> str:
    """Deterministic validation/test assignment from a hash of the session ID."""

    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    bucket = zlib.crc32(f"{seed}:{session_id}".encode()) / 2**32
    return "validation" if bucket < validation_fraction else "test"


def build_clean_examples(
    ratings: pd.DataFrame,
    *,
    min_history_length: int = 3,
    positive_threshold: float = 4.0,
    validation_fraction: float = 0.3,
    split_seed: int = 0,
) -> tuple[list[BenchmarkExample], dict[str, int]]:
    """One example per user whose final interaction is a clean positive target.

    A user is kept only if their final interaction is rated at least
    ``positive_threshold`` and its timestamp is strictly later than the
    previous interaction's (tie policy: exclude, since the order inside a
    shared second is unknown). The prefix is every earlier interaction.
    Returns the examples and counts of users dropped by each rule.
    """

    required = {"user_id", "item_id", "timestamp", "rating"}
    missing = required.difference(ratings.columns)
    if missing:
        raise ValueError(f"ratings is missing required columns: {sorted(missing)}")

    ordered = ratings.sort_values(["user_id", "timestamp", "item_id"], kind="mergesort")
    examples: list[BenchmarkExample] = []
    dropped = {
        "short_history": 0,
        "non_positive_target": 0,
        "tied_final_timestamp": 0,
        "target_in_history": 0,
    }
    for user_id, rows in ordered.groupby("user_id", sort=True):
        if len(rows) <= min_history_length:
            dropped["short_history"] += 1
            continue
        target = rows.iloc[-1]
        if float(target["rating"]) < positive_threshold:
            dropped["non_positive_target"] += 1
            continue
        if int(rows.iloc[-2]["timestamp"]) == int(target["timestamp"]):
            dropped["tied_final_timestamp"] += 1
            continue
        prefix = rows.iloc[:-1]["item_id"].astype(int).tolist()
        if int(target["item_id"]) in prefix:
            dropped["target_in_history"] += 1
            continue
        session_id = f"user-{int(user_id)}"
        examples.append(
            BenchmarkExample(
                user_id=int(user_id),
                session_id=session_id,
                prefix_item_ids=tuple(prefix),
                target_item_id=int(target["item_id"]),
                target_timestamp=int(target["timestamp"]),
                target_rating=float(target["rating"]),
                split=assign_split(
                    session_id, validation_fraction=validation_fraction, seed=split_seed
                ),
            )
        )
    return examples, dropped


def _hash_key(item_id: int, seed: int) -> int:
    return zlib.crc32(f"{seed}:{item_id}".encode())


def popularity_midrank(candidates: Sequence[int], target: int, popularity: Mapping[int, int]) -> float:
    """Target's popularity rank from the bottom (1 = least popular), ties averaged.

    Uniform over 1..pool_size on a pool without a popularity shortcut. Many
    items share low counts, so ties must be averaged, not counted one way.
    """

    target_count = popularity.get(target, 0)
    less = sum(1 for item in candidates if item != target and popularity.get(item, 0) < target_count)
    tied = sum(1 for item in candidates if item != target and popularity.get(item, 0) == target_count)
    return 1 + less + tied / 2


def _popularity_ordering(items: Iterable[int], item_popularity: Mapping[int, int], seed: int):
    """Items in ascending popularity (seeded hash within ties), plus lookups."""

    order = sorted(
        set(int(item) for item in items),
        key=lambda item: (item_popularity.get(item, 0), _hash_key(item, seed), item),
    )
    index_of = {item: position for position, item in enumerate(order)}
    keys = [(item_popularity.get(item, 0), _hash_key(item, seed), item) for item in order]
    return order, index_of, keys


def _matched_negatives(
    example: BenchmarkExample,
    ordering,
    item_popularity: Mapping[int, int],
    rng: random.Random,
    *,
    pool_size: int,
    seed: int,
    window_factor: int,
    min_window: int,
) -> list[int]:
    """Popularity-matched negatives from ``ordering`` (see the builder below)."""

    order, index_of, keys = ordering
    negatives_needed = pool_size - 1
    target = example.target_item_id
    excluded = set(example.prefix_item_ids) | {target}
    if target in index_of:
        anchor = index_of[target]
    else:
        anchor = bisect.bisect_left(
            keys, (item_popularity.get(target, 0), _hash_key(target, seed), target)
        )

    def nearest(step: int, count: int, start: int) -> list[int]:
        found: list[int] = []
        position = start
        while 0 <= position < len(order) and len(found) < count:
            item = order[position]
            if item not in excluded:
                found.append(item)
            position += step
        return found

    above_start = anchor + 1 if target in index_of else anchor

    def windows(below: int, above: int) -> tuple[list[int], list[int]]:
        return (
            nearest(-1, max(window_factor * below, min_window), anchor - 1),
            nearest(1, max(window_factor * above, min_window), above_start),
        )

    below_count = rng.randrange(pool_size)
    above_count = negatives_needed - below_count
    below_window, above_window = windows(below_count, above_count)
    # At the popularity extremes one side can run short; borrow from the other.
    if len(below_window) < below_count:
        below_count = len(below_window)
        above_count = negatives_needed - below_count
        below_window, above_window = windows(below_count, above_count)
    elif len(above_window) < above_count:
        above_count = len(above_window)
        below_count = negatives_needed - above_count
        below_window, above_window = windows(below_count, above_count)
    if len(below_window) < below_count or len(above_window) < above_count:
        raise ValueError(f"Not enough eligible negatives for {example.session_id}")
    negatives = rng.sample(below_window, below_count) + rng.sample(above_window, above_count)
    if len(negatives) != negatives_needed:
        raise ValueError(f"Not enough eligible negatives for {example.session_id}")
    return negatives


def build_popularity_matched_pools(
    examples: Sequence[BenchmarkExample],
    item_popularity: Mapping[int, int],
    catalog_item_ids: Iterable[int],
    *,
    pool_size: int = 20,
    seed: int = 0,
    window_factor: int = 3,
    min_window: int = 30,
    item_attributes: Mapping[int, frozenset[str]] | None = None,
    pool_type: str = "popularity_matched",
) -> list[BenchmarkPool]:
    """Negatives drawn from the target's popularity neighbourhood.

    For each example a seeded draw fixes ``k``, the number of negatives less
    popular than the target, uniformly in ``0..pool_size-1``. The ``k``
    negatives are sampled from the nearest less-popular eligible items and the
    rest from the nearest more-popular ones (items tied with the target are
    split between the two sides by a seeded hash). The target's popularity
    rank inside the pool is therefore uniform, so popularity and inverse
    popularity rankers score at the random level. Only at the extremes of the
    popularity range, where one side runs out, does the draw deviate.

    With ``item_attributes`` (the hard ladder level), negatives must also
    share at least one attribute with the target - a genre on MovieLens, the
    platform on Amazon - so attribute overlap alone cannot find the target.
    A target without attributes, or with too few same-attribute items, falls
    back to plain popularity matching and is flagged ``attribute_matched``.
    """

    if pool_size < 2:
        raise ValueError("pool_size must be at least 2")
    catalog = sorted(set(int(item) for item in catalog_item_ids))
    full = _popularity_ordering(catalog, item_popularity, seed)
    by_attributes: dict[frozenset[str], tuple] = {}
    # L2 keeps its original seed label so its pools are unchanged.
    rng_label = "matched" if pool_type == "popularity_matched" else pool_type
    params = dict(pool_size=pool_size, seed=seed, window_factor=window_factor, min_window=min_window)

    pools: list[BenchmarkPool] = []
    for example in examples:
        rng = random.Random(f"{seed}:{rng_label}:{example.session_id}")
        if item_attributes is None:
            negatives = _matched_negatives(example, full, item_popularity, rng, **params)
            pools.append(_shuffled_pool(example, negatives, pool_type, item_popularity, seed))
            continue
        attributes = item_attributes.get(example.target_item_id, frozenset())
        matched = False
        if attributes:
            if attributes not in by_attributes:
                eligible = [i for i in catalog if item_attributes.get(i, frozenset()) & attributes]
                by_attributes[attributes] = _popularity_ordering(eligible, item_popularity, seed)
            try:
                negatives = _matched_negatives(
                    example, by_attributes[attributes], item_popularity, rng, **params
                )
                matched = True
            except ValueError:
                # Too few same-attribute items: redraw from the full catalog.
                rng = random.Random(f"{seed}:{rng_label}:fallback:{example.session_id}")
        if not matched:
            negatives = _matched_negatives(example, full, item_popularity, rng, **params)
        pools.append(
            _shuffled_pool(
                example, negatives, pool_type, item_popularity, seed, attribute_matched=matched
            )
        )
    return pools


def build_random_pools(
    examples: Sequence[BenchmarkExample],
    item_popularity: Mapping[int, int],
    catalog_item_ids: Iterable[int],
    *,
    pool_size: int = 20,
    seed: int = 0,
) -> list[BenchmarkPool]:
    """Easy ladder level: 19 negatives drawn uniformly from the catalog.

    Most catalog items are rarely chosen, so these pools carry a strong
    popularity signal (the target is usually more popular than its
    negatives). That is the point of the easy level; the shortcut checks
    report it rather than hide it.
    """

    catalog = sorted(set(int(item) for item in catalog_item_ids))
    pools: list[BenchmarkPool] = []
    for example in examples:
        rng = random.Random(f"{seed}:random:{example.session_id}")
        excluded = set(example.prefix_item_ids) | {example.target_item_id}
        negatives: list[int] = []
        while len(negatives) < pool_size - 1:
            item = catalog[rng.randrange(len(catalog))]
            if item not in excluded and item not in negatives:
                negatives.append(item)
        pools.append(_shuffled_pool(example, negatives, "random", item_popularity, seed))
    return pools


def amazon_platform(genres: str) -> str | None:
    """Platform from an Amazon category path, e.g. 'Video Games|Xbox One|Games'.

    The first category after 'Video Games' is the platform, except under
    'Legacy Systems', where the console is the next level down.
    """

    path = [part for part in genres.split("|") if part and part not in ("Video Games", "Games")]
    if not path or path[0] == "(unknown)":
        return None
    if path[0] == "Legacy Systems":
        return f"Legacy Systems|{path[1]}" if len(path) > 1 else None
    return path[0]


def _shuffled_pool(
    example: BenchmarkExample,
    negatives: Sequence[int],
    pool_type: str,
    item_popularity: Mapping[int, int],
    seed: int,
    **retrieval_fields: object,
) -> BenchmarkPool:
    """Store candidates in a seeded random order so the target slot is uninformative."""

    candidates = [example.target_item_id, *negatives]
    random.Random(f"{seed}:{pool_type}:order:{example.session_id}").shuffle(candidates)
    return BenchmarkPool(
        session_id=example.session_id,
        split=example.split,
        pool_type=pool_type,
        target_item_id=example.target_item_id,
        candidate_item_ids=tuple(candidates),
        target_position=candidates.index(example.target_item_id),
        target_popularity_rank=popularity_midrank(candidates, example.target_item_id, item_popularity),
        **retrieval_fields,
    )


class ItemKNNRetriever:
    """Item-based cosine retriever fitted on training interactions only.

    Scores every catalog item by its summed cosine similarity (over users who
    interacted with both) to the last ``history_window`` prefix items. Ties
    are broken by training popularity, then item ID, so rankings are
    deterministic.
    """

    def __init__(
        self,
        training_ratings: pd.DataFrame,
        catalog_item_ids: Iterable[int],
        item_popularity: Mapping[int, int],
        *,
        history_window: int = 20,
    ) -> None:
        self.history_window = history_window
        self.items = np.array(sorted(set(int(i) for i in catalog_item_ids)))
        self.column = {int(item): column for column, item in enumerate(self.items)}
        known = training_ratings[training_ratings["item_id"].isin(self.column)]
        users = pd.factorize(known["user_id"])[0]
        columns = known["item_id"].map(self.column).to_numpy()
        matrix = sp.csr_matrix(
            (np.ones(len(known)), (users, columns)),
            shape=(users.max() + 1 if len(users) else 0, len(self.items)),
        )
        matrix.data[:] = 1.0  # binary: repeated interactions count once
        self.matrix = matrix.tocsc()
        self.norms = np.sqrt(np.asarray(self.matrix.power(2).sum(axis=0)).ravel())
        popularity = np.array([item_popularity.get(int(i), 0) for i in self.items], dtype=float)
        # Secondary sort key: more popular first, then smaller item ID.
        self._tiebreak = np.lexsort((self.items, -popularity))
        self._tiebreak_rank = np.empty(len(self.items), dtype=np.int64)
        self._tiebreak_rank[self._tiebreak] = np.arange(len(self.items))

    def scores(self, prefix_item_ids: Sequence[int]) -> np.ndarray:
        """Similarity score for every catalog item (aligned with ``self.items``)."""

        history = [self.column[i] for i in prefix_item_ids[-self.history_window:] if i in self.column]
        if not history:
            return np.zeros(len(self.items))
        weights = np.divide(1.0, self.norms[history], out=np.zeros(len(history)), where=self.norms[history] > 0)
        user_vector = self.matrix[:, history] @ weights
        raw = self.matrix.T @ user_vector
        return np.divide(raw, self.norms, out=np.zeros_like(raw), where=self.norms > 0)

    def retrieve(
        self, prefix_item_ids: Sequence[int], k: int, target_item_id: int | None = None
    ) -> tuple[list[int], int | None]:
        """Top-``k`` catalog items outside the prefix, and the target's full rank."""

        scores = self.scores(prefix_item_ids)
        prefix_columns = [self.column[i] for i in prefix_item_ids if i in self.column]
        scores[prefix_columns] = -np.inf  # never re-recommend history items
        order = np.lexsort((self._tiebreak_rank, -scores))
        top = [int(self.items[j]) for j in order[:k]]
        target_rank = None
        if target_item_id is not None and target_item_id in self.column:
            target_rank = int(np.flatnonzero(order == self.column[target_item_id])[0]) + 1
        return top, target_rank


def build_retrieval_pools(
    examples: Sequence[BenchmarkExample],
    retriever: ItemKNNRetriever,
    item_popularity: Mapping[int, int],
    *,
    pool_size: int = 20,
    seed: int = 0,
) -> list[BenchmarkPool]:
    """Top-``pool_size`` retrieved items, with the target inserted when missed.

    ``retrieved_item_ids`` is the end-to-end candidate list. When the target
    is not in it, ``candidate_item_ids`` replaces the lowest-ranked retrieved
    item with the target so a reranker can still be scored assuming retrieval
    worked; ``target_retrieved`` records which case applies, and an end-to-end
    score must count unretrieved targets as misses.
    """

    pools: list[BenchmarkPool] = []
    for example in examples:
        target = example.target_item_id
        retrieved, retrieval_rank = retriever.retrieve(
            example.prefix_item_ids, pool_size, target_item_id=target
        )
        found = target in retrieved
        negatives = [item for item in retrieved if item != target][: pool_size - 1]
        pools.append(
            _shuffled_pool(
                example,
                negatives,
                "retrieval",
                item_popularity,
                seed,
                retrieved_item_ids=tuple(retrieved),
                target_retrieved=found,
                target_retrieval_rank=retrieval_rank,
            )
        )
    return pools


def write_records(records: Iterable[object], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True))
            handle.write("\n")
