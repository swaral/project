"""Non-LLM reference rankers scored on the same candidate lists as the LLM.

Each ranker maps (prefix item IDs, candidate item IDs) to a score per
candidate; higher is better, and ties are resolved by the tie-aware metrics.
All statistics come from training interactions only. The popularity and
inverse-popularity rankers double as shortcut detectors: on a fair pool both
should score at the random level.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence

import pandas as pd

from .benchmark import ItemKNNRetriever

Ranker = Callable[[Sequence[int], Sequence[int]], dict[int, float]]

_STOPWORDS = frozenset(
    "the a an of and for with to in on edition game games video by".split()
)


def title_tokens(title: str) -> frozenset[str]:
    """Lower-cased content words of a title, without a trailing (year)."""

    title = re.sub(r"\(\d{4}\)", "", title.lower())
    return frozenset(
        word for word in re.findall(r"[a-z0-9]+", title) if word not in _STOPWORDS and len(word) > 1
    )


def build_reference_rankers(
    training_ratings: pd.DataFrame,
    item_popularity: Mapping[int, int],
    item_titles: Mapping[int, str],
    item_genres: Mapping[int, set[str]],
    retriever: ItemKNNRetriever,
    *,
    profile_window: int = 50,
    transition_window: int = 5,
) -> dict[str, Ranker]:
    """The baseline family reported next to every LLM result."""

    tokens = {item: title_tokens(title) for item, title in item_titles.items()}
    transitions: dict[int, Counter] = defaultdict(Counter)
    ordered = training_ratings.sort_values(["user_id", "timestamp", "item_id"], kind="mergesort")
    for _, sequence in ordered.groupby("user_id", sort=False)["item_id"]:
        items = sequence.tolist()
        for previous, following in zip(items, items[1:]):
            transitions[previous][following] += 1

    def constant(prefix: Sequence[int], candidates: Sequence[int]) -> dict[int, float]:
        return {item: 0.0 for item in candidates}

    def popularity(prefix: Sequence[int], candidates: Sequence[int]) -> dict[int, float]:
        return {item: float(item_popularity.get(item, 0)) for item in candidates}

    def inverse_popularity(prefix: Sequence[int], candidates: Sequence[int]) -> dict[int, float]:
        return {item: -float(item_popularity.get(item, 0)) for item in candidates}

    def genre_overlap(prefix: Sequence[int], candidates: Sequence[int]) -> dict[int, float]:
        profile = Counter(g for item in prefix[-profile_window:] for g in item_genres.get(item, ()))
        total = sum(profile.values()) or 1
        return {item: sum(profile[g] for g in item_genres.get(item, ())) / total for item in candidates}

    def title_overlap(prefix: Sequence[int], candidates: Sequence[int]) -> dict[int, float]:
        history = [tokens.get(item, frozenset()) for item in prefix[-profile_window:]]
        scores = {}
        for item in candidates:
            own = tokens.get(item, frozenset())
            scores[item] = max(
                (len(own & past) / len(own | past) for past in history if own | past), default=0.0
            )
        return scores

    def item_knn(prefix: Sequence[int], candidates: Sequence[int]) -> dict[int, float]:
        all_scores = retriever.scores(prefix)
        return {
            item: float(all_scores[retriever.column[item]]) if item in retriever.column else 0.0
            for item in candidates
        }

    def sequential(prefix: Sequence[int], candidates: Sequence[int]) -> dict[int, float]:
        recent = list(reversed(prefix[-transition_window:]))
        return {
            item: sum(transitions[past][item] / (distance + 1) for distance, past in enumerate(recent))
            for item in candidates
        }

    return {
        "random": constant,
        "popularity": popularity,
        "inverse_popularity": inverse_popularity,
        "genre_overlap": genre_overlap,
        "title_overlap": title_overlap,
        "item_knn": item_knn,
        "sequential_transitions": sequential,
    }
