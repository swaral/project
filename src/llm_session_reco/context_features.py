"""Four-field item context variants built only from the datasets (Addendum v8).

Every variant shows the title plus three fields, and no field repeats across
variants:

- ``context_content_v2``: genres, series, year (MovieLens: from the title;
  Amazon: first year the item appears in the training ratings);
- ``context_crowd_v2``: average rating, audience size, trend;
- ``context_personal_v2``: the user's own rating, recency, overlap with the
  user's most frequent genres;
- ``context_collab_v2``: the history item it is most often chosen with, how
  strong that link is relative to the other candidates, and the recent
  history item it most often follows.

Addendum v15 (Experiment 2b) adds two semantic variants that read the item
fields stored by the Amazon domain modules (``item_metadata``):

- ``context_subgenre_v2``: sub-genre, the wider category it belongs to, and
  how many of the user's recent items share it;
- ``context_format_v2``: format, listed price, and whether the price is
  low, mid or high for this catalog (tertiles over all priced items).

All statistics come from the leave-one-out training frame, so no evaluated
target contributes. Candidates never show a rating by this user. Fields are
rendered for the last ``history_window`` history items only, which keeps long
MovieLens prompts bounded; earlier history items show the title alone.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .benchmark import ItemKNNRetriever

CONTEXT_V2_VARIANTS = (
    "context_content_v2",
    "context_crowd_v2",
    "context_personal_v2",
    "context_collab_v2",
    "context_subgenre_v2",
    "context_format_v2",
)

_YEAR = re.compile(r"\((\d{4})\)\s*$")
_ORDINALS = {1: "most recent", 2: "2nd most recent", 3: "3rd most recent"}


def series_base(title: str) -> str:
    """Franchise stem of a title: 'Shrek 2 (2004)' -> 'shrek'."""

    text = _YEAR.sub("", title)
    text = re.sub(r"\[.*?\]|\(.*?\)", "", text)
    text = re.split(r":| - ", text)[0]
    # MovieLens moves leading articles to the end: 'Godfather, The'.
    text = re.sub(r",\s*(the|a|an)\s*$", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s+(\d+|ii|iii|iv|v|vi)\s*$", "", text.strip(), flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip().lower()


def _ordinal(position: int) -> str:
    return _ORDINALS.get(position, f"{position}th most recent")


def _short(title: str, limit: int = 40) -> str:
    title = title.strip()
    return title if len(title) <= limit else title[: limit - 1].rstrip() + "…"


class ContextFeatureBuilder:
    """Precomputes per-item statistics and renders per-session fields."""

    def __init__(
        self,
        training_ratings: pd.DataFrame,
        item_titles: Mapping[int, str],
        item_genres: Mapping[int, Sequence[str]],
        retriever: ItemKNNRetriever,
        *,
        year_source: str,
        history_window: int = 50,
        transition_window: int = 5,
        item_meta: Mapping[int, Mapping[str, object]] | None = None,
    ) -> None:
        if year_source not in ("title", "first_seen"):
            raise ValueError("year_source must be 'title' or 'first_seen'")
        self.titles = item_titles
        # Experiment 2b fields (sub-genre, format, price) per item, if available.
        self.item_meta = item_meta
        prices = [float(m["price"]) for m in (item_meta or {}).values() if m.get("price")]
        self.price_cuts = tuple(np.quantile(prices, [1 / 3, 2 / 3])) if prices else None
        self.genres = {item: list(genres) for item, genres in item_genres.items()}
        self.retriever = retriever
        self.year_source = year_source
        self.history_window = history_window
        self.transition_window = transition_window

        # Content: series stems shared by at least two catalog items.
        bases = {item: series_base(title) for item, title in item_titles.items()}
        counts = Counter(b for b in bases.values() if b)
        display: dict[str, str] = {}
        for item, base in bases.items():
            if base and counts[base] >= 2 and base not in display:
                raw = _YEAR.sub("", item_titles[item])
                display[base] = re.split(r":| - ", raw)[0].strip()
        self.series = {item: display.get(base, "none") for item, base in bases.items()}

        ratings = training_ratings
        grouped = ratings.groupby("item_id")
        self.first_year = {
            int(item): int(pd.to_datetime(ts, unit="s").year)
            for item, ts in grouped["timestamp"].min().items()
        }
        # Crowd: average rating, audience tertile, recent-share trend.
        self.avg_rating = grouped["rating"].mean().to_dict()
        count = grouped.size()
        low, high = np.quantile(count.to_numpy(), [1 / 3, 2 / 3])
        self.audience = {
            int(item): "niche" if n <= low else "moderate" if n <= high else "popular"
            for item, n in count.items()
        }
        cutoff = ratings["timestamp"].quantile(0.75)
        recent_share_all = float((ratings["timestamp"] >= cutoff).mean())
        recent = ratings[ratings["timestamp"] >= cutoff].groupby("item_id").size()
        self.trend = {}
        for item, n in count.items():
            if n < 5:
                self.trend[int(item)] = "n/a"
                continue
            ratio = (recent.get(item, 0) / n) / recent_share_all if recent_share_all else 1.0
            self.trend[int(item)] = "rising" if ratio > 1.5 else "falling" if ratio < 0.5 else "steady"
        # Personal: each user's own ratings of their history items.
        self.user_ratings = {
            (int(u), int(i)): float(r)
            for u, i, r in zip(ratings["user_id"], ratings["item_id"], ratings["rating"])
        }
        # Collab: consecutive transitions within training sequences.
        self.transitions: dict[int, Counter] = defaultdict(Counter)
        ordered = ratings.sort_values(["user_id", "timestamp", "item_id"], kind="mergesort")
        for _, sequence in ordered.groupby("user_id", sort=False)["item_id"]:
            items = sequence.tolist()
            for previous, following in zip(items, items[1:]):
                self.transitions[int(previous)][int(following)] += 1

    def _year(self, item: int) -> str:
        if self.year_source == "title":
            match = _YEAR.search(self.titles.get(item, ""))
            return f"year={match.group(1) if match else 'unknown'}"
        return f"first_seen={self.first_year.get(item, 'unknown')}"

    def _genres_text(self, item: int, limit: int = 3) -> str:
        genres = self.genres.get(item) or []
        return ", ".join(genres[:limit]) if genres else "unknown"

    def fields(
        self,
        variant: str,
        user_id: int,
        prefix_item_ids: Sequence[int],
        candidate_item_ids: Sequence[int],
    ) -> tuple[dict[int, str], dict[int, str]]:
        """(history fields, candidate fields), each ``item -> 'k=v; k=v; k=v'``."""

        recent = list(prefix_item_ids[-self.history_window:])
        if variant == "context_content_v2":
            def content(item: int) -> str:
                return f"genres={self._genres_text(item)}; series={self.series.get(item, 'none')}; {self._year(item)}"
            return {i: content(i) for i in recent}, {i: content(i) for i in candidate_item_ids}

        if variant == "context_crowd_v2":
            def crowd(item: int) -> str:
                avg = self.avg_rating.get(item)
                rating = f"{avg:.1f}/5" if avg is not None else "n/a"
                return (f"avg_rating={rating}; audience={self.audience.get(item, 'unseen')}; "
                        f"trend={self.trend.get(item, 'n/a')}")
            return {i: crowd(i) for i in recent}, {i: crowd(i) for i in candidate_item_ids}

        if variant == "context_personal_v2":
            top = [g for g, _ in Counter(g for i in recent for g in self.genres.get(i, [])).most_common(3)]

            def matches(item: int) -> str:
                shared = [g for g in self.genres.get(item, []) if g in top]
                return ", ".join(shared) if shared else "none"

            history = {}
            for position, item in enumerate(reversed(recent), start=1):
                score = self.user_ratings.get((int(user_id), int(item)))
                if score is None:
                    rating = "unknown"
                else:
                    label = "liked" if score >= 4 else "disliked" if score <= 2 else "neutral"
                    rating = f"{label} ({score:g}/5)"
                history[item] = f"your_rating={rating}; recency={_ordinal(position)}; matches_you={matches(item)}"
            candidates = {
                i: f"your_rating=not rated; recency=new; matches_you={matches(i)}" for i in candidate_item_ids
            }
            return history, candidates

        if variant == "context_collab_v2":
            return {}, self._collab_fields(recent, candidate_item_ids)

        if variant in ("context_subgenre_v2", "context_format_v2"):
            if self.item_meta is None:
                raise ValueError(f"{variant} needs item metadata (item_meta)")
            if variant == "context_subgenre_v2":
                shares = Counter(self._meta(i, "subgenre") for i in recent)

                def subgenre(item: int, own: int) -> str:
                    name = self._meta(item, "subgenre")
                    count = shares[name] - own if name else 0
                    return (f"subgenre={name or 'none'}; "
                            f"family={self._meta(item, 'subgenre_family') or 'none'}; "
                            f"in_your_history={count} of your last {len(recent)}")
                # A history item does not count itself.
                return ({i: subgenre(i, 1) for i in recent},
                        {i: subgenre(i, 0) for i in candidate_item_ids})

            def purchase(item: int) -> str:
                price = self.item_meta.get(item, {}).get("price")
                if price:
                    low, high = self.price_cuts
                    level = "low" if price <= low else "mid" if price <= high else "high"
                    shown = f"${float(price):.2f}"
                else:
                    level = shown = "unknown"
                return f"format={self._meta(item, 'format') or 'unknown'}; price={shown}; price_level={level}"
            return {i: purchase(i) for i in recent}, {i: purchase(i) for i in candidate_item_ids}

        raise ValueError(f"Unknown context variant {variant!r}")

    def _meta(self, item: int, field: str) -> str:
        return str((self.item_meta or {}).get(item, {}).get(field) or "")

    def _collab_fields(self, recent: Sequence[int], candidates: Sequence[int]) -> dict[int, str]:
        r = self.retriever
        history_cols = [(i, r.column[i]) for i in recent if i in r.column]
        best: dict[int, tuple[float, int | None]] = {}
        if history_cols:
            h_idx = [c for _, c in history_cols]
            co = (r.matrix[:, h_idx].T @ r.matrix).tocsc()
            for item in candidates:
                if item not in r.column:
                    best[item] = (0.0, None)
                    continue
                column = co[:, r.column[item]].toarray().ravel()
                denominator = r.norms[h_idx] * r.norms[r.column[item]]
                cosine = np.divide(column, denominator, out=np.zeros_like(column), where=denominator > 0)
                k = int(np.argmax(cosine))
                best[item] = (float(cosine[k]), history_cols[k][0] if cosine[k] > 0 else None)
        else:
            best = {item: (0.0, None) for item in candidates}
        # Link strength relative to this session's candidates: top 5 strong,
        # next 5 medium, other linked items weak, unlinked none.
        linked = sorted((v[0], item) for item, v in best.items() if v[0] > 0)[::-1]
        strength = {}
        for rank, (_, item) in enumerate(linked):
            strength[item] = "strong" if rank < 5 else "medium" if rank < 10 else "weak"
        last = list(recent[-self.transition_window:])
        out = {}
        for item in candidates:
            partner = best[item][1]
            co_with = _short(self.titles[partner]) if partner is not None else "none"
            follows = max(last, key=lambda h: self.transitions[h][item], default=None)
            follows_text = (
                _short(self.titles[follows])
                if follows is not None and self.transitions[follows][item] > 0 else "none"
            )
            out[item] = (f"co_chosen_with={co_with}; link={strength.get(item, 'none')}; "
                         f"follows={follows_text}")
        return out
