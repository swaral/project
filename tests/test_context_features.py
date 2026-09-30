"""Tests for the four-field context variants (Addendum v8)."""

from __future__ import annotations

import pandas as pd
import pytest

from llm_session_reco.benchmark import ItemKNNRetriever
from llm_session_reco.context_features import ContextFeatureBuilder, series_base
from llm_session_reco.prompts import STRATEGY_PROMPT_SET, render_baseline_prompt

TITLES = {
    1: "Toy Story (1995)", 2: "Toy Story 2 (1999)", 3: "Shrek (2001)",
    4: "Shrek 2 (2004)", 5: "Heat (1995)", 6: "Godfather, The (1972)",
}
GENRES = {1: ["Animation", "Comedy"], 2: ["Animation", "Comedy"], 3: ["Animation", "Comedy"],
          4: ["Animation", "Comedy"], 5: ["Crime"], 6: ["Crime", "Drama"]}


def _builder(history_window=50, item_meta=None):
    rows = []
    # Users 10-19 watch 1 then 2 (Toy Story fans); users 20-24 watch 5 then 6.
    for u in range(10, 20):
        rows += [(u, 1, 5, 100), (u, 2, 4, 200), (u, 3, 5, 300)]
    for u in range(20, 25):
        rows += [(u, 5, 4, 100), (u, 6, 5, 200)]
    rows += [(99, 1, 5, 100), (99, 3, 2, 200)]  # the evaluated user's history
    training = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    popularity = training["item_id"].value_counts().to_dict()
    retriever = ItemKNNRetriever(training, list(TITLES), popularity)
    return ContextFeatureBuilder(training, TITLES, GENRES, retriever, year_source="title",
                                 history_window=history_window, item_meta=item_meta)


def test_series_base_strips_numbers_years_and_subtitles():
    assert series_base("Toy Story 2 (1999)") == series_base("Toy Story (1995)") == "toy story"
    assert series_base("Call of Duty: Ghosts - PS3 [Digital Code]") == "call of duty"
    assert series_base("Godfather, The (1972)") == "godfather"


def test_content_fields_show_series_and_year():
    history, candidates = _builder().fields("context_content_v2", 99, [1, 3], [2, 5])
    assert candidates[2] == "genres=Animation, Comedy; series=Toy Story; year=1999"
    assert "series=none" in candidates[5]


def test_personal_fields_never_show_a_candidate_rating():
    history, candidates = _builder().fields("context_personal_v2", 99, [1, 3], [2, 4, 5])
    assert history[3].startswith("your_rating=disliked (2/5); recency=most recent")
    assert history[1].startswith("your_rating=liked (5/5); recency=2nd most recent")
    assert all(v.startswith("your_rating=not rated; recency=new") for v in candidates.values())
    assert candidates[5].endswith("matches_you=none")


def test_crowd_fields_come_from_training_ratings():
    _, candidates = _builder().fields("context_crowd_v2", 99, [1], [2, 6])
    assert candidates[2].startswith("avg_rating=4.0/5")
    assert candidates[6].startswith("avg_rating=5.0/5")


def test_collab_fields_link_candidates_to_history():
    history, candidates = _builder().fields("context_collab_v2", 99, [1], [2, 5, 6])
    assert history == {}
    assert candidates[2] == "co_chosen_with=Toy Story (1995); link=strong; follows=Toy Story (1995)"
    assert candidates[6] == "co_chosen_with=none; link=none; follows=none"


def test_prompt_shows_fields_only_inside_the_history_window():
    builder = _builder(history_window=1)
    fields = builder.fields("context_content_v2", 99, [1, 3], [2, 5])
    prompt = render_baseline_prompt([1, 3], [2, 5], TITLES, context_variant_id="context_content_v2",
                                    item_fields=fields)
    assert "1. title=Toy Story (1995)\n" in prompt.user_message  # outside the window
    assert "2. title=Shrek (2001); genres=" in prompt.user_message
    assert "Use only the item information shown and the history" in prompt.user_message
    with pytest.raises(ValueError):
        render_baseline_prompt([1], [2, 5], TITLES, context_variant_id="context_crowd_v2")


def test_strategy_prompt_set_renders_for_both_domains():
    assert len(STRATEGY_PROMPT_SET) == 8 and STRATEGY_PROMPT_SET[0] == "baseline_scores_v1"
    for variant in STRATEGY_PROMPT_SET:
        for noun in ("movie", "game"):
            prompt = render_baseline_prompt([1], [2, 5], TITLES, variant_id=variant, domain_noun=noun)
            assert "{noun}" not in prompt.user_message


META = {
    1: {"subgenre": "Pixar", "subgenre_family": "Kids", "format": "DVD", "price": 5.0},
    2: {"subgenre": "Pixar", "subgenre_family": "Kids", "format": "Blu-ray", "price": 20.0},
    3: {"subgenre": "DreamWorks", "subgenre_family": "Kids", "format": "DVD", "price": 10.0},
    5: {"subgenre": "", "subgenre_family": "", "format": "", "price": None},
}


def _builder_with_meta():
    return _builder(item_meta=META)


def test_subgenre_fields_count_matches_in_recent_history():
    history, candidates = _builder_with_meta().fields("context_subgenre_v2", 99, [1, 3], [2, 5])
    assert candidates[2] == "subgenre=Pixar; family=Kids; in_your_history=1 of your last 2"
    assert candidates[5] == "subgenre=none; family=none; in_your_history=0 of your last 2"
    # A history item does not count itself.
    assert history[1] == "subgenre=Pixar; family=Kids; in_your_history=0 of your last 2"


def test_format_fields_show_price_level_within_the_catalog():
    _, candidates = _builder_with_meta().fields("context_format_v2", 99, [1], [1, 2, 5])
    assert candidates[1] == "format=DVD; price=$5.00; price_level=low"
    assert candidates[2] == "format=Blu-ray; price=$20.00; price_level=high"
    assert candidates[5] == "format=unknown; price=unknown; price_level=unknown"


def test_new_contexts_need_item_metadata_and_render():
    with pytest.raises(ValueError):
        _builder().fields("context_format_v2", 99, [1], [2])
    fields = _builder_with_meta().fields("context_format_v2", 99, [1, 3], [2, 5])
    prompt = render_baseline_prompt([1, 3], [2, 5], TITLES, context_variant_id="context_format_v2",
                                    item_fields=fields, domain_noun="book")
    assert "2. title=Shrek (2001); format=DVD; price=$10.00; price_level=mid" in prompt.user_message
    assert "Item context format: book title, its format, its listed price" in prompt.user_message
