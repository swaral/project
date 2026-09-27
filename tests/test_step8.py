import pytest

from llm_session_reco.prompts import render_baseline_prompt


ITEM_TITLES = {
    10: "Toy Story (1995)",
    20: "Heat (1995)",
    30: "The Matrix (1999)",
}

ITEM_GENRES = {
    10: "Animation|Children's|Comedy",
    20: "Action|Crime|Thriller",
    30: "Action|Sci-Fi|Thriller",
}


def _render(context_variant_id: str):
    return render_baseline_prompt(
        prefix_item_ids=[10, 20],
        candidate_item_ids=[30, 10, 20],
        item_titles=ITEM_TITLES,
        item_genres=ITEM_GENRES,
        context_variant_id=context_variant_id,
    )


def test_context_variants_change_context_but_keep_the_same_task_contract():
    prompts = [
        _render("context_title_v1"),
        _render("context_genre_v1"),
        _render("context_rich_v1"),
    ]

    assert {prompt.context_variant_id for prompt in prompts} == {
        "context_title_v1",
        "context_genre_v1",
        "context_rich_v1",
    }
    assert len({prompt.user_message for prompt in prompts}) == 3
    assert len({prompt.system_message for prompt in prompts}) == 1
    for prompt in prompts:
        assert "Assign one relevance score" in prompt.user_message
        assert '"scores"' in prompt.user_message
        assert "1. " in prompt.user_message

    title_prompt, genre_prompt, rich_prompt = prompts
    assert "genres=" not in title_prompt.user_message
    assert "genres=Action|Sci-Fi|Thriller" in genre_prompt.user_message
    assert "release_year=1999" in rich_prompt.user_message


def test_non_title_context_requires_genre_metadata():
    with pytest.raises(ValueError, match="item_genres is required"):
        render_baseline_prompt(
            prefix_item_ids=[10],
            candidate_item_ids=[20, 30],
            item_titles=ITEM_TITLES,
            context_variant_id="context_genre_v1",
        )


def test_unknown_context_variant_is_rejected():
    with pytest.raises(ValueError, match="Unknown context variant"):
        render_baseline_prompt(
            prefix_item_ids=[10],
            candidate_item_ids=[20, 30],
            item_titles=ITEM_TITLES,
            context_variant_id="not_a_real_context",
        )
