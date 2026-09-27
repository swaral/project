import json

import pytest

from llm_session_reco.llm_client import ChatCompletionsClient, StubLLMClient, create_client
from llm_session_reco.parser import parse_ranking
from llm_session_reco.prompts import render_baseline_prompt


def _prompt():
    return render_baseline_prompt(
        prefix_item_ids=[10, 20],
        candidate_item_ids=[30, 40, 50],
        item_titles={
            10: "Movie A",
            20: "Movie B",
            30: "Movie C",
            40: "Movie D",
            50: "Movie E",
        },
    )


def test_baseline_prompt_preserves_candidate_order_and_requires_json():
    prompt = _prompt()

    assert prompt.variant_id == "baseline_scores_v1"
    assert "1. title=Movie C" in prompt.user_message
    assert prompt.user_message.index("1. title=Movie C") < prompt.user_message.index(
        "2. title=Movie D"
    )
    assert '"scores"' in prompt.user_message
    assert "Movie A" in prompt.user_message


def test_wording_variants_change_only_the_prompt_wording():
    variant_ids = [
        "baseline_scores_v1",
        "wording_direct_v1",
        "wording_preference_v1",
        "wording_detailed_v1",
    ]
    prompts = [
        render_baseline_prompt(
            prefix_item_ids=[10, 20],
            candidate_item_ids=[30, 40, 50],
            item_titles={
                10: "Movie A",
                20: "Movie B",
                30: "Movie C",
                40: "Movie D",
                50: "Movie E",
            },
            variant_id=variant_id,
        )
        for variant_id in variant_ids
    ]

    assert {prompt.variant_id for prompt in prompts} == set(variant_ids)
    assert len({prompt.user_message for prompt in prompts}) == len(variant_ids)
    assert len({prompt.system_message for prompt in prompts}) == 1
    for prompt in prompts:
        assert "1. title=Movie C" in prompt.user_message
        assert '"scores"' in prompt.user_message
        assert "Return only one JSON object" in prompt.user_message


def test_unknown_prompt_variant_is_rejected():
    with pytest.raises(ValueError, match="Unknown prompt variant"):
        render_baseline_prompt(
            prefix_item_ids=[10],
            candidate_item_ids=[20, 30],
            item_titles={10: "Movie A", 20: "Movie B", 30: "Movie C"},
            variant_id="not_a_real_variant",
        )


def test_parser_converts_scores_to_ranking_and_computes_one_based_target_rank():
    parsed = parse_ranking(
        '{"scores": [20, 90, 10]}',
        [30, 40, 50],
        target_item_id=30,
    )

    assert parsed.parse_success is True
    assert parsed.position_scores == (20.0, 90.0, 10.0)
    assert parsed.ranked_positions == (2, 1, 3)
    assert parsed.ranked_item_ids == (40, 30, 50)
    assert parsed.target_rank == 2
    assert parsed.parse_error is None
    assert parsed.to_dict()["position_scores"] == [20.0, 90.0, 10.0]
    assert parsed.to_dict()["ranked_positions"] == [2, 1, 3]


def test_parser_rejects_incomplete_or_out_of_range_scores():
    parsed = parse_ranking(
        '{"scores": [20, 101, 10]}',
        [30, 40, 50],
        target_item_id=30,
    )

    assert parsed.parse_success is False
    assert "0..100" in parsed.parse_error


def test_stub_client_round_trips_through_strict_parser():
    prompt = _prompt()
    response = StubLLMClient().generate(prompt.system_message, prompt.user_message)
    parsed = parse_ranking(
        response.raw_text,
        [30, 40, 50],
        target_item_id=30,
    )

    assert json.loads(response.raw_text)["scores"] == [3, 2, 1]
    assert parsed.parse_success is True
    assert parsed.ranked_positions == (1, 2, 3)
    assert parsed.ranked_item_ids == (30, 40, 50)
    assert parsed.target_rank == 1


def test_openrouter_client_uses_secret_environment_and_explicit_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "provider/model")

    client = create_client(
        "openrouter",
        model=None,
        base_url=None,
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
        timeout_seconds=600,
        json_mode=True,
        candidate_count=20,
    )

    assert isinstance(client, ChatCompletionsClient)
    assert client.api_key == "test-key"
    assert client.model == "provider/model"
    assert client.base_url == "https://openrouter.ai/api/v1"
    assert "api_key" not in client.config


def test_openrouter_requires_key_and_model(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    options = {
        "model": None,
        "base_url": None,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 256,
        "timeout_seconds": 600,
        "json_mode": True,
        "candidate_count": 20,
    }

    with pytest.raises(ValueError, match="OPENROUTER_MODEL"):
        create_client("openrouter", **options)

    monkeypatch.setenv("OPENROUTER_MODEL", "provider/model")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        create_client("openrouter", **options)
