"""Tests for crash-safe resume in scripts/run_ensemble.py."""

from __future__ import annotations

import json

from scripts.run_ensemble import (
    _default_output_path,
    _existing_model_names,
    _existing_trial_ids,
    _repair_truncated_tail,
)


def _line(trial_id: str) -> str:
    return json.dumps({"trial_id": trial_id, "value": 1}) + "\n"


def test_missing_and_empty_files_are_left_alone(tmp_path):
    missing = tmp_path / "missing.jsonl"
    assert _repair_truncated_tail(missing) == 0
    assert not missing.exists()

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert _repair_truncated_tail(empty) == 0
    assert empty.read_text(encoding="utf-8") == ""


def test_a_cleanly_terminated_file_is_unchanged(tmp_path):
    path = tmp_path / "trials.jsonl"
    content = _line("user-1") + _line("user-2")
    path.write_text(content, encoding="utf-8")

    assert _repair_truncated_tail(path) == 0
    assert path.read_text(encoding="utf-8") == content


def test_a_partial_final_line_is_truncated_back_to_the_last_complete_record(tmp_path):
    path = tmp_path / "trials.jsonl"
    complete = _line("user-1") + _line("user-2")
    fragment = '{"trial_id": "user-3", "val'
    path.write_text(complete + fragment, encoding="utf-8")

    removed = _repair_truncated_tail(path)

    assert removed == len(fragment.encode("utf-8"))
    assert path.read_text(encoding="utf-8") == complete
    # The fragment's trial never parsed, so it is not treated as done and will
    # be re-run on resume.
    assert _existing_trial_ids(path) == {"user-1", "user-2"}


def test_a_file_holding_only_a_fragment_is_emptied(tmp_path):
    path = tmp_path / "trials.jsonl"
    fragment = '{"trial_id": "user-1", "dri'
    path.write_text(fragment, encoding="utf-8")

    assert _repair_truncated_tail(path) == len(fragment.encode("utf-8"))
    assert path.read_bytes() == b""


def test_appending_after_repair_yields_a_fully_parseable_file(tmp_path):
    """The failure this guards against: resume fusing a new record onto a fragment."""

    path = tmp_path / "trials.jsonl"
    path.write_text(_line("user-1") + '{"trial_id": "user-2", "dr', encoding="utf-8")

    _repair_truncated_tail(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_line("user-2"))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["trial_id"] for line in lines] == ["user-1", "user-2"]


def test_default_ensemble_output_paths_separate_qwen_sizes():
    output_3b = _default_output_path("qwen2.5:3b-instruct", "movielens")
    output_7b = _default_output_path("qwen2.5:7b-instruct", "movielens")

    assert output_3b.name == "ml1m_3b_ensemble_trials.jsonl"
    assert output_7b.name == "ml1m_7b_ensemble_trials.jsonl"
    assert output_3b != output_7b


def test_existing_model_names_are_read_for_resume_guard(tmp_path):
    path = tmp_path / "trials.jsonl"
    path.write_text(
        '{"session_id":"user-1","client_config":{"model":"qwen2.5:3b-instruct"}}\n',
        encoding="utf-8",
    )

    assert _existing_model_names(path) == {"qwen2.5:3b-instruct"}
