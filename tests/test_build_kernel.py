import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _metadata(tmp_path, *args):
    subprocess.run(
        [sys.executable, str(ROOT / "kaggle" / "build_kernel.py"), "--username", "someone",
         "--output-dir", str(tmp_path), *args],
        check=True, capture_output=True,
    )
    return json.loads((tmp_path / "kernel-metadata.json").read_text(encoding="utf-8"))


def test_long_kernel_title_is_shortened_and_matches_slug(tmp_path):
    meta = _metadata(
        tmp_path, "--experiment", "context2b", "--domains", "amazon_books,amazon_music",
        "--pools", "random,popularity_matched,attribute_matched,recency_matched",
        "--conditions", "shuffled",
    )
    assert len(meta["title"]) <= 50
    assert meta["title"] == "LLM Reco 3b-context2b-ladder-l4-books-music"
    assert meta["id"] == "someone/" + meta["title"].lower().replace(" ", "-")


def test_short_kernel_title_is_unchanged(tmp_path):
    meta = _metadata(tmp_path, "--domains", "amazon_games", "--conditions", "shuffled")
    assert meta["title"] == "LLM Session Reco Pilot 3b-shuffled-games"
    assert meta["id"] == "someone/llm-session-reco-pilot-3b-shuffled-games"
