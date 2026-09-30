"""Tests for the Amazon Books and CDs & Vinyl preprocessing."""

from __future__ import annotations

import json

import pandas as pd

from llm_session_reco import amazon_media
from llm_session_reco.amazon_media import (
    CATEGORIES,
    MUSIC_GENRE_MAP,
    build_subset,
    creator,
    display_title,
    genre,
    load_items,
    load_ratings,
    stream_metadata,
)


def test_creator_from_author_object_or_store():
    assert creator({"author": {"name": "Peter Ackroyd", "about": []}, "store": "x"}) == "Peter Ackroyd"
    assert creator({"author": None, "store": "Shrimp City Slim  (Artist)    Format: Audio CD"}) == "Shrimp City Slim"
    assert creator({"store": "SWV   Format: Audio CD"}) == "SWV"
    assert creator({"store": "John Barry  (Composer),     Barry, John  (Composer)    Format: Audio CD"}) == "John Barry"
    assert creator({"store": "Remington Kane (Author)   Format: Kindle Edition"}) == "Remington Kane"
    assert creator({"store": None}) == ""
    assert creator({"store": "Various Artists  (Artist)   Format: Audio CD"}) == ""
    assert creator({"store": "Rated: Unrated   Format: Audio CD"}) == ""


def test_genre_is_the_first_mapped_label_on_the_path():
    genres = MUSIC_GENRE_MAP
    assert genre(["CDs & Vinyl", "Jazz", "Bebop"], genres) == ["Jazz"]
    # Store sections are skipped; the real genre sits one level down.
    assert genre(["CDs & Vinyl", "Today's Deals in Music", "Country"], genres) == ["Country"]
    assert genre(["CDs & Vinyl", "Today's Deals in Music", "Hard Rock & Metal"], genres) == ["Metal"]
    assert genre(["CDs & Vinyl", "Rhino Records"], genres) == []
    assert genre(["CDs & Vinyl"], genres) == []


def test_display_title():
    assert display_title("Greatest Hits", "Queen") == "Greatest Hits by Queen"
    assert display_title("Greatest Hits", "") == "Greatest Hits"


def _meta_line(asin, title, store, categories):
    return json.dumps({"parent_asin": asin, "title": title, "store": store,
                       "categories": categories, "description": ["long text"]})


def test_stream_metadata_keeps_only_wanted_items(tmp_path):
    source = tmp_path / "meta.jsonl"
    source.write_text("\n".join([
        _meta_line("A1", "Kind of  Blue", "Miles Davis  (Artist)  Format: Audio CD", ["CDs & Vinyl", "Jazz"]),
        _meta_line("A2", "Unwanted", "Nobody", ["CDs & Vinyl", "Pop"]),
    ]) + "\n")
    output = tmp_path / "compact.jsonl"
    assert stream_metadata(source.as_uri(), {"A1"}, output) == 1
    record = json.loads(output.read_text())
    assert record == {"parent_asin": "A1", "title": "Kind of Blue", "creator": "Miles Davis",
                      "categories": ["CDs & Vinyl", "Jazz"], "store_format": "Audio CD",
                      "main_category": "", "details_keys": [], "price": None}
    assert not output.with_suffix(".jsonl.part").exists()


class _FakeResponse:
    def __init__(self, body, *, announced, status):
        self._body, self.status = body, status
        self.headers = {"Content-Length": str(announced)}

    def read(self, size):
        chunk, self._body = self._body[:size], self._body[size:]
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_lines_resumes_after_a_silently_truncated_response(monkeypatch):
    data = b"line-one\nline-two\nline-three\n"
    requests = []

    def fake_urlopen(request, timeout):
        start = int(request.headers.get("Range", "bytes=0-")[6:-1])
        requests.append(start)
        if start == 0:
            # The connection drops mid-line but read() just returns b"".
            return _FakeResponse(data[:13], announced=len(data), status=200)
        return _FakeResponse(data[start:], announced=len(data) - start, status=206)

    monkeypatch.setattr(amazon_media, "urlopen", fake_urlopen)
    monkeypatch.setattr(amazon_media.time, "sleep", lambda seconds: None)
    assert list(amazon_media._lines("https://example.test/meta.jsonl")) == [
        b"line-one", b"line-two", b"line-three"
    ]
    # Resumed exactly after the last complete line, not mid-line.
    assert requests == [0, len(b"line-one\n")]


def test_build_subset_round_trips_through_loaders(tmp_path):
    category = CATEGORIES["amazon_music"]
    root = tmp_path / category.subdir
    root.mkdir()
    rows = [
        {"user_id": f"U{u}", "parent_asin": f"A{i}", "rating": 5.0,
         "timestamp": 1_600_000_000_000 + (u * 10 + i) * 86_400_000}
        for u in range(8) for i in range(6)
    ]
    pd.DataFrame(rows).to_csv(root / category.raw_ratings, index=False)
    meta = tmp_path / "meta.jsonl"
    meta.write_text("\n".join(
        _meta_line(f"A{i}", f"Album {i}", f"Band {i}  (Artist)  Format: Audio CD", ["CDs & Vinyl", "Rock"])
        for i in range(6)
    ) + "\n")

    summary = build_subset("amazon_music", root, user_fraction=1.0, metadata_url=meta.as_uri())
    assert summary["users"] == 8 and summary["items"] == 6 and summary["items_with_creator"] == 6
    # The metadata cache is reused, so a rebuild needs no network.
    meta.unlink()
    assert build_subset("amazon_music", root, user_fraction=1.0)["ratings_sha256"] == summary["ratings_sha256"]

    ratings = load_ratings("amazon_music", tmp_path, id_map_output=tmp_path / "map.json")
    items = load_items("amazon_music", tmp_path, json.loads((tmp_path / "map.json").read_text())["item_id_map"])
    assert len(ratings) == 48
    assert set(items["title"]) == {f"Album {i} by Band {i}" for i in range(6)}
    assert set(items["genres"]) == {"Rock"}
