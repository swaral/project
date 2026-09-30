"""Tests for the Experiment 2b item fields: sub-genre, format and price."""

from __future__ import annotations

from llm_session_reco.amazon_games import game_item_fields
from llm_session_reco.amazon_media import media_format
from llm_session_reco.amazon_movies import movie_item_fields
from llm_session_reco.item_metadata import (
    clean_format,
    items_frame,
    parse_price,
    store_format,
    subgenre_below,
)


def test_parse_price_and_format_labels():
    assert parse_price(12.99) == 12.99
    assert parse_price("$1,299.00") == 1299.0
    assert parse_price(0.0) is None and parse_price(None) is None and parse_price("None") is None
    assert store_format("Shrimp City Slim  (Artist)    Format: Audio CD") == "Audio CD"
    assert clean_format("DVD-R  This title is manufactured on demand") == "DVD-R"
    assert clean_format("Unknown Binding") == ""


def test_subgenre_is_the_path_below_the_first_anchor():
    anchors = {"Jazz", "Country"}
    assert subgenre_below(["CDs & Vinyl", "Jazz", "Bebop"], anchors) == ("Bebop", "")
    assert subgenre_below(["CDs & Vinyl", "Today's Deals", "Country", "Honky Tonk", "Classic"], anchors) == (
        "Classic", "Honky Tonk")
    assert subgenre_below(["CDs & Vinyl", "Jazz"], anchors) == ("", "")
    # Store sections and "General" are not sub-genres.
    assert subgenre_below(["CDs & Vinyl", "Jazz", "Bebop", "CDs $7 - $10"], anchors) == ("Bebop", "")
    assert subgenre_below(["Movies & TV", "Jazz", "General"], anchors) == ("", "")
    assert subgenre_below(["CDs & Vinyl", "Rhino Records"], anchors) == ("", "")


def test_book_and_music_formats():
    assert media_format({"store_format": "Kindle Edition", "details_keys": ["Paperback"]}) == "Kindle Edition"
    assert media_format({"store_format": "", "details_keys": ["Language", "Hardcover"]}) == "Hardcover"
    assert media_format({"store_format": "", "details_keys": [], "main_category": "Audible Audiobooks"}) == "Audiobook"
    assert media_format({"store_format": "", "details_keys": ["File size"]}) == "Kindle Edition"
    assert media_format({"store_format": "", "details_keys": ["Language"]}) == ""


def test_movie_fields():
    fields = movie_item_fields({"title": "Heat", "store": "Al Pacino  Format: Blu-ray",
                                "categories": ["Movies & TV", "Featured Categories", "Action", "Heist"],
                                "price": 9.99})
    assert fields == {"subgenre": "Heist", "subgenre_family": "", "format": "Blu-ray", "price": 9.99}
    assert movie_item_fields({"title": "Die Hard [DVD]", "categories": []})["format"] == "DVD"
    assert movie_item_fields({"title": "Vegas", "main_category": "Prime Video"})["format"] == "Prime Video"


def test_game_fields():
    headset = game_item_fields({"title": "HyperX Cloud", "price": "49.99",
                                "categories": ["Video Games", "PC", "Accessories", "Headsets"]})
    assert headset == {"subgenre": "Headsets", "subgenre_family": "", "format": "accessory", "price": 49.99}
    game = game_item_fields({"title": "Halo [Digital Code]", "categories": ["Video Games", "Xbox One", "Games"]})
    assert game["format"] == "digital code" and game["subgenre"] == "" and game["price"] is None


def test_items_frame_keeps_missing_prices_as_none():
    frame = items_frame([{"item_id": 1, "title": "A", "genres": "", "price": None},
                         {"item_id": 2, "title": "B", "genres": "", "price": 5.0, "format": "Vinyl"}])
    records = frame.to_dict("records")
    assert records[0]["price"] is None and records[1]["price"] == 5.0
    assert records[0]["format"] == "" and records[1]["format"] == "Vinyl"
