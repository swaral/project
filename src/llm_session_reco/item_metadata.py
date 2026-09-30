"""Sub-genre and format/price item fields for the Experiment 2b contexts (Addendum v15).

Each Amazon domain module turns its raw metadata into the same four fields,
stored with every item in ``<domain>_items.jsonl``:

- ``subgenre``: the deepest category label below the item's genre (Books,
  Music, Film, TV) or product type (Games), e.g. "Bebop" in "CDs & Vinyl >
  Jazz > Bebop"; "" when the path stops at the genre.
- ``subgenre_family``: the labels between the genre and ``subgenre``
  (joined with " > "), "" when there are none.
- ``format``: how the item is sold (Paperback, Kindle Edition, Audio CD,
  Vinyl, DVD, Blu-ray, Prime Video, digital code, game, accessory...).
- ``price``: the listed price in US dollars, or None. Prices are those in
  the 2023 metadata snapshot, not at the time of each rating.

Only dataset fields are used, and none depends on any user's ratings.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

import pandas as pd

FIELD_NAMES = ("subgenre", "subgenre_family", "format", "price")
EMPTY_FIELDS = {"subgenre": "", "subgenre_family": "", "format": "", "price": None}

_STORE_FORMAT = re.compile(r"Format:\s*(.+)$")
# Labels below a genre that name a store section or nothing at all, e.g.
# "CDs $7 - $10", "Today's Deals", "Vinyl Store", "General".
_NOT_SUBGENRE = re.compile(r"\$|\bdeals?\b|\bstore\b|^general$", re.IGNORECASE)
_PRICE = re.compile(r"(\d+(?:\.\d+)?)")


def parse_price(value: object) -> float | None:
    """A positive price from a float or a string such as '$12.99', else None."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    match = _PRICE.search(str(value).replace(",", ""))
    if match is None:
        return None
    price = float(match.group(1))
    return price if price > 0 else None


def clean_format(text: object) -> str:
    """First phrase of a format label ('DVD-R  This title is...' -> 'DVD-R')."""

    label = re.split(r"\s{2,}", str(text or "").strip())[0].strip()
    return "" if label.lower() in ("", "none", "unknown binding", "unknown") else label


def store_format(store: object) -> str:
    """Format named in a ``store`` string: 'Band (Artist)  Format: Vinyl' -> 'Vinyl'."""

    match = _STORE_FORMAT.search(str(store or ""))
    return clean_format(match.group(1)) if match else ""


def subgenre_below(
    categories: Sequence[str], anchors: Mapping[str, str] | set[str] | frozenset[str]
) -> tuple[str, str]:
    """(subgenre, family) from the labels after the first anchor on the path.

    ``anchors`` are the labels that name a genre (or product type); the path
    below the first one found is the sub-genre part.
    """

    labels = [" ".join(str(label).split()) for label in categories]
    for index, label in enumerate(labels):
        if label in anchors:
            below = [x for x in labels[index + 1:] if x and not _NOT_SUBGENRE.search(x)]
            if not below:
                return "", ""
            return below[-1], " > ".join(below[:-1])
    return "", ""


def item_fields(subgenre: tuple[str, str], item_format: str, price: object) -> dict[str, object]:
    """The four stored fields, in the shape every items file uses."""

    return {
        "subgenre": subgenre[0],
        "subgenre_family": subgenre[1],
        "format": item_format,
        "price": parse_price(price),
    }


def items_frame(rows: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """Items table shared by every Amazon domain: IDs, title, genres, the four fields.

    ``price`` stays an object column so a missing price is written as JSON
    null rather than NaN.
    """

    rows = [{**EMPTY_FIELDS, **row} for row in rows]
    frame = pd.DataFrame(rows, columns=["item_id", "title", "genres", *FIELD_NAMES])
    frame = frame.astype({
        "item_id": "int32", "title": "string", "genres": "string",
        "subgenre": "string", "subgenre_family": "string", "format": "string",
    })
    frame["price"] = pd.Series([row["price"] for row in rows], dtype=object)
    return frame
