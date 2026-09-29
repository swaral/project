"""Amazon Books and CDs & Vinyl (2023): more taste-driven media domains (Addendum v12).

Both come from the same McAuley-Lab "Amazon-Reviews-2023" release as the
games and Movies & TV domains (5-core, rating-only ratings plus the raw
per-category metadata) and are prepared the same way as Movies & TV: a
hash-selected fraction of users, re-5-cored (:func:`amazon_movies.write_subset`).

Two differences:

- The Books metadata is about 14 GB, so it is never stored. It is streamed
  from Hugging Face once (resuming with HTTP Range requests if the
  connection drops) and only the fields used here are cached, for the items
  that appear in the ratings file.
- Album titles alone rarely identify an album ('Greatest Hits'), so items
  are shown as '<title> by <artist or author>'. Reference rankers read the
  same string, so they see the same information as the LLM.

The genre is the first label in Amazon's category path that is in the
domain's fixed genre map ('CDs & Vinyl > Jazz > Bebop' -> Jazz). Store
sections (deals, record labels, imports) are not genres; items filed under
one usually carry the real genre one level down ('CDs & Vinyl > Today's
Deals in Music > Country'), which the scan picks up.
"""

from __future__ import annotations

import http.client
import json
import re
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from .amazon_games import _stream_download, load_ratings_file
from .amazon_movies import write_subset

HF_ROOT = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main"


@dataclass(frozen=True)
class MediaCategory:
    """One Amazon category prepared as a sampled domain."""

    domain: str
    hf_name: str  # category name in the Hugging Face file paths
    subdir: str  # directory under data/raw
    stem: str  # file-name stem of the raw and subset files
    noun: str  # what the prompt calls one item
    user_fraction: float  # chosen so the subset is about the size of Movies & TV
    genre_map: Mapping[str, str]  # category label -> canonical genre

    @property
    def ratings_url(self) -> str:
        return f"{HF_ROOT}/benchmark/5core/rating_only/{self.hf_name}.csv"

    @property
    def metadata_url(self) -> str:
        return f"{HF_ROOT}/raw/meta_categories/meta_{self.hf_name}.jsonl"

    @property
    def raw_ratings(self) -> str:
        return f"{self.stem}_ratings.csv"

    @property
    def compact_metadata(self) -> str:
        return f"{self.stem}_meta_compact.jsonl"

    @property
    def subset_ratings(self) -> str:
        return f"{self.stem}_ratings_subset.csv"

    @property
    def subset_items(self) -> str:
        return f"{self.stem}_items_subset.jsonl"


MUSIC_GENRES = (
    "Pop", "Rock", "Indie & Alternative", "Jazz", "Country", "International Music",
    "Classic Rock", "Classical", "Metal", "R&B", "Christian & Gospel", "Blues",
    "Dance & Electronic", "Rap & Hip-Hop", "Broadway & Vocalists", "Soundtracks", "Folk",
    "Opera & Classical Vocal", "Holiday & Wedding", "New Age", "Comedy & Spoken Word",
    "Latin Music", "Children's Music", "Reggae",
)
MUSIC_GENRE_MAP = {
    **{name: name for name in MUSIC_GENRES},
    # Labels used under the deals sections for the same genres.
    "Hard Rock & Metal": "Metal",
    "Alternative Rock": "Rock",
    "Dance & DJ": "Dance & Electronic",
}

# Amazon's top-level Books genres; store sections (Boxed Sets, Calendars,
# Large Print, deals, textbook rentals) are left out.
BOOK_GENRES = (
    "Literature & Fiction", "Children's Books", "Mystery, Thriller & Suspense",
    "Crafts, Hobbies & Home", "Arts & Photography", "Christian Books & Bibles",
    "Teen & Young Adult", "Biographies & Memoirs", "History", "Cookbooks, Food & Wine",
    "Science Fiction & Fantasy", "Politics & Social Sciences", "Business & Money", "Romance",
    "Comics & Graphic Novels", "Humor & Entertainment", "Reference", "Science & Math",
    "Computers & Technology", "Travel", "Religion & Spirituality", "Health, Fitness & Dieting",
    "Engineering & Transportation", "Sports & Outdoors", "Education & Teaching", "Self-Help",
    "Medical Books", "LGBTQ+ Books", "Parenting & Relationships", "Law", "Test Preparation",
)
BOOK_GENRE_MAP = {name: name for name in BOOK_GENRES}

CATEGORIES = {
    "amazon_books": MediaCategory(
        domain="amazon_books",
        hf_name="Books",
        subdir="amazon-books",
        stem="books",
        noun="book",
        user_fraction=0.10,
        genre_map=BOOK_GENRE_MAP,
    ),
    "amazon_music": MediaCategory(
        domain="amazon_music",
        hf_name="CDs_and_Vinyl",
        subdir="amazon-music",
        stem="music",
        noun="album",
        user_fraction=0.30,
        genre_map=MUSIC_GENRE_MAP,
    ),
}

_ASIN = re.compile(rb'"parent_asin":\s*"([^"]+)"')
_ROLE = re.compile(r"\s*\(.*$")


def creator(record: dict[str, object]) -> str:
    """First author or artist of a metadata record, or '' when unknown.

    Books carry an ``author`` object; music (and books without one) name
    contributors in ``store``, e.g. 'John Barry  (Composer),  Barry, John
    (Composer)    Format: Audio CD'.
    """

    author = record.get("author")
    if isinstance(author, dict) and str(author.get("name") or "").strip():
        name = str(author["name"])
    else:
        store = str(record.get("store") or "").split("Format:")[0]
        name = _ROLE.sub("", re.split(r"\)\s*,", store)[0])
    return clean_creator(name)


def clean_creator(name: str) -> str:
    """Whitespace-normalized name, or '' for placeholders that name nobody.

    Also applied to cached names, so tightening it needs no re-download.
    """

    name = " ".join(str(name).split())
    if name.lower() in ("", "none", "various", "various artists", "unknown") or name.startswith("Rated:"):
        return ""
    return name


def genre(categories: list[str], genre_map: Mapping[str, str]) -> list[str]:
    """First mapped genre along the category path below the root, as [genre] or []."""

    for label in categories[1:]:
        canonical = genre_map.get(" ".join(str(label).split()))
        if canonical:
            return [canonical]
    return []


def display_title(title: str, by: str) -> str:
    return f"{title} by {by}" if by else title


def _lines(url: str, *, timeout: int = 300, retries: int = 8) -> Iterator[bytes]:
    """Yield the lines of a remote file, resuming after dropped connections.

    ``offset`` counts only bytes of lines already yielded, so a resumed
    request restarts exactly after the last complete line. A connection that
    closes early can look like a normal end of file, so the bytes received
    are checked against the announced Content-Length and a short read is
    treated as a dropped connection.
    """

    offset, failures = 0, 0
    while True:
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        buffer = b""
        try:
            with urlopen(Request(url, headers=headers), timeout=timeout) as response:
                if offset and getattr(response, "status", 206) != 206:
                    raise OSError("server ignored the Range request; cannot resume")
                announced = response.headers.get("Content-Length")
                received = 0
                while chunk := response.read(1 << 22):
                    received += len(chunk)
                    buffer += chunk
                    *complete, buffer = buffer.split(b"\n")
                    for line in complete:
                        offset += len(line) + 1
                        yield line
                if announced is not None and received < int(announced):
                    raise http.client.IncompleteRead(b"", int(announced) - received)
            if buffer:
                yield buffer
            return
        except (OSError, http.client.HTTPException):
            failures += 1
            if failures > retries:
                raise
            time.sleep(min(60, 5 * failures))


def stream_metadata(url: str, asins: set[str], output: Path) -> int:
    """Cache title, creator and categories for ``asins`` from a JSONL metadata URL.

    Only lines whose ``parent_asin`` is wanted are parsed, which keeps a 14 GB
    stream fast. Written atomically, so an interrupted run leaves no cache.
    """

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_suffix(output.suffix + ".part")
    kept = 0
    with tmp_path.open("w", encoding="utf-8") as handle:
        for line in _lines(url):
            match = _ASIN.search(line)
            if match is None or match.group(1).decode() not in asins:
                continue
            record = json.loads(line)
            handle.write(json.dumps({
                "parent_asin": record["parent_asin"],
                "title": " ".join(str(record.get("title") or "").split()),
                "creator": creator(record),
                "categories": list(record.get("categories") or []),
            }, sort_keys=True) + "\n")
            kept += 1
    tmp_path.replace(output)
    return kept


def download_ratings(domain: str, data_dir: str | Path, *, force: bool = False) -> Path:
    """Download one category's 5-core rating-only CSV once; return its directory."""

    category = CATEGORIES[domain]
    root = Path(data_dir) / category.subdir
    if force or not (root / category.raw_ratings).is_file():
        _stream_download(category.ratings_url, root / category.raw_ratings)
    return root


def build_subset(
    domain: str,
    raw_dir: str | Path,
    *,
    user_fraction: float | None = None,
    seed: int = 0,
    min_interactions: int = 5,
    metadata_url: str | None = None,
) -> dict[str, object]:
    """Write the compact subset files next to the raw ratings; return a summary."""

    category = CATEGORIES[domain]
    fraction = category.user_fraction if user_fraction is None else user_fraction
    raw_dir = Path(raw_dir)
    raw = pd.read_csv(
        raw_dir / category.raw_ratings,
        dtype={"user_id": "string", "parent_asin": "string", "rating": "float64", "timestamp": "int64"},
    )
    asins = set(raw["parent_asin"].unique())

    compact = raw_dir / category.compact_metadata
    if not compact.is_file():
        stream_metadata(metadata_url or category.metadata_url, asins, compact)
    items: dict[str, dict[str, object]] = {}
    with compact.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["title"]:
                items[record["parent_asin"]] = {
                    "title": display_title(record["title"], clean_creator(record["creator"])),
                    "genres": genre(record["categories"], category.genre_map),
                }

    summary = {
        "domain": domain,
        "raw_ratings": int(len(raw)),
        "raw_users": int(raw["user_id"].nunique()),
        "raw_items": int(len(asins)),
        "items_with_title": len(items),
        "items_with_creator": sum(" by " in item["title"] for item in items.values()),
        **write_subset(
            raw, items, raw_dir / category.subset_ratings, raw_dir / category.subset_items,
            user_fraction=fraction, seed=seed, min_interactions=min_interactions,
        ),
    }
    return summary


def _dataset_directory(domain: str, data_dir: str | Path) -> Path:
    category = CATEGORIES[domain]
    root = Path(data_dir)
    for candidate in (root / category.subdir, root):
        if (candidate / category.subset_ratings).is_file():
            return candidate
    raise FileNotFoundError(
        f"{domain} subset not found under {root}. Run scripts/prepare_amazon_media.py first."
    )


def load_ratings(domain: str, data_dir: str | Path, *, id_map_output: str | Path | None = None) -> pd.DataFrame:
    """Subset ratings with integer IDs; same schema as the games domain."""

    path = _dataset_directory(domain, data_dir) / CATEGORIES[domain].subset_ratings
    return load_ratings_file(path, id_map_output)


def load_items(domain: str, data_dir: str | Path, item_id_map: dict[str, int]) -> pd.DataFrame:
    """Items as ``item_id``, ``title`` ('<title> by <creator>'), ``genres``."""

    rows = []
    path = _dataset_directory(domain, data_dir) / CATEGORIES[domain].subset_items
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["parent_asin"] in item_id_map:
                rows.append({
                    "item_id": item_id_map[record["parent_asin"]],
                    "title": record["title"],
                    "genres": "|".join(record["genres"]),
                })
    return pd.DataFrame(rows, columns=["item_id", "title", "genres"]).astype(
        {"item_id": "int32", "title": "string", "genres": "string"}
    )
