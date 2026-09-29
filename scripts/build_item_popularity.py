"""Write the training-only item popularity table used by the reference baselines.

Standalone on purpose: it reproduces the popularity counts from the raw
ratings without re-running the full prepare pipeline, so an existing
data/processed/ directory does not have to be regenerated. The counts come
from the same leave-one-out training frame that candidate-pool construction
uses, so the popularity baseline never sees a held-out target.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_session_reco.baselines import build_item_popularity, write_item_popularity
from llm_session_reco.session_dataset import build_leave_one_out_training_ratings

DEFAULT_OUTPUTS = {
    "movielens": Path("data/processed/ml1m_item_popularity.json"),
    "amazon-games": Path("data/processed/amazon_games_item_popularity.json"),
    "amazon-movies": Path("data/processed/amazon_movies_item_popularity.json"),
    "amazon-film": Path("data/processed/amazon_film_item_popularity.json"),
    "amazon-tv": Path("data/processed/amazon_tv_item_popularity.json"),
    "amazon-books": Path("data/processed/amazon_books_item_popularity.json"),
    "amazon-music": Path("data/processed/amazon_music_item_popularity.json"),
}


def load_domain_ratings(domain: str, data_root: Path):
    if domain == "movielens":
        from llm_session_reco.movielens import load_ratings

        return load_ratings(data_root / "ml-1m")
    if domain == "amazon-games":
        from llm_session_reco.amazon_games import load_ratings

        return load_ratings(data_root / "amazon-games")
    if domain in ("amazon-movies", "amazon-film", "amazon-tv"):
        from llm_session_reco.amazon_movies import load_ratings
        from llm_session_reco.domains import AMAZON_MOVIES_MEDIA

        return load_ratings(data_root, media=AMAZON_MOVIES_MEDIA[domain.replace("-", "_")])
    if domain in ("amazon-books", "amazon-music"):
        from llm_session_reco.amazon_media import load_ratings

        return load_ratings(domain.replace("-", "_"), data_root)
    raise ValueError(f"unknown domain: {domain}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain", choices=sorted(DEFAULT_OUTPUTS), default="movielens"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or DEFAULT_OUTPUTS[args.domain]

    ratings = load_domain_ratings(args.domain, args.data_root)
    training_ratings = build_leave_one_out_training_ratings(ratings)
    item_popularity = build_item_popularity(training_ratings)
    write_item_popularity(item_popularity, output)

    print(
        f"domain={args.domain} "
        f"ratings={len(ratings)} training_ratings={len(training_ratings)} "
        f"items={len(item_popularity)} -> {output}"
    )


if __name__ == "__main__":
    main()
