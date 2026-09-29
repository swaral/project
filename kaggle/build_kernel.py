"""Build a pushable Kaggle kernel for the GPU pilot.

Embeds the current src/, scripts/ and pyproject.toml into kaggle/pilot_job.py
and writes kaggle/build/{run_pilot.py,kernel-metadata.json}. Push it with:

    python kaggle/build_kernel.py --username <kaggle-user>
    kaggle kernels push -p kaggle/build
    kaggle kernels status <kaggle-user>/llm-session-reco-pilot

    kaggle kernels output <kaggle-user>/llm-session-reco-pilot -p data/processed/kaggle_pilot

Pass --model, --conditions (fixed, shuffled), --pools (stress,
popularity_matched, retrieval) and --sessions to pilot another setup; any
setup other than the original fixed-order 3B run gets its own kernel, e.g.
llm-session-reco-pilot-7b-fixed-shuffled.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_SLUG = "llm-session-reco-pilot"
DEFAULT_MODEL = "qwen2.5:3b-instruct"


def _source_archive() -> str:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name in ("pyproject.toml", "src", "scripts"):
            tar.add(
                ROOT / name,
                arcname=name,
                filter=lambda info: None
                if "__pycache__" in info.name or info.name.endswith(".egg-info")
                else info,
            )
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag to pilot")
    parser.add_argument(
        "--conditions",
        default="fixed",
        help="comma-separated candidate-order conditions: fixed, shuffled",
    )
    parser.add_argument(
        "--pools",
        default="stress",
        help="comma-separated pools: stress, popularity_matched, retrieval",
    )
    parser.add_argument("--context", default="title", choices=("title", "genre"),
                        help="item context shown to every ensemble member")
    parser.add_argument("--sessions", type=int, default=300, help="sessions per domain and pool")
    parser.add_argument(
        "--experiment",
        default="wording",
        choices=("wording", "strategy", "context"),
        help="ensemble members: original wordings, 8 strategy prompts, or 4 field contexts",
    )
    parser.add_argument("--ladder-sessions", type=int, default=300,
                        help="sessions per domain on the random and attribute-matched levels")
    parser.add_argument("--domains", default="movielens,amazon_games",
                        help="comma-separated domains: movielens, amazon_games, amazon_movies; "
                        "a single domain is split over both GPUs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "kaggle" / "build")
    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    if not conditions or set(conditions) - {"fixed", "shuffled"}:
        parser.error("--conditions must list fixed and/or shuffled")
    pools = [p.strip() for p in args.pools.split(",") if p.strip()]
    allowed_pools = {"stress", "popularity_matched", "retrieval", "random", "attribute_matched"}
    if not pools or set(pools) - allowed_pools:
        parser.error(f"--pools must be drawn from {sorted(allowed_pools)}")
    if set(pools) - {"stress"} and "shuffled" not in conditions:
        parser.error("benchmark pools run shuffled only; include shuffled in --conditions")

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    if not domains or set(domains) - {"movielens", "amazon_games", "amazon_movies"}:
        parser.error("--domains must be drawn from movielens, amazon_games, amazon_movies")

    # The original fixed-order 3B run keeps the base kernel; every other setup
    # gets its own kernel so a new run never replaces earlier outputs.
    size = args.model.split(":")[-1].split("-")[0]
    suffix = f"{size}-{'-'.join(conditions)}"
    if pools != ["stress"]:
        suffix = f"{size}-" + "-".join(p.replace("popularity_", "") for p in pools)
    if args.context != "title":
        suffix += f"-{args.context}"
    if args.experiment != "wording":
        suffix = f"{size}-{args.experiment}-ladder"
    if domains != ["movielens", "amazon_games"]:
        suffix += "-" + "-".join(d.replace("amazon_", "") for d in domains)
    slug = KERNEL_SLUG
    if (args.model != DEFAULT_MODEL or conditions != ["fixed"] or pools != ["stress"]
            or args.context != "title" or args.experiment != "wording"
            or domains != ["movielens", "amazon_games"]):
        slug = f"{KERNEL_SLUG}-{suffix}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    job = (ROOT / "kaggle" / "pilot_job.py").read_text(encoding="utf-8")
    script = (
        job.replace(
            'os.environ.get("PILOT_MODEL", "qwen2.5:3b-instruct")', repr(args.model)
        ).replace(
            'os.environ.get("PILOT_CONDITIONS", "fixed").split(",")', repr(conditions)
        ).replace(
            'os.environ.get("PILOT_POOLS", "stress").split(",")', repr(pools)
        ).replace(
            'int(os.environ.get("PILOT_SESSIONS", "300"))', repr(args.sessions)
        ).replace(
            'os.environ.get("PILOT_CONTEXT", "title")', repr(args.context)
        ).replace(
            'os.environ.get("PILOT_EXPERIMENT", "wording")', repr(args.experiment)
        ).replace(
            'int(os.environ.get("PILOT_LADDER_SESSIONS", "300"))', repr(args.ladder_sessions)
        ).replace(
            'os.environ.get("PILOT_DOMAINS", "movielens,amazon_games").split(",")', repr(domains)
        )
        + f"\n\nPAYLOAD = {_source_archive()!r}\n\n"
        + 'if __name__ == "__main__":\n    main()\n'
    )
    (args.output_dir / "run_pilot.py").write_text(script, encoding="utf-8")
    metadata = {
        "id": f"{args.username}/{slug}",
        "title": "LLM Session Reco Pilot" + ("" if slug == KERNEL_SLUG else f" {suffix}"),
        "code_file": "run_pilot.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (args.output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.output_dir} ({len(script) // 1024} KiB)")


if __name__ == "__main__":
    main()
