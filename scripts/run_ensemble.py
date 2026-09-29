"""Run the reliability-weighted prompt ensemble over prepared session examples.

For every session, scores are collected from several prompt/context variants
(the ensemble members), then combined two ways: a naive mean (a plain
self-consistency baseline) and Reliability-Weighted Rank Aggregation (RWRA,
the project's mitigation method -- see src/llm_session_reco/ensemble.py).
Each trial record keeps every member's raw response alongside both
aggregated rankings and the per-session drift score, so single-prompt,
naive-ensemble, and RWRA performance can all be recomputed from one file.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

from llm_session_reco.amazon_games import load_items as load_amazon_items
from llm_session_reco.ensemble import AggregatedRanking, EnsembleMember, combine_ensemble
from llm_session_reco.llm_client import (
    LLMRequestError,
    create_client,
)
from llm_session_reco.movielens import load_movies
from llm_session_reco.parser import parse_ranking
from llm_session_reco.prompts import FIELD_CONTEXT_VARIANTS, render_baseline_prompt

DEFAULT_MEMBERS: tuple[tuple[str, str], ...] = (
    ("baseline_scores_v1", "context_title_v1"),
    ("wording_direct_v1", "context_title_v1"),
    ("wording_preference_v1", "context_title_v1"),
    ("wording_detailed_v1", "context_title_v1"),
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(record)
    return records


def _existing_trial_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    trial_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and record.get("trial_id") is not None:
                trial_ids.add(str(record["trial_id"]))
    return trial_ids


def _existing_model_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    models: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            model = record.get("model")
            config = record.get("client_config")
            if model is None and isinstance(config, dict):
                model = config.get("model")
            if not isinstance(model, str) or not model.strip():
                raise ValueError(
                    f"{path}:{line_number} does not record a model; cannot safely resume"
                )
            models.add(model.strip())
    if len(models) > 1:
        raise ValueError(
            f"Cannot resume mixed-model output {path}: {', '.join(sorted(models))}"
        )
    return models


def _default_output_path(model_name: str, domain: str) -> Path:
    model_slug = model_name.lower()
    if model_slug.startswith("qwen2.5:"):
        model_slug = model_slug.removeprefix("qwen2.5:")
    model_slug = model_slug.removesuffix("-instruct")
    model_slug = re.sub(r"[^a-z0-9]+", "_", model_slug).strip("_") or "model"
    dataset_slug = "ml1m" if domain == "movielens" else domain
    return Path("data/processed") / f"{dataset_slug}_{model_slug}_ensemble_trials.jsonl"


def _repair_truncated_tail(path: Path) -> int:
    """Drop a partial final line left by a killed run; return bytes removed.

    Records are appended one JSON line at a time. If the process is killed
    mid-write (e.g. a Kaggle session hitting its time limit), the file ends in a
    fragment with no trailing newline. Appending on --resume would fuse the next
    record onto that fragment and corrupt both. Truncating back to the last
    complete line is safe: the fragment's trial never parsed, so its trial ID is
    not in the resume set and the session is simply re-run.
    """

    if not path.exists():
        return 0
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return 0
    keep = data.rfind(b"\n") + 1  # 0 when there is no complete line at all
    with path.open("r+b") as handle:
        handle.truncate(keep)
    return len(data) - keep


def presented_candidate_order(
    candidate_item_ids: list[int],
    *,
    session_id: str,
    variant_id: str,
    context_variant_id: str,
    shuffle_seed: int | None,
) -> list[int]:
    """Candidate order shown to one ensemble member.

    With ``shuffle_seed`` set, every (session, member) pair gets its own
    reproducible permutation, so a model that favours a list slot favours a
    different item under each prompt and the bias averages out in the
    ensemble. With ``None`` the stored pool order is kept unchanged.
    """

    presented = list(candidate_item_ids)
    if shuffle_seed is not None:
        random.Random(
            f"{shuffle_seed}:{session_id}:{variant_id}:{context_variant_id}"
        ).shuffle(presented)
    return presented


def scores_in_pool_order(
    presented_item_ids: list[int],
    presented_scores: tuple[float, ...],
    candidate_item_ids: list[int],
) -> tuple[float, ...]:
    """Map a member's scores from its presented order back to pool order."""

    score_by_item = dict(zip(presented_item_ids, presented_scores))
    return tuple(score_by_item[item_id] for item_id in candidate_item_ids)


def _client(args: argparse.Namespace, *, candidate_count: int):
    return create_client(
        args.provider,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout,
        json_mode=not args.no_json_mode,
        candidate_count=candidate_count,
    )


def _aggregated_ranking_to_dict(ranking: AggregatedRanking | None) -> dict[str, object] | None:
    if ranking is None:
        return None
    return {
        "strategy": ranking.strategy,
        "position_scores": list(ranking.position_scores),
        "ranked_positions": list(ranking.ranked_positions),
        "ranked_item_ids": list(ranking.ranked_item_ids),
        "target_rank": ranking.target_rank,
    }


def _parse_members(raw_members: list[str] | None) -> tuple[tuple[str, str], ...]:
    if not raw_members:
        return DEFAULT_MEMBERS
    parsed: list[tuple[str, str]] = []
    for value in raw_members:
        variant_id, separator, context_variant_id = value.partition(":")
        if not separator:
            raise ValueError(
                f"--member must use VARIANT_ID:CONTEXT_VARIANT_ID format, got {value!r}"
            )
        parsed.append((variant_id, context_variant_id))
    return tuple(parsed)


def _load_item_context(
    domain: str,
    data_root: Path,
    id_map_path: Path | None,
    items_file: Path | None = None,
) -> tuple[dict[int, str], dict[int, str], str]:
    if domain == "movielens":
        movies = load_movies(data_root)
        item_titles = {
            int(item_id): str(title)
            for item_id, title in zip(movies["item_id"], movies["title"])
        }
        item_genres = {
            int(item_id): str(genres)
            for item_id, genres in zip(movies["item_id"], movies["genres"])
        }
        return item_titles, item_genres, "movie"

    if domain == "amazon_games":
        if items_file is not None:
            # Read the already-extracted item records written by
            # prepare_amazon_games.py, instead of re-parsing the ~417 MB raw
            # metadata file -- avoids shipping that file to a cloud runner.
            item_titles = {}
            item_genres = {}
            for record in _read_jsonl(items_file):
                item_id = int(record["item_id"])
                item_titles[item_id] = str(record["title"])
                item_genres[item_id] = str(record["genres"])
            return item_titles, item_genres, "game"

        if id_map_path is None:
            raise ValueError(
                "--items-file or --id-map is required when --domain=amazon_games"
            )
        id_maps = json.loads(id_map_path.read_text(encoding="utf-8"))
        item_id_map = id_maps["item_id_map"]
        items = load_amazon_items(data_root, item_id_map)
        item_titles = {
            int(item_id): str(title)
            for item_id, title in zip(items["item_id"], items["title"])
        }
        item_genres = {
            int(item_id): str(genres)
            for item_id, genres in zip(items["item_id"], items["genres"])
        }
        return item_titles, item_genres, "game"

    raise ValueError(f"Unknown domain {domain!r}; expected movielens or amazon_games")


def _context_feature_builder(args: argparse.Namespace, item_titles, item_genres):
    """Build the Addendum v8 four-field context features from training data only."""

    from llm_session_reco.baselines import build_item_popularity
    from llm_session_reco.benchmark import ItemKNNRetriever
    from llm_session_reco.context_features import ContextFeatureBuilder
    from llm_session_reco.domains import load_domain
    from llm_session_reco.session_dataset import build_leave_one_out_training_ratings

    processed_dir = args.items_file.parent if args.items_file else Path("data/processed")
    data = load_domain(args.domain, args.data_root, processed_dir)
    training = build_leave_one_out_training_ratings(data.ratings)
    popularity = build_item_popularity(training)
    catalog = sorted(set(int(i) for i in data.ratings["item_id"].unique()) & set(item_titles))
    ordered_genres = {
        item: [g for g in str(genres).split("|") if g and g not in ("Video Games", "Games")]
        for item, genres in item_genres.items()
    }
    return ContextFeatureBuilder(
        training,
        item_titles,
        ordered_genres,
        ItemKNNRetriever(training, catalog, popularity),
        year_source="title" if args.domain == "movielens" else "first_seen",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider", choices=["stub", "chat-completions", "openrouter"], default="stub"
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--domain", choices=["movielens", "amazon_games"], default="movielens"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--id-map",
        type=Path,
        default=None,
        help="path to the amazon_games id-map JSON (needed only if --items-file "
        "is not given, since it falls back to parsing the raw metadata file)",
    )
    parser.add_argument(
        "--items-file",
        type=Path,
        default=None,
        help="path to a precomputed items JSONL (e.g. amazon_games_items.jsonl from "
        "prepare_amazon_games.py); avoids needing the raw metadata file on this machine",
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=Path("data/processed/ml1m_leave_one_out.jsonl"),
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/processed/ml1m_candidate_pools.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output JSONL path; defaults to a model-specific file in data/processed",
    )
    parser.add_argument(
        "--member",
        action="append",
        dest="members",
        help="VARIANT_ID:CONTEXT_VARIANT_ID; repeatable; defaults to the 4 wording "
        "variants at context_title_v1",
    )
    parser.add_argument("--min-valid-members", type=int, default=2)
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.7,
        help="long-tail-weighted blend: overall scale on the ensemble-trust weight "
        "(Llama4Rec-style, see EXPERIMENT_SPEC.md addendum)",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.3,
        help="long-tail-weighted blend: floor multiplier for long-prefix sessions",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="take the first N prepared candidate records, in file order",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="randomly sample N candidate records (seeded, reproducible) instead of "
        "taking the first N; mutually exclusive with --limit",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="random seed for --sample-size",
    )
    parser.add_argument(
        "--shuffle-candidates",
        action="store_true",
        help="show each ensemble member its own seeded permutation of the "
        "candidate pool, so list-position bias averages out across members",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help="seed for --shuffle-candidates",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--no-json-mode", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append output and skip trial IDs already present in the output file",
    )
    parser.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=None,
        help="stop cleanly before starting a new session once this much wall-clock "
        "time has elapsed; pair with --resume to continue in a later run (used to "
        "finish inside hosted-notebook session limits instead of being killed)",
    )
    return parser.parse_args()


def main() -> None:
    started_at = time.monotonic()
    args = parse_args()
    members_config = _parse_members(args.members)
    if args.max_runtime_minutes is not None and args.max_runtime_minutes <= 0:
        raise ValueError("--max-runtime-minutes must be positive")

    examples = {
        str(record["session_id"]): record for record in _read_jsonl(args.examples)
    }
    # Computed over the full dataset (not the possibly --limit-ed batch below) so a
    # session's long-tail weight is stable regardless of which subsample is run.
    prefix_lengths = [len(record["prefix_item_ids"]) for record in examples.values()]
    min_prefix_length = min(prefix_lengths) if prefix_lengths else 0
    max_prefix_length = max(prefix_lengths) if prefix_lengths else 0
    if args.limit is not None and args.sample_size is not None:
        raise ValueError("--limit and --sample-size are mutually exclusive")

    candidate_records = _read_jsonl(args.candidates)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        candidate_records = candidate_records[: args.limit]
    elif args.sample_size is not None:
        if args.sample_size < 1:
            raise ValueError("--sample-size must be positive")
        rng = random.Random(args.sample_seed)
        # Sort by session_id first so the sample is deterministic regardless of
        # the candidate-pool file's on-disk row order.
        ordered = sorted(candidate_records, key=lambda record: str(record["session_id"]))
        sample_size = min(args.sample_size, len(ordered))
        candidate_records = rng.sample(ordered, sample_size)
        candidate_records.sort(key=lambda record: str(record["session_id"]))

    item_titles, item_genres, domain_noun = _load_item_context(
        args.domain, args.data_root, args.id_map, args.items_file
    )
    feature_builder = None
    if any(context in FIELD_CONTEXT_VARIANTS for _, context in members_config):
        feature_builder = _context_feature_builder(args, item_titles, item_genres)
    candidate_count = (
        len(candidate_records[0]["candidate_item_ids"]) if candidate_records else 20
    )
    client = _client(args, candidate_count=candidate_count)

    output = args.output or _default_output_path(str(client.model), args.domain)
    if args.output is None and args.shuffle_candidates:
        # Keep shuffled and fixed-order trials in separate files so --resume
        # can never mix the two conditions.
        output = output.with_name(output.name.replace("_trials", "_shuffled_trials"))
    output.parent.mkdir(parents=True, exist_ok=True)
    repaired_tail_bytes = _repair_truncated_tail(output) if args.resume else 0
    if args.resume:
        existing_models = _existing_model_names(output)
        if existing_models and existing_models != {str(client.model)}:
            raise ValueError(
                f"Cannot resume {output} with model {client.model!r}; "
                f"the file contains {', '.join(sorted(existing_models))}"
            )
    existing_trial_ids = _existing_trial_ids(output) if args.resume else set()
    records_to_run: list[dict[str, object]] = []
    skipped_existing = 0
    for candidate_record in candidate_records:
        trial_id = str(candidate_record["session_id"])
        if trial_id in existing_trial_ids:
            skipped_existing += 1
            continue
        records_to_run.append(candidate_record)

    ensembles_combined = 0
    ensembles_failed = 0
    member_parse_failures = 0
    member_request_failures = 0

    output_mode = "a" if args.resume else "w"
    with output.open(output_mode, encoding="utf-8") as handle:
        completed = 0
        stopped_early = False
        for candidate_record in records_to_run:
            if (
                args.max_runtime_minutes is not None
                and time.monotonic() - started_at >= args.max_runtime_minutes * 60
            ):
                stopped_early = True
                break
            session_id = str(candidate_record["session_id"])
            if session_id not in examples:
                raise ValueError(f"No session example found for {session_id}")
            example = examples[session_id]
            target_item_id = int(candidate_record["target_item_id"])
            if target_item_id != int(example["target_item_id"]):
                raise ValueError(f"Target mismatch for {session_id}")

            prefix_item_ids = [int(item_id) for item_id in example["prefix_item_ids"]]
            candidate_item_ids = [
                int(item_id) for item_id in candidate_record["candidate_item_ids"]
            ]

            member_trials: list[dict[str, object]] = []
            ensemble_members: list[EnsembleMember] = []
            shuffle_seed = args.shuffle_seed if args.shuffle_candidates else None
            session_fields: dict[str, tuple] = {}
            for variant_id, context_variant_id in members_config:
                item_fields = None
                if context_variant_id in FIELD_CONTEXT_VARIANTS:
                    if context_variant_id not in session_fields:
                        session_fields[context_variant_id] = feature_builder.fields(
                            context_variant_id,
                            int(example["user_id"]),
                            prefix_item_ids,
                            candidate_item_ids,
                        )
                    item_fields = session_fields[context_variant_id]
                presented_item_ids = presented_candidate_order(
                    candidate_item_ids,
                    session_id=session_id,
                    variant_id=variant_id,
                    context_variant_id=context_variant_id,
                    shuffle_seed=shuffle_seed,
                )
                prompt = render_baseline_prompt(
                    prefix_item_ids,
                    presented_item_ids,
                    item_titles,
                    variant_id=variant_id,
                    context_variant_id=context_variant_id,
                    item_genres=item_genres,
                    domain_noun=domain_noun,
                    item_fields=item_fields,
                )
                try:
                    response = client.generate(prompt.system_message, prompt.user_message)
                    parsed = parse_ranking(
                        response.raw_text,
                        presented_item_ids,
                        target_item_id=target_item_id,
                    )
                    member_trials.append(
                        {
                            "variant_id": variant_id,
                            "context_variant_id": context_variant_id,
                            "request_success": True,
                            "request_error": None,
                            "raw_response": response.raw_text,
                            # Server-reported token usage and latency: evidence
                            # that no prompt was truncated to the context window,
                            # and the inputs to the cost-vs-gain analysis.
                            "usage": response.usage,
                            "latency_ms": response.latency_ms,
                            # position_scores/ranked_positions below refer to
                            # this presented order, not the stored pool order.
                            "presented_candidate_item_ids": presented_item_ids,
                            **parsed.to_dict(),
                        }
                    )
                    if not parsed.parse_success:
                        member_parse_failures += 1
                    ensemble_members.append(
                        EnsembleMember(
                            variant_id=variant_id,
                            context_variant_id=context_variant_id,
                            parse_success=parsed.parse_success,
                            # The ensemble aligns members by pool position.
                            position_scores=(
                                scores_in_pool_order(
                                    presented_item_ids,
                                    parsed.position_scores,
                                    candidate_item_ids,
                                )
                                if parsed.parse_success
                                else ()
                            ),
                        )
                    )
                except (LLMRequestError, ValueError, KeyError) as exc:
                    member_request_failures += 1
                    if not args.continue_on_error:
                        raise
                    member_trials.append(
                        {
                            "variant_id": variant_id,
                            "context_variant_id": context_variant_id,
                            "request_success": False,
                            "request_error": str(exc),
                            "raw_response": None,
                            "presented_candidate_item_ids": presented_item_ids,
                            "usage": None,
                            "latency_ms": None,
                            "parse_success": False,
                            "parse_error": f"request_error: {exc}",
                            "target_rank": None,
                        }
                    )
                    ensemble_members.append(
                        EnsembleMember(
                            variant_id=variant_id,
                            context_variant_id=context_variant_id,
                            parse_success=False,
                        )
                    )

            long_tail_bounds = (
                {
                    "prefix_length": len(prefix_item_ids),
                    "min_prefix_length": min_prefix_length,
                    "max_prefix_length": max_prefix_length,
                }
                if max_prefix_length > min_prefix_length
                else {}
            )
            result = combine_ensemble(
                ensemble_members,
                candidate_item_ids,
                target_item_id,
                min_valid_members=args.min_valid_members,
                beta1=args.beta1,
                beta2=args.beta2,
                **long_tail_bounds,
            )
            if result.failure_reason is None:
                ensembles_combined += 1
            else:
                ensembles_failed += 1

            record = {
                "schema_version": "ensemble_rwra_v1",
                "trial_id": session_id,
                "session_id": session_id,
                "domain": args.domain,
                "prefix_item_ids": prefix_item_ids,
                "candidate_item_ids": candidate_item_ids,
                "target_item_id": target_item_id,
                "target_position_in_candidates": int(
                    candidate_record["target_position"]
                ),
                "members": [
                    {"variant_id": v, "context_variant_id": c} for v, c in members_config
                ],
                "member_trials": member_trials,
                "candidate_shuffle": {
                    "enabled": args.shuffle_candidates,
                    "seed": shuffle_seed,
                },
                "client_config": client.config,
                "valid_member_count": result.valid_member_count,
                "member_weights": list(result.member_weights),
                "failure_reason": result.failure_reason,
                "drift": result.drift,
                "naive_mean": _aggregated_ranking_to_dict(result.naive),
                "reliability_weighted": _aggregated_ranking_to_dict(
                    result.reliability_weighted
                ),
                "long_tail_weighted": _aggregated_ranking_to_dict(
                    result.long_tail_weighted
                ),
                "long_tail_beta": result.long_tail_beta,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            # Flush per record so a killed process loses at most the session in
            # flight, never finished (already paid-for) sessions still buffered.
            handle.flush()
            completed += 1

    print(
        json.dumps(
            {
                "output": str(output),
                "trials": len(records_to_run),
                "completed_this_run": completed,
                "remaining": len(records_to_run) - completed,
                "stopped_early": stopped_early,
                "repaired_tail_bytes": repaired_tail_bytes,
                "elapsed_minutes": round((time.monotonic() - started_at) / 60, 2),
                "skipped_existing": skipped_existing,
                "ensembles_combined": ensembles_combined,
                "ensembles_failed": ensembles_failed,
                "member_parse_failures": member_parse_failures,
                "member_request_failures": member_request_failures,
                "provider": args.provider,
                "domain": args.domain,
                "members": list(members_config),
                "shuffle_candidates": args.shuffle_candidates,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
