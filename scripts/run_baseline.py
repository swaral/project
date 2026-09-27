"""Run controlled LLM recommendation trials over prepared MovieLens examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_session_reco.llm_client import (
    LLMRequestError,
    create_client,
)
from llm_session_reco.movielens import load_movies
from llm_session_reco.parser import parse_ranking
from llm_session_reco.prompts import (
    CONTEXT_VARIANT_DESCRIPTIONS,
    render_baseline_prompt,
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
    """Return trial IDs already written to an output file for resumable runs."""

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
                # An interrupted process may leave an incomplete final line.
                continue
            if isinstance(record, dict) and record.get("trial_id") is not None:
                trial_ids.add(str(record["trial_id"]))
    return trial_ids


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider", choices=["stub", "chat-completions", "openrouter"], default="stub"
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
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
        default=Path("data/processed/ml1m_baseline_trials.jsonl"),
    )
    parser.add_argument("--variant-id", default="baseline_scores_v1")
    parser.add_argument(
        "--context-variant-id",
        "--context-id",
        dest="context_variant_id",
        choices=sorted(CONTEXT_VARIANT_DESCRIPTIONS),
        default=None,
        help="semantic-context condition; omit for the existing title-only baseline",
    )
    parser.add_argument("--limit", type=int, default=None)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context_variant_id = args.context_variant_id or "context_title_v1"
    condition_id = args.variant_id
    if args.context_variant_id is not None:
        condition_id = f"{args.variant_id}::{context_variant_id}"
    examples = {
        str(record["session_id"]): record for record in _read_jsonl(args.examples)
    }
    candidate_records = _read_jsonl(args.candidates)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        candidate_records = candidate_records[: args.limit]

    movies = load_movies(args.data_root)
    item_titles = {
        int(item_id): str(title)
        for item_id, title in zip(movies["item_id"], movies["title"])
    }
    item_genres = {
        int(item_id): str(genres)
        for item_id, genres in zip(movies["item_id"], movies["genres"])
    }
    candidate_count = (
        len(candidate_records[0]["candidate_item_ids"])
        if candidate_records
        else 20
    )
    client = _client(args, candidate_count=candidate_count)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    existing_trial_ids = _existing_trial_ids(output) if args.resume else set()
    records_to_run: list[dict[str, object]] = []
    skipped_existing = 0
    for candidate_record in candidate_records:
        trial_id = f"{candidate_record['session_id']}::{condition_id}"
        if trial_id in existing_trial_ids:
            skipped_existing += 1
            continue
        records_to_run.append(candidate_record)

    successes = 0
    parse_failures = 0
    request_failures = 0

    output_mode = "a" if args.resume else "w"
    with output.open(output_mode, encoding="utf-8") as handle:
        for candidate_record in records_to_run:
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
            prompt = render_baseline_prompt(
                prefix_item_ids,
                candidate_item_ids,
                item_titles,
                variant_id=args.variant_id,
                context_variant_id=context_variant_id,
                item_genres=item_genres,
            )
            base_record: dict[str, object] = {
                "schema_version": (
                    "step8_context_scores_v1"
                    if args.context_variant_id is not None
                    else "step4_scores_v1"
                ),
                "trial_id": f"{session_id}::{condition_id}",
                "session_id": session_id,
                "prompt_variant_id": args.variant_id,
                "context_variant_id": context_variant_id,
                "prefix_item_ids": prefix_item_ids,
                "candidate_item_ids": candidate_item_ids,
                "target_item_id": target_item_id,
                "target_position_in_candidates": int(
                    candidate_record["target_position"]
                ),
                "prompt": prompt.to_dict(),
                "client_config": client.config,
            }

            try:
                response = client.generate(
                    prompt.system_message,
                    prompt.user_message,
                )
                parsed = parse_ranking(
                    response.raw_text,
                    candidate_item_ids,
                    target_item_id=target_item_id,
                )
                record = {
                    **base_record,
                    "request_success": True,
                    "request_error": None,
                    "model": response.model,
                    "request_id": response.request_id,
                    "latency_ms": response.latency_ms,
                    "usage": response.usage,
                    "raw_response": response.raw_text,
                    **parsed.to_dict(),
                }
                if parsed.parse_success:
                    successes += 1
                else:
                    parse_failures += 1
            except (LLMRequestError, ValueError, KeyError) as exc:
                request_failures += 1
                if not args.continue_on_error:
                    raise
                record = {
                    **base_record,
                    "request_success": False,
                    "request_error": str(exc),
                    "model": getattr(client, "model", None),
                    "request_id": None,
                    "latency_ms": None,
                    "usage": None,
                    "raw_response": None,
                    "position_scores": [],
                    "ranked_positions": [],
                    "ranked_item_ids": [],
                    "parse_success": False,
                    "parse_error": f"request_error: {exc}",
                    "target_rank": None,
                }

            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    print(
        json.dumps(
            {
                "output": str(output),
                "trials": len(records_to_run),
                "skipped_existing": skipped_existing,
                "valid_rankings": successes,
                "parse_failures": parse_failures,
                "request_failures": request_failures,
                "provider": args.provider,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
