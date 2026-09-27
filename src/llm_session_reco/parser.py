"""Strict validation of LLM-produced candidate rankings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
import math


@dataclass(frozen=True)
class ParsedRanking:
    """Validated ranking result, including a research-safe error state."""

    position_scores: tuple[float, ...]
    ranked_positions: tuple[int, ...]
    ranked_item_ids: tuple[int, ...]
    parse_success: bool
    parse_error: str | None
    target_rank: int | None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["position_scores"] = list(self.position_scores)
        data["ranked_positions"] = list(self.ranked_positions)
        data["ranked_item_ids"] = list(self.ranked_item_ids)
        return data


def _failure(message: str) -> ParsedRanking:
    return ParsedRanking(
        position_scores=(),
        ranked_positions=(),
        ranked_item_ids=(),
        parse_success=False,
        parse_error=message,
        target_rank=None,
    )


def parse_ranking(
    raw_response: str,
    candidate_item_ids: Sequence[int],
    *,
    target_item_id: int | None = None,
) -> ParsedRanking:
    """Parse and strictly validate one LLM ranking response.

    A valid response must be a JSON object containing exactly one key,
    ``scores``. The scores must contain one finite number in the range 0..100
    for every candidate in the original presented order. The parser converts
    scores into a deterministic ranking (descending score, then ascending
    candidate position) and maps positions back to item IDs.

    Invalid responses are returned as structured failures and are never
    silently repaired.
    """

    expected = tuple(int(item_id) for item_id in candidate_item_ids)
    expected_set = set(expected)
    if len(expected_set) != len(expected):
        return _failure("candidate_item_ids contains duplicates")
    if not isinstance(raw_response, str) or not raw_response.strip():
        return _failure("response is empty")

    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return _failure(f"response is not valid JSON: {exc.msg}")

    if not isinstance(payload, dict):
        return _failure("JSON response must be an object")
    if set(payload) != {"scores"}:
        return _failure("JSON object must contain exactly scores")

    scores = payload["scores"]
    if not isinstance(scores, list):
        return _failure("scores must be a list")
    if len(scores) != len(expected):
        return _failure(
            f"scores must contain exactly {len(expected)} values; found {len(scores)}"
        )
    if any(
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or not 0 <= float(score) <= 100
        for score in scores
    ):
        return _failure("scores must contain only finite numbers in the range 0..100")

    position_scores = tuple(float(score) for score in scores)
    ranked_positions = tuple(
        sorted(
            range(1, len(expected) + 1),
            key=lambda position: (-position_scores[position - 1], position),
        )
    )
    ranked_item_ids = tuple(expected[position - 1] for position in ranked_positions)

    target_rank: int | None = None
    if target_item_id is not None:
        target_item_id = int(target_item_id)
        if target_item_id not in expected_set:
            return _failure("target_item_id is not present in candidate_item_ids")
        target_rank = ranked_item_ids.index(target_item_id) + 1

    return ParsedRanking(
        position_scores=position_scores,
        ranked_positions=ranked_positions,
        ranked_item_ids=ranked_item_ids,
        parse_success=True,
        parse_error=None,
        target_rank=target_rank,
    )
