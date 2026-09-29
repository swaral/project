"""Prompt construction for controlled LLM recommendation experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re


PROMPT_VARIANT_INSTRUCTION_TEMPLATES: dict[str, tuple[str, ...]] = {
    "baseline_scores_v1": (
        "Assign one relevance score from 0 to 100 to every candidate",
        "position according to how likely the user is to interact with that",
        "{noun} next. Higher scores mean stronger recommendations.",
    ),
    "wording_direct_v1": (
        "Give a direct next-item relevance score from 0 to 100 for every",
        "candidate position. Higher scores indicate that the user is more",
        "likely to interact with that {noun} next.",
    ),
    "wording_preference_v1": (
        "Estimate how well each candidate matches the user's {noun} preferences",
        "for the next interaction. Assign a score from 0 to 100, where higher",
        "scores mean a stronger match.",
    ),
    "wording_detailed_v1": (
        "Carefully compare the user's chronological history with every",
        "candidate and assign a score from 0 to 100 for likely next interaction.",
        "Use higher scores for stronger recommendations.",
    ),
    # Addendum v8: seven prompts that differ in reasoning strategy, not just
    # wording. Each is used alongside baseline_scores_v1 as an ensemble member.
    "next_step_v2": (
        "Look at the last few {noun}s the user chose. Score each candidate",
        "from 0 to 100 by how naturally it would come right after them, like",
        "the next part of a series or the obvious follow-up.",
    ),
    "long_term_taste_v2": (
        "Look at the whole history and find what the user likes again and",
        "again (type, style, theme). Score each candidate from 0 to 100 by how",
        "well it fits that lasting taste.",
    ),
    "closest_match_v2": (
        "For each candidate, find the one {noun} in the history it is most",
        "like. Score each candidate from 0 to 100 by how close that match is.",
    ),
    "rule_out_rank_v2": (
        "First give low scores (below 30) to candidates that clearly do not fit",
        "this user. Then score the remaining candidates from 30 to 100 by how",
        "well they fit.",
    ),
    "preference_enjoy_v2": (
        "Based on the user's history, how much would this user enjoy each",
        "candidate? Score from 0 (would not enjoy it) to 100 (would love it).",
    ),
    "preference_pick_now_v2": (
        "Imagine you are this user and you just chose the {noun}s above. How",
        "much would you want to pick each candidate next? Score from 0 (not at",
        "all) to 100 (right away).",
    ),
    "skip_risk_v2": (
        "For each candidate, think about how likely the user would skip or",
        "ignore it. Give 100 to candidates they would almost surely NOT skip,",
        "and 0 to ones they would surely skip.",
    ),
}

#: Experiment 1 (Addendum v8): the anchor prompt plus the seven strategy prompts.
STRATEGY_PROMPT_SET: tuple[str, ...] = (
    "baseline_scores_v1",
    "next_step_v2",
    "long_term_taste_v2",
    "closest_match_v2",
    "rule_out_rank_v2",
    "preference_enjoy_v2",
    "preference_pick_now_v2",
    "skip_risk_v2",
)

# Backward-compatible alias for the pre-multi-domain name.
PROMPT_VARIANT_INSTRUCTIONS = PROMPT_VARIANT_INSTRUCTION_TEMPLATES


CONTEXT_VARIANT_DESCRIPTION_TEMPLATES: dict[str, str] = {
    "context_title_v1": "{noun} titles only",
    "context_genre_v1": "{noun} titles and genres",
    "context_rich_v1": "{noun} titles, genres, and release year",
    # Addendum v8: four-field variants built from the datasets only
    # (fields come from context_features.ContextFeatureBuilder).
    "context_content_v2": "{noun} title, genres, series, and year",
    "context_crowd_v2": "{noun} title, average rating, audience size, and rating trend",
    "context_personal_v2": (
        "{noun} title, the user's own rating, how recent it is, and which of the "
        "user's most frequent genres it shares"
    ),
    "context_collab_v2": (
        "{noun} title, the past {noun} it is most often chosen with by other users, "
        "how strong that link is compared with the other candidates, and the recent "
        "past {noun} it most often follows"
    ),
}

#: Context variants whose fields are supplied by the caller via ``item_fields``.
FIELD_CONTEXT_VARIANTS: frozenset[str] = frozenset(
    {"context_content_v2", "context_crowd_v2", "context_personal_v2", "context_collab_v2"}
)

# Backward-compatible alias, rendered with the default domain noun ("movie").
CONTEXT_VARIANT_DESCRIPTIONS: dict[str, str] = {
    variant_id: template.format(noun="movie")
    for variant_id, template in CONTEXT_VARIANT_DESCRIPTION_TEMPLATES.items()
}


@dataclass(frozen=True)
class RenderedPrompt:
    """The exact messages sent to an LLM for one trial."""

    variant_id: str
    system_message: str
    user_message: str
    context_variant_id: str = "context_title_v1"

    def to_dict(self) -> dict[str, str]:
        return {
            "variant_id": self.variant_id,
            "system_message": self.system_message,
            "user_message": self.user_message,
            "context_variant_id": self.context_variant_id,
        }


def _title(item_id: int, item_titles: Mapping[int, str]) -> str:
    if item_id not in item_titles:
        raise KeyError(f"No title is available for item_id={item_id}")
    return str(item_titles[item_id]).replace("\r", " ").replace("\n", " ").strip()


def _genre(item_id: int, item_genres: Mapping[int, str]) -> str:
    if item_id not in item_genres:
        raise KeyError(f"No genre metadata is available for item_id={item_id}")
    value = str(item_genres[item_id]).replace("\r", " ").replace("\n", " ").strip()
    return value or "(unknown)"


def _release_year(title: str) -> str:
    match = re.search(r"\((\d{4})\)\s*$", title)
    return match.group(1) if match else "unknown"


def _item_line(
    index: int,
    item_id: int,
    item_titles: Mapping[int, str],
    *,
    context_variant_id: str,
    item_genres: Mapping[int, str] | None,
    fields: Mapping[int, str] | None = None,
) -> str:
    title = _title(item_id, item_titles)
    if context_variant_id in FIELD_CONTEXT_VARIANTS:
        # Items without fields (older history beyond the window) show the title only.
        extra = fields.get(item_id) if fields else None
        return f"{index}. title={title}" + (f"; {extra}" if extra else "")
    if context_variant_id == "context_title_v1":
        context = f"title={title}"
    else:
        if item_genres is None:
            raise ValueError(
                f"item_genres is required for context variant {context_variant_id}"
            )
        genre = _genre(item_id, item_genres)
        context = f"title={title}; genres={genre}"
        if context_variant_id == "context_rich_v1":
            context += f"; release_year={_release_year(title)}"
    return f"{index}. {context}"


def render_baseline_prompt(
    prefix_item_ids: Sequence[int],
    candidate_item_ids: Sequence[int],
    item_titles: Mapping[int, str],
    *,
    variant_id: str = "baseline_scores_v1",
    context_variant_id: str = "context_title_v1",
    item_genres: Mapping[int, str] | None = None,
    domain_noun: str = "movie",
    item_fields: tuple[Mapping[int, str], Mapping[int, str]] | None = None,
) -> RenderedPrompt:
    """Render the fixed, neutral baseline recommendation prompt.

    The target item is intentionally not an argument. Only the session prefix
    and candidate pool are rendered, so the target cannot leak into the prompt.
    Candidate order is preserved exactly as supplied by Step 3.
    """

    prefix = [int(item_id) for item_id in prefix_item_ids]
    candidates = [int(item_id) for item_id in candidate_item_ids]
    if not prefix:
        raise ValueError("prefix_item_ids cannot be empty")
    if not candidates:
        raise ValueError("candidate_item_ids cannot be empty")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate_item_ids must be unique")
    if variant_id not in PROMPT_VARIANT_INSTRUCTION_TEMPLATES:
        supported = ", ".join(sorted(PROMPT_VARIANT_INSTRUCTION_TEMPLATES))
        raise ValueError(
            f"Unknown prompt variant {variant_id!r}; supported variants: {supported}"
        )
    if context_variant_id not in CONTEXT_VARIANT_DESCRIPTION_TEMPLATES:
        supported = ", ".join(sorted(CONTEXT_VARIANT_DESCRIPTION_TEMPLATES))
        raise ValueError(
            f"Unknown context variant {context_variant_id!r}; supported variants: {supported}"
        )
    candidate_count = len(candidates)
    field_context = context_variant_id in FIELD_CONTEXT_VARIANTS
    if field_context and item_fields is None:
        raise ValueError(f"item_fields is required for context variant {context_variant_id}")
    history_fields, candidate_fields = item_fields if field_context else (None, None)

    history_lines = [
        _item_line(
            index,
            item_id,
            item_titles,
            context_variant_id=context_variant_id,
            item_genres=item_genres,
            fields=history_fields,
        )
        for index, item_id in enumerate(prefix, start=1)
    ]
    candidate_lines = [
        _item_line(
            index,
            item_id,
            item_titles,
            context_variant_id=context_variant_id,
            item_genres=item_genres,
            fields=candidate_fields,
        )
        for index, item_id in enumerate(candidates, start=1)
    ]
    evidence = (
        "the item information shown and the history"
        if field_context
        else f"the {domain_noun} titles and history"
    )

    system_message = (
        f"You are a precise next-item {domain_noun} recommendation system. "
        "Rank only the numbered candidate positions and follow the requested "
        "JSON format exactly."
    )
    instruction_lines = [
        line.format(noun=domain_noun)
        for line in PROMPT_VARIANT_INSTRUCTION_TEMPLATES[variant_id]
    ]
    context_description = CONTEXT_VARIANT_DESCRIPTION_TEMPLATES[
        context_variant_id
    ].format(noun=domain_noun)
    user_message = "\n".join(
        [
            f"The following {domain_noun}s were previously interacted with by the user,",
            "listed in chronological order:",
            "",
            *history_lines,
            "",
            f"Item context format: {context_description}.",
            "",
            *instruction_lines,
            "",
            f"Candidate positions to rank (exactly {candidate_count} candidates):",
            *candidate_lines,
            "",
            "Rules:",
            f"- Return exactly {candidate_count} numeric scores in the scores array.",
            f"- Array entry 1 scores candidate position 1, entry 2 scores position 2, and so on through position {candidate_count}.",
            f"- Use only {evidence} for relevance; do not use the position number as the score.",
            "- Score every candidate; do not stop early or return a top-k answer.",
            "- Return only one JSON object; do not use Markdown or explanations.",
            "- The JSON object must have exactly this shape:",
            '{"scores": [50, 25, 75]}',
            "- The example above only illustrates the key name; for this request, "
            f"the scores array must contain all {candidate_count} numeric scores.",
        ]
    )
    return RenderedPrompt(
        variant_id=variant_id,
        system_message=system_message,
        user_message=user_message,
        context_variant_id=context_variant_id,
    )
