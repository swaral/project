# LLM-Based Session Recommendation — Experiment Contract

This document is the contract for the first experiment. No implementation or
dataset preprocessing should begin until the decisions below are confirmed.

## Research question

How sensitive is an LLM-based session recommender to changes in prompt wording
and semantic context when the session, candidate set, model, and decoding
configuration are held constant?

## Unit of evaluation

One trial consists of:

1. A chronological session prefix.
2. One held-out next item as the target.
3. A fixed candidate pool containing the target.
4. One prompt condition.
5. One LLM response parsed into a ranked list.

## Initial scope

| Decision | Initial value | Status |
|---|---|---|
| Dataset | MovieLens-1M | Confirmed |
| Task | Next-item session recommendation | Confirmed |
| Candidate-pool size | 20 items | Confirmed and generated |
| Baseline prompt | Minimal, neutral candidate-scoring recommendation prompt (`baseline_scores_v1`) | Defined |
| Sensitivity factors | Wording and semantic-context perturbations | Wording variants defined; semantic context next |
| Metrics | HR@1, HR@5, HR@10, NDCG@5, NDCG@10 | Defined and implemented |
| Evaluation split | Chronological train/validation/test split | To confirm |

## Frozen baseline configuration

The first local baseline is frozen as follows:

- Ollama version: `0.32.15`.
- Model tag: `qwen2.5:3b-instruct`.
- Ollama model ID observed on 2026-08-21: `357c53fb659c`.
- Endpoint: `http://localhost:11434/v1`.
- Temperature: `0`.
- Top-p: `1`.
- Maximum output tokens: `256`.
- JSON mode: enabled.
- Request timeout: `600` seconds.
- Prompt variant: `baseline_scores_v1`.
- Trial schema: `step4_scores_v1`.

The model sees movie titles and numbered candidate positions. It does not need
to reproduce MovieLens item IDs. The model returns exactly one numeric relevance
score for each candidate position. The parser sorts scores in descending order,
breaks ties by ascending candidate position, maps positions back to the recorded
candidate item IDs, and never silently repairs failures.

## Candidate-pool policy

- Candidate popularity is calculated from training interactions only.
- Each pool contains exactly 20 unique items: the target plus 19 negatives.
- Negative items cannot already occur in the session prefix.
- Popularity ties are broken by ascending item ID.
- Target positions are balanced deterministically across positions 0 through 19.

## Variables

### Held constant in every comparison

- User/session examples.
- Target item and candidate pool.
- Item identifiers and item metadata.
- LLM name and exact version.
- System and user message structure, except for the tested factor.
- Temperature, top-p, maximum output tokens, and response format.
- Candidate-score output encoding and strict parser rules.
- Prompt order and evaluation code.

### Changed one factor at a time

- Prompt wording, or
- Semantic context supplied to the model.

Each variant must have a stable identifier, such as `baseline_v1` or
`wording_v2`, and its full prompt must be saved with the trial output.

### Step 6 wording conditions

The first sensitivity comparison changes only the three-line scoring
instruction in the user message. The history, candidate titles and order,
JSON rules, model, decoding settings, parser, and trial data remain fixed.

- `baseline_scores_v1`: neutral next-interaction scoring instruction.
- `wording_direct_v1`: direct next-item scoring instruction.
- `wording_preference_v1`: candidate-to-preference matching instruction.
- `wording_detailed_v1`: careful history-to-candidate comparison instruction.

The implementation rejects unknown variant IDs so an accidental prompt change
cannot be recorded as a controlled condition.

### Step 8 semantic-context conditions

The second sensitivity comparison keeps the baseline scoring instruction fixed
and changes only how movie information is presented to the model. The same
session prefixes, candidate pools, model, decoding settings, output schema,
parser, and evaluator are reused for every condition.

- `context_title_v1`: movie titles only.
- `context_genre_v1`: movie titles plus MovieLens genres.
- `context_rich_v1`: movie titles, MovieLens genres, and release year extracted
  from the title metadata.

The context condition is stored separately from the wording condition in every
trial record. This makes it possible to compare wording effects and context
effects without treating them as the same experimental factor.

## Required record for every trial

- Dataset and split identifiers.
- Session prefix and target item.
- Candidate list in presented order.
- Candidate-position scores, ranked candidate positions, and the corresponding
  mapped item IDs.
- Prompt-variant identifier and complete rendered prompt.
- Model identifier and decoding parameters.
- Raw model response.
- Parsed ranked list.
- Parsing success/error.
- HR/NDCG results and timestamp.

## First acceptance checks

- [ ] The target item is present in every candidate pool.
- [ ] The target item is not exposed as the labeled target or in the session
      prefix; it appears only as an unlabeled member of the candidate pool.
- [ ] Candidate order is recorded and can be reproduced.
- [ ] Valid model output contains exactly one finite score in the range 0..100
      for every candidate position.
- [ ] Every prompt variant changes only its intended factor.
- [ ] A malformed or incomplete model response is retained and marked as a
      parsing failure rather than silently discarded.

## Decisions to confirm before Step 2

1. Use MovieLens-1M as the first dataset. **Confirmed.**
2. Use a 20-item candidate pool. **Confirmed; construction is next.**
3. Evaluate next-item recommendation with chronological splitting. **Initial working choice.**
4. Start with one fixed LLM and deterministic decoding where available. **Confirmed: Ollama `0.32.15`, `qwen2.5:3b-instruct`, temperature `0`, top-p `1`, maximum output tokens `256`.**

## Addendum v2: Reliability-Weighted Ensembling and Cross-Domain Extension

The wording/context sensitivity pilot above (Steps 1-8) diagnosed the problem
but did not propose or validate a fix, and stayed within a single domain.
This addendum extends the contract to a mitigation method, a second domain,
and a statistical evaluation protocol, directly addressing that gap.

### Research question

Does reliability-weighted self-consistency ensembling recover PO4ISR++-style
robustness on a small local LLM (Qwen2.5-3B), across two domains, without an
expensive frontier-model-driven fusion step?

### Related work and honest positioning

Self-consistency prompting (sampling multiple reasoning traces and
aggregating by vote), prompt ensembling (e.g. PREFER: Prompt Ensemble
Learning via Feedback-Reflect-Refine), and confidence/accuracy-weighted
ensemble combination for recommenders are established techniques. This
project does not claim to invent ensembling or weighted aggregation. The
contribution is applying and evaluating these techniques in a combination
that has not been tested elsewhere: (a) on a resource-constrained local
open-weight model rather than a frontier API model, (b) against PO4ISR-style
semantic-drift diagnosis specifically, (c) with a per-session (not only
aggregate) drift score, and (d) across two domains with no domain-specific
tuning or fusion step. The write-up must cite the self-consistency and
prompt-ensemble literature explicitly rather than presenting Reliability-
Weighted Rank Aggregation (RWRA) as a wholly new algorithm.

### Mitigation method: Reliability-Weighted Rank Aggregation (RWRA)

Implemented in `src/llm_session_reco/ensemble.py`.

1. **Ensemble members**: for each session, render and score candidates under
   `k` prompt/context variant combinations (the "ensemble"). Default
   configuration (`scripts/run_ensemble.py`'s `DEFAULT_MEMBERS`): the 4
   existing wording variants at `context_title_v1` (`k=4`). A `k=12`
   wording x context ensemble is a planned follow-up ablation, not the
   default, to keep the first full run tractable on local CPU inference.
2. **Naive-mean aggregation** (self-consistency baseline): the elementwise
   mean of each member's per-candidate-position score
   (`aggregate_naive_mean`).
3. **Reliability weights**: for each member, the mean Kendall's tau between
   its ranking and every other member's ranking for that session, shifted
   from `[-1, 1]` to `[0, 2]` and normalized to sum to 1
   (`compute_reliability_weights`). A member whose ranking disagrees with
   the rest of the ensemble for this specific session is downweighted,
   without requiring any labeled calibration data.
4. **RWRA aggregation**: the weighted combination of member scores using
   the reliability weights (`aggregate_reliability_weighted`).
5. **Failure handling**: members that fail to parse are excluded, never
   silently repaired (same principle as the Step 4 parser). If fewer than
   `min_valid_members` (default 2) members parse successfully, both
   aggregations are recorded as a structured failure for that session.

### Drift score

Per-session ensemble disagreement, computed once per trial
(`stability.pairwise_agreement`): mean pairwise Kendall's tau and mean
pairwise Jaccard@5/@10 across all valid ensemble members' rankings. This
operationalizes "semantic drift" as a per-instance, inspectable value, rather
than only inferring it from an aggregate performance drop as the reference
paper does.

### Second domain: Amazon Video Games (2023)

Source: McAuley-Lab `Amazon-Reviews-2023` on Hugging Face
(`benchmark/5core/rating_only/Video_Games.csv`, ~47 MB ratings; `raw/
meta_categories/meta_Video_Games.jsonl`, ~417 MB metadata -- no smaller
metadata-only file is published for this category). 5-core filtered (users
and items with >= 5 interactions), matching the "eligible users have enough
history" spirit of `min_history_length`. Loader: `src/llm_session_reco/
amazon_games.py`; preparation script: `scripts/prepare_amazon_games.py`.

String reviewer IDs / ASINs are remapped to deterministic dense integer IDs
(sorted-string -> index) so the existing integer-ID pipeline
(`session_dataset.py`, `parser.py`, `prompts.py`) works unchanged. The
mapping is persisted to `data/processed/amazon_games_id_map.json` for
traceability. Product `categories` stand in for MovieLens `genres` in the
`context_genre_v1`/`context_rich_v1` conditions. `prompts.py`'s
`domain_noun` parameter is set to `"game"` for this domain (default
`"movie"` for MovieLens is unchanged, so existing wording/context variant
text and tests are unaffected).

### Sample size

- Single-prompt conditions (1 LLM call/session): full dataset, as already
  planned for Steps 6-8.
- Ensemble conditions (`k` LLM calls/session): a ~200-session validation run
  per dataset first, to sanity-check timing and results; then a stratified
  sample of ~1,000-1,200 sessions per dataset for the numbers reported in
  the paper. Full-dataset ensemble runs are optional if time/compute allow.

### Statistical testing protocol

Implemented in `src/llm_session_reco/statistics.py`, run via
`scripts/evaluate_ensemble.py`:

- Three conditions compared per dataset: `single_prompt_baseline` (the
  first ensemble member alone), `naive_mean_ensemble`, `reliability_weighted`.
- Comparisons restricted to sessions valid (successfully parsed/aggregated)
  under both compared conditions.
- Metric: reciprocal rank (`1/target_rank`, 0 on failure).
- Paired bootstrap confidence interval (`paired_bootstrap_ci`, default 2000
  resamples) and paired Wilcoxon signed-rank test (`wilcoxon_signed_rank`)
  for each pairwise comparison.
- HR@{1,5,10} and NDCG@{5,10} per condition via the shared
  `src/llm_session_reco/metrics.py` (also used by the existing
  `evaluate_baseline.py`, so single-prompt and ensemble metrics are always
  computed identically).

### Decisions confirmed for this addendum

1. Ensemble scope: wording-only (`k=4`), context ensembling deferred.
   **Confirmed.**
2. Second domain: Amazon Video Games (2023), 5-core rating-only.
   **Confirmed.**
3. Ensemble-run sample size: ~200-session validation, then ~1,000-1,200
   stratified sample per dataset. **Confirmed as the default; full-dataset
   run optional.**
4. Model-scaling ablation (Qwen2.5-7B): optional stretch, not required for
   the core contribution. **Deferred.**

## Addendum v3: Long-Tail-Aware Weighting (adapted from Llama4Rec)

Implemented in `src/llm_session_reco/ensemble.py`, wired through
`scripts/run_ensemble.py` and `scripts/evaluate_ensemble.py`.

### Source and honest positioning

Adapted from Luo et al., "Integrating Large Language Models Into
Recommendation via Mutual Augmentation and Adaptive Aggregation"
(Llama4Rec), IEEE JSTSP, 2026. Llama4Rec blends a fine-tuned LLM's score
with a *separately trained conventional recommender's* score, weighted by
how sparse a *user's* interaction history is (their long-tail coefficient
ℓ_u = log(N(u)+1) and adaptive weight α_u, Eq. 4/6). This project has no
second trained model, so the same mechanism is repurposed: it blends the
**RWRA ensemble** with the **single baseline prompt** (`members[0]`),
weighted by how short *this session's prefix* is. The write-up must state
this substitution explicitly (conventional-model-vs-LLM blend -> single-
prompt-vs-ensemble blend) rather than presenting it as an unmodified reuse
of Llama4Rec.

### Mechanism

- `long_tail_coefficient(prefix_length) = log(prefix_length + 1)` (Eq. 4
  analogue).
- `long_tail_weight(prefix_length, min_prefix_length, max_prefix_length,
  beta1=0.7, beta2=0.3)`: session-level trust in the ensemble, computed the
  same way as Llama4Rec's α_u (Eq. 6) but over session prefix length instead
  of per-user total interaction count. `min_prefix_length`/
  `max_prefix_length` are computed once over the full prepared dataset (not
  a `--limit`-ed subsample), so a session's weight is stable across runs.
- `aggregate_long_tail_weighted(rwra_scores, baseline_scores, weight)`:
  `weight * rwra + (1 - weight) * baseline`, the same linear-interpolation
  structure as Llama4Rec's Eq. 5.
- Computed as a 4th condition (`long_tail_weighted`) in every
  `combine_ensemble` call where `members[0]` parsed successfully and dataset
  prefix-length bounds are available; `None` otherwise (never guessed at).
- `beta1`/`beta2` are fixed hyperparameters (CLI flags `--beta1`/`--beta2`
  on `run_ensemble.py`, default 0.7/0.3) -- not swept as a full grid, to
  keep this an ablation rather than a new hyperparameter study.

### Research question this adds

Does ensembling's benefit concentrate in long-tail (short-prefix) sessions,
the same population Llama4Rec targets -- and can a cheap history-length
heuristic capture that, without training a second model at all?

### Evaluation

`scripts/evaluate_ensemble.py` reports:

- `long_tail_weighted` as a 4th condition alongside the existing three, with
  `long_tail_weighted_vs_single_prompt_baseline` and
  `long_tail_weighted_vs_reliability_weighted` paired comparisons (bootstrap
  CI + Wilcoxon, same protocol as the existing comparisons).
- `stratified_by_history_length`: all four conditions' HR/NDCG recomputed
  separately for short/medium/long prefix-length tertiles (split by the
  actual distribution of prefix lengths in the evaluated records), so the
  research question above can be answered directly rather than inferred
  from an aggregate number.

### Decision confirmed for this addendum

Long-tail-aware weighting is implemented as **Option A** from the three
options considered (long-tail weighting / similar-session context prompt
augmentation / full two-model fusion with a trained conventional
recommender). Option A was chosen because it requires no new model, no new
LLM calls beyond what RWRA already makes, and no new data -- it reuses the
session prefix length already present in every trial record. Options B
(similar-session prompt context) and C (full two-model fusion) remain
possible follow-ups, not implemented here.

## Addendum v4: Non-LLM Reference Baselines (popularity and random)

### Motivation

Addenda v2 and v3 compare LLM conditions only (single prompt vs. naive
ensemble vs. RWRA vs. long-tail-weighted RWRA). That leaves the prior
question unanswered: *is this candidate-ranking task hard at all, or would a
trivial ranker match the model?* Two non-LLM reference rows answer it at zero
inference cost. They are **reference floors, not competitors**: neither ever
becomes a component of the proposed method, so the "no second trained model"
property of RWRA is unchanged.

### Definitions

- **Popularity baseline.** Rank the pool by descending training interaction
  count, ties broken by ascending item ID -- the same ordering used by
  `_item_popularity` in `session_dataset.py`. Counts come from the
  leave-one-out *training* frame, so the baseline never observes a held-out
  target. Items absent from training are assigned count 0 and rank last.
- **Random baseline (shuffled).** A per-session uniform shuffle seeded from
  `(random_seed, session_id)`, so a session's ordering is reproducible and
  independent of how many sessions precede it.
- **Random baseline (analytic).** The closed-form expectation for pool size
  `N`: `E[HR@k] = min(k, N) / N` and `E[NDCG@k] = (1/N) * sum_{r=1..min(k,N)}
  1/log2(r+1)`. For `N = 20` this is HR@1 = 0.050, HR@5 = 0.250,
  HR@10 = 0.500, NDCG@5 = 0.147423, NDCG@10 = 0.227178.

The analytic values are reported alongside the shuffled ones because the
shuffle is a Monte Carlo estimate: on 6,040 MovieLens sessions, seed 0 draws
HR@1 = 0.0419, roughly three standard errors below 0.05, while seeds 1-5 land
on 0.049-0.054. The analytic row fixes the floor exactly without selecting a
flattering seed.

### Measured values (computed on the existing candidate pools, no LLM calls)

| Condition | HR@1 | HR@5 | HR@10 | NDCG@10 | Mean rank of 20 |
|---|---|---|---|---|---|
| Random (analytic) | 0.0500 | 0.2500 | 0.5000 | 0.2272 | 10.50 |
| Popularity — MovieLens-1M (n=6,040) | 0.0061 | 0.0180 | 0.0353 | 0.0174 | 19.35 |
| Popularity — Amazon Video Games (n=94,762) | 0.0046 | 0.0140 | 0.0249 | 0.0127 | 19.56 |

### Interpretation, and the caveat that must be reported with it

Popularity lands roughly **14x below the random floor**: 5,667 of 6,040
MovieLens targets (93.8%) and 91,254 of 94,762 Amazon targets (96.3%) rank
last in their pool. This is a direct and intended consequence of the
candidate-pool policy above -- the 19 negatives in every pool *are* the
most popular eligible training items, which makes the pools
**popularity-adversarial by construction**, i.e. popularity-based hard
negatives.

Two conclusions follow, and both must be stated together:

1. The task cannot be solved by memorising item frequency, so any gain a
   model shows over the floor reflects session modelling rather than
   popularity priors.
2. **Random, not popularity, is the honest floor for this setup.** Quoting
   the popularity row alone would make any method look like it beats a
   classical recommender, when in fact the evaluation protocol was designed
   so popularity cannot win. Both rows are therefore always reported.

### Implementation

- `src/llm_session_reco/baselines.py`: `build_item_popularity`,
  `write_item_popularity`, `load_item_popularity`, `popularity_ranking`,
  `random_ranking`, `rank_of_target`, `expected_random_metrics`.
- `scripts/build_item_popularity.py`: writes the training-only popularity
  table per domain (`--domain movielens|amazon-games`). Standalone by design,
  so an existing `data/processed/` directory need not be regenerated.
- `scripts/evaluate_ensemble.py`: adds `--popularity-file` and `--random-seed`.
  The `random_baseline` row appears whenever trial records carry their
  candidate pool; `popularity_baseline` additionally requires the popularity
  table. Missing inputs omit the row rather than emitting a silently empty
  one. Two paired comparisons are added
  (`reliability_weighted_vs_random_baseline`,
  `reliability_weighted_vs_popularity_baseline`), and the summary schema
  becomes `ensemble_evaluation_v3` with a new `reference_baselines` block.

## Addendum v5: Hosted-GPU Execution Protocol (Kaggle)

### Context window (new frozen parameter)

The prepared sessions have **no history cap** (`max_history_length = None`), so
MovieLens prompts grow with a user's full rating history. Measured on the
3,000-session MovieLens sample that is actually run (estimated at ~3.5
characters per token; the notebook measures the real longest prompt with the
model's own tokenizer before the main run):

| Statistic | MovieLens-1M | Amazon Video Games |
|---|---|---|
| Prefix length, median / p90 / max | 96 / 395 / 1,849 | 6 / 13 / 427 |
| Est. prompt tokens, median / p90 / max | 1,462 / 4,497 / 19,571 | 781 / 948 / 8,320 |
| Sessions over a 2,048-token window | 39.6% | 0.4% |
| Sessions over a 4,096-token window | 13.2% | 0.1% |
| Sessions over a 32,768-token window | 0% | 0% |

Ollama does not reject an over-long prompt; it truncates it silently, so the
model loses instructions or candidates and the trial degrades without any
error. The context window is therefore frozen at **`OLLAMA_CONTEXT_LENGTH =
32768`**, within Qwen2.5-3B's native 32k window. Its KV cache at 32k is about
1.2 GB, negligible on a 15-16 GB GPU.

Keeping the full, uncapped history is a deliberate design choice, not an
oversight: capping it would change the prepared data and the long-tail weights.
It does mean long-history sessions test the 3B model at long context, so the
`stratified_by_history_length` analysis doubles as a check on whether long
context itself drives drift.

### Evidence recorded per model call

Every member trial now stores the server-reported `usage` (including
`prompt_tokens`) and `latency_ms`. This provides (1) per-call evidence that no
prompt reached the context limit, audited after the run alongside a scan of
the Ollama server logs, and (2) the inputs to the cost-versus-gain analysis.

### Crash-safe, resumable runs

- Each record is flushed as soon as it is written, so a killed process loses
  at most the session in flight.
- `--resume` first truncates any partial final line left by a kill, then skips
  completed trial IDs. The partial session is re-run.
- `--max-runtime-minutes` stops cleanly between sessions, so a run ends before
  a hosted notebook's hard time limit instead of being killed.

### Serving configuration

- Ollama is installed at the pinned version (`0.32.15`), and the model digest
  is checked against `357c53fb659c`. Any mismatch is warned about and
  recorded rather than hidden.
- One Ollama server runs per GPU with `OLLAMA_NUM_PARALLEL = 1`, so requests
  are processed one at a time exactly as in the local configuration; the two
  domains run in parallel on separate GPUs. `OLLAMA_KEEP_ALIVE = -1` keeps the
  model loaded for the whole run.
- Each run writes a timestamped `run_manifest_*.json` recording versions,
  digest, GPUs, settings, a SHA-256 fingerprint of the code, and progress. A
  continued run refuses to proceed if sample size, seed, model, or context
  length differ from the run it continues, and warns if the code changed.

## Addendum v6: Second Model Size

### Research question

Does Reliability-Weighted Rank Aggregation (RWRA) help Qwen2.5-3B and
Qwen2.5-7B equally, or does its effect change with model size? Both models
receive identical treatment so model size is the only experimental change.

### Frozen comparison setup

- First model: `qwen2.5:3b-instruct` (existing baseline; previously observed
  Ollama model ID `357c53fb659c`).
- Second model: `qwen2.5:7b-instruct`; record its digest/fingerprint after its
  first experimental run.
- Run order: finish the 3B experiment first, then run 7B.
- Both runs cover **both domains**: the same 3,000 MovieLens-1M sessions and
  the same 3,000 Amazon Video Games sessions, each sampled with seed `0`, so
  the model-size comparison keeps the study's cross-domain design. Both use
  the same candidate pools and item metadata, and the same four prompt members
  (`baseline_scores_v1`, `wording_direct_v1`, `wording_preference_v1`, and
  `wording_detailed_v1`).
- Both sizes must be served the same way: same provider, same quantization,
  same serving software. Do not run 3B on one backend (e.g. local Ollama) and
  7B on another (e.g. a hosted API), because model size would then be
  confounded with the serving stack. Record each model's name as its provider
  reports it and pass the names to `compare_models.py` with `--model-3b` /
  `--model-7b` if they differ from the Ollama tags.
- Both use a 32,768-token context window, temperature `0`, top-p `1`, and the
  same output schema and parser.
- Popularity and random reference baselines are model-independent. Compute
  them with the 3B evaluation and pass `--skip-reference-baselines` when
  evaluating 7B; use the same reference values in the combined report.

### Separate files and model safety

`scripts/run_ensemble.py` uses the model tag in the default output name, such
as `data/processed/ml1m_3b_ensemble_trials.jsonl` and
`data/processed/ml1m_7b_ensemble_trials.jsonl`. Pass `--model` explicitly for
each run. `scripts/evaluate_ensemble.py` rejects files containing multiple
model tags and records the model in each metrics JSON. By default each
metrics file is named from its input trial file, so the two runs remain
separate.

### Paired model comparison

Run `scripts/compare_models.py` once per domain after both trial files exist.
It matches rows by `session_id` and verifies that matched sessions have
identical histories, candidate lists, targets, and prompt members.

The **analysis sample** is every matched session where both models produced
the single-prompt and RWRA ranks being compared, the same availability rule
`evaluate_ensemble.py` uses. A session where RWRA succeeded despite one broken
prompt answer stays in. Requiring all four answers to parse would remove
exactly the sessions that are hardest for the weaker model, which biases the
comparison in 3B's favour and hides RWRA's tolerance of broken answers. The
stricter all-four-answers cohort is reported separately as
`complete_case_sensitivity`. If the two disagree, the difference is driven by
sessions with broken answers. Failure-rate comparisons use all matched
sessions with recorded member trials, so failed answers remain measurable.

The comparison reports:

- The per-session reciprocal-rank gain from single prompt to RWRA for each
  model, and a paired test of the difference in those gains.
- Kendall's tau drift and broken-answer rate compared between model sizes.
- Model metrics and paired RWRA comparisons within short, medium, and long
  history groups, using the same analysis sample.
- `3B + RWRA` against `7B single prompt`, using the four answers already
  collected for each model; this requires no extra model calls.

Each paired comparison uses the existing paired bootstrap confidence interval
and Wilcoxon signed-rank test from `src/llm_session_reco/statistics.py`.
Results are saved per domain beside the 3B input, as
`data/processed/ml1m_model_comparison.json` and
`data/processed/amazon_games_model_comparison.json`, so the two domains never
overwrite each other. The generated comparisons are the source for the report
and presentation updates; do not rewrite those documents until 7B results are
available.

## Addendum v7: Repaired Benchmark and Debiased Ensemble

### Why the benchmark changed

The 300-session pilots (results/pilot_*) showed that the original pools cannot
measure personalization:

- The 19 negatives are the most popular non-history items, so the target is
  the least popular candidate in ~95% of pools. A "pick the least popular
  item" rule reaches HR@1 = 0.94 (MovieLens) and 0.95 (Amazon). Qwen2.5-7B
  scores below random there because it prefers well-known items.
- On MovieLens, 42% of final ratings share a timestamp with the previous
  rating, so the target was chosen by item ID within that second; 18% of
  targets were rated 2 stars or lower.

The original pool is kept only as a shortcut diagnostic (`top_popular`).

### Clean targets (`scripts/build_benchmark.py`)

- Target: each user's final interaction, kept only if rated >= 4 stars and
  strictly later than the previous interaction. Tied final timestamps are
  excluded, because their order is unknown.
- Prefix: every earlier interaction, unchanged.
- Training frame: every user's final interaction is removed, so no evaluated
  target is ever seen by popularity counts, retrievers or baselines.
- Split: 30% validation / 70% test by a seeded hash of the session ID.
- Result: MovieLens 2,053 sessions (619 / 1,434); Amazon 74,017 sessions
  (22,337 / 51,680).

### Pools (identical candidate lists for every method)

- `popularity_matched` (controlled comparison): 19 negatives from the target's
  popularity neighbourhood; the number less popular than the target is drawn
  uniformly from 0..19, so the target's popularity rank carries no signal.
  Candidates are stored in a seeded random order.
- `retrieval` (practical system test): item-KNN cosine over the last 20
  history items, fitted on training data only; top 20 non-history items.
  Reported three ways: retriever recall@20; reranking quality on sessions
  where the retriever found the target; and end to end, where an unretrieved
  target is a miss (= recall x reranking quality). Inserting a missed target
  is kept only as a flagged diagnostic: the retriever favours popular items,
  so an inserted target is usually the least popular candidate and inverse
  popularity finds it (MovieLens MRR 0.686).
- `top_popular` (diagnostic only): the original construction on the clean
  examples.

### Shortcut checks before any GPU run (`scripts/check_benchmark.py`)

On every pool, test split: the target's popularity mid-rank distribution,
popularity and inverse-popularity rankers against random (flag when the whole
MRR interval is more than 0.02 from random), and the non-LLM baselines
(genre/platform overlap, title-word overlap, item-KNN, sequential transitions,
retrieval order). Results: results/benchmark_checks/.

### Debiased ensemble (`scripts/evaluate_debiased.py`)

Benchmark pools are always run with `--shuffle-candidates`. Aggregation:
z-score each member's scores within the session, subtract that member's slot
prior (its mean z-score per presented slot), then average.

Everything data-driven is fitted on validation sessions only:

- the slot priors;
- the preselected single prompt (the member with the best validation MRR).

Metrics are tie-aware (expected values under random tie-breaking): MRR, HR@1,
HR@5, NDCG@10. HR@10 is dropped: with 20 candidates random already scores 0.5.

### Pre-registered primary comparisons (per domain and pool, test split)

1. Debiased ensemble vs preselected single prompt.
2. Debiased ensemble vs naive mean of raw scores.

Test: paired sign-flip permutation test on the mean reciprocal-rank
difference, Holm-corrected across the two, with a paired bootstrap 95% CI.
The Wilcoxon signed-rank test is no longer primary: it tests a median shift
and disagreed with the mean in the pilots. Secondary comparisons (z-score mean
vs naive mean, naive mean vs single prompt, RWRA vs naive mean) are
Holm-corrected as a separate family. Every LLM result is reported next to the
non-LLM baselines on the same candidates.

## Addendum v8: Strategy Prompts, Four-Field Contexts, Difficulty Ladder

Recorded before any run. Model `qwen2.5:3b-instruct`, shuffled candidate
order, clean examples and validation/test split from Addendum v7.

### Difficulty ladder (same examples at every level)

- L1 easy, `random`: 19 negatives uniform over the catalog.
- L2 medium, `popularity_matched`: unchanged from Addendum v7 (pools are
  byte-identical, so earlier L2 results stay comparable).
- L3 hard, `attribute_matched`: popularity-matched negatives that also share
  a genre (MovieLens) or the platform (Amazon, read from the existing genre
  path) with the target. Targets without a usable attribute fall back to L2
  sampling and are flagged `attribute_matched: false`.

Sessions: L2 uses the same 500 per domain as the earlier benchmark pilots
(seed 0); L1 and L3 use a seeded 300-session subset of those, so every level
scores the same users.

### Experiment 1: strategy prompts (8 members, title context)

`baseline_scores_v1` plus `next_step_v2`, `long_term_taste_v2`,
`closest_match_v2`, `rule_out_rank_v2`, `preference_enjoy_v2`,
`preference_pick_now_v2`, `skip_risk_v2` (texts in `prompts.py`).

### Experiment 2: four-field contexts (4 members, baseline wording)

Built only from the datasets (no external metadata); statistics from the
leave-one-out training frame; candidates never show a rating by the user;
fields shown for the last 50 history items.

- `context_content_v2`: title, genres, series (from titles), year (MovieLens
  title; Amazon first year seen in training ratings).
- `context_crowd_v2`: title, average rating, audience tertile, rating trend.
- `context_personal_v2`: title, user's own rating, recency, overlap with the
  user's top-3 genres.
- `context_collab_v2`: title, history item most often co-chosen (cosine),
  link strength relative to the other candidates, recent history item it
  most often follows.

### Pre-registered comparisons (per experiment, domain and ladder level; test split)

Primary (Holm across the two): RWRA vs the validation-preselected single
prompt, and RWRA vs the naive mean. Also reported, Holm-corrected as their
own families: the Addendum v7 debiased-ensemble comparisons and the
secondary comparisons. Paired sign-flip test on mean reciprocal rank with a
bootstrap 95% CI; tie-aware metrics.

Descriptive: each member's test MRR, mean pairwise tau-b between members on
raw scores, and all methods split by target rarity (training-popularity
tertiles of the test targets: rare / moderate / popular).

## Addendum v9: Amazon Movies & TV as the Movie Domain

Recorded 2026-09-29, before any Movies & TV run.

### Why

MovieLens-1M has no usable consumption order: 53% of consecutive ratings by
a user share the same second, 89% fall within a minute, and 63% of users
rated their whole history within one day (bulk rating on sign-up). The
held-out "next" item is therefore the last one clicked on a rating page, not
the next one watched. Amazon Movies & TV 2023 (5-core, rating-only) has 0%
same-second gaps, 61% gaps over a day and a 12-day median gap, like the games
domain. MovieLens results are kept as a contrast domain without sequential
signal; nothing about them changes.

### Data (`scripts/prepare_amazon_movies.py`, `amazon_movies.py`)

- Source: McAuley-Lab Amazon-Reviews-2023, `benchmark/5core/rating_only/
  Movies_and_TV.csv` and `raw/meta_categories/meta_Movies_and_TV.jsonl`.
  Only dataset fields are used: title and `categories`.
- Items without a title in the metadata are dropped.
- Users: kept when `sha256("0:<user_id>")[:8] < 0.15 * 2^32` (about 15%,
  independent of library versions and row order), then the 5-core filter is
  re-applied until stable. The subset's SHA-256 is recorded locally and by
  the Kaggle job, and the two must match.
- Genres: `categories` mapped to a fixed canonical set (`GENRE_MAP`:
  Drama, Comedy, Documentary, Thriller, Horror, Action & Adventure, Sci-Fi &
  Fantasy, Romance, Animation, Kids & Family, Special Interest, Fitness,
  Music & Performing Arts, Sports, Western, War, Historical, International,
  Arthouse, Unscripted, LGBTQ, Faith & Spirituality). Formats, studios, store
  sections and mood tags are dropped.
- Everything downstream is unchanged from v7/v8: clean targets (final rating
  >= 4, strictly later than the previous interaction), 30/70
  validation/test hash split, training-only statistics, the three ladder
  levels (L3 matches on a shared canonical genre, as for MovieLens), the
  shortcut checks and the non-LLM reference rankers. Context `year` uses the
  first year seen in training ratings, as for games.

### Runs

Experiment 1 (8 strategy prompts) and Experiment 2 (4 four-field contexts)
exactly as in Addendum v8: same model, Ollama version, shuffled candidates,
L2 500 sessions and L1/L3 a seeded 300-session subset. With a single domain,
the Kaggle job splits each level's sessions over the two GPUs (alternate
sessions) and merges the trial files; sessions are independent, so this does
not change any result.

### Comparisons

The same pre-registered comparisons as Addendum v8, per experiment and
ladder level. The Movies & TV runs are an additional family: Holm correction
stays within each (experiment, domain, level) pair of primary comparisons.

## Addendum v10: Movies and TV as Separate Domains

Recorded 2026-09-29, before any Movies & TV run. Supersedes Addendum v9's
single combined domain for the runs; the combined `amazon_movies` files are
kept unchanged.

### Why

Films and TV series are different recommendation problems (a user who buys
season 2 of a show usually wants season 3), and mixing them in one pool lets
the item type itself act as a signal. The category is therefore split into
`amazon_film` (movies) and `amazon_tv` (TV).

### Labelling (`amazon_movies.media_type`)

Amazon has no movie/TV field, so each raw metadata record is labelled by
rules on dataset fields only, in order:

1. An explicit `TV`/`Television` category, or TV words in the title (season,
   complete series, series + number, episodes, miniseries, TV series/show),
   means TV; an explicit `Movies` category means movie. Both at once: unknown.
2. Otherwise run time from `details`: under 40 minutes is unknown (mostly
   single episodes and clips), 40 minutes to 4 hours is a movie, over 4 hours
   is TV unless the title marks a multi-film collection (unknown).
3. No run time: unknown.

Unknown items are dropped from both domains. Over the 115,101 titled raw
items: 70,550 movie, 19,830 TV, 24,721 unknown. The rules were tuned on a
hand-read random sample; remaining known errors are kids' TV discs between
40 minutes and 4 hours with no TV words in the title (labelled movie). A
hand-check of a fresh random sample of each label must be reported with the
results.

### Data (`scripts/prepare_amazon_movies.py --media movie|tv`)

Each domain keeps only its own items, then applies Addendum v9's subset rule
unchanged (the same hashed 15% of users, seed 0) and re-applies the 5-core
filter on its own interactions. The same user can appear in both domains.
Everything downstream is unchanged from v7-v9 (clean targets, 30/70 split,
training-only statistics, ladder levels with L3 matched on a shared
canonical genre, shortcut checks, reference rankers).

| | `amazon_film` | `amazon_tv` |
|---|---|---|
| Ratings / users / items | 162,705 / 13,187 / 13,211 | 30,307 / 3,177 / 3,104 |
| Items with a canonical genre | 6,182 | 1,543 |
| Gaps over one day | 48.6% | 46.5% |
| Clean sessions (validation / test) | 11,410 (3,437 / 7,973) | 2,879 (843 / 2,036) |
| L3 pools genre-matched | 45.1% | 48.5% |
| Retriever recall@20 | 6.4% | 36.6% |
| Subset ratings SHA-256 (prefix) | `6c8306d9` | `829a86b6` |

### Shortcut checks (test split, MRR; random = 0.180)

- TV targets are strongly predictable from the history: title-word overlap
  reaches 0.54-0.57 and item-KNN 0.56-0.60 at every ladder level (film:
  0.29 and 0.37-0.41), mostly because the next item is often another season
  of a show already in the history. TV results must be read against these
  non-LLM baselines, not against random.
- Both domains carry the recency shortcut found on the combined set: ranking
  by first year seen in training ("newest first") scores 0.25-0.26 on film
  and 0.24-0.25 on TV at L2 and L3. `context_content_v2` (first seen year)
  and `context_crowd_v2` (trend) expose this signal, so any gain from those
  two contexts must be compared with the newest-first ranker.

### Runs

Experiment 1 and Experiment 2 as in Addendum v8, on both domains in one
Kaggle kernel (`build_kernel.py --domains amazon_film,amazon_tv`), one domain
per GPU. Session counts per level are as in v8.

### Comparisons

The Addendum v8 pre-registered comparisons, per experiment, domain and
ladder level; Holm correction stays within each (experiment, domain, level)
pair of primary comparisons.

## Addendum v11: Recency Baseline

Recorded 2026-09-29, before any Film or TV run.

### Why

Targets are each user's latest interaction, and the ladder pools match
popularity (and at L3 genre) but not time, so "the newest item wins" beats
random on every Amazon ladder level (Addendum v10 shortcut checks).

### Change

- `reference_rankers.newest_first`: scores each candidate by the first
  timestamp at which it appears in the training frame; items never seen in
  training count as newest (an upper bound; under 0.25% of Film/TV targets).
- `check_benchmark.py` flags it as a shortcut alongside popularity and
  inverse popularity.
- `evaluate_hybrid.py` adds it to the non-LLM blend, so the primary hybrid
  comparison (hybrid vs `nonllm_blend`) must beat the recency signal too.
  Hybrid results in `results/` from earlier runs were computed without it
  and are not re-run.

### Null calibration

With the no-information stub client on shuffled TV pools (8 strategy
members, 400 sessions, 40 shuffle seeds), the sign-flip tests rejected at
p < 0.05 in 4, 3, 1, 0 and 1 of 40 runs for the five debiased-evaluation
comparisons, within the 0-5 expected by chance.

## Addendum v12: Books and Music Domains

Recorded 2026-09-30, before any Books or Music run.

### Why

Two more taste-driven media domains from the same Amazon-Reviews-2023
release, so that cross-domain differences come from the domain and not from
how a dataset was collected: reading (`amazon_books`, category `Books`) and
music (`amazon_music`, category `CDs_and_Vinyl`). Together with film, TV and
games this covers five kinds of media with real purchase order.

### Data (`scripts/prepare_amazon_media.py`, `amazon_media.py`)

- Source: `benchmark/5core/rating_only/{Books,CDs_and_Vinyl}.csv` and
  `raw/meta_categories/meta_{Books,CDs_and_Vinyl}.jsonl`.
- The Books metadata is 14.7 GB and is never stored: it is streamed once and
  only title, creator and category path are cached for rated items. The
  stream checks received bytes against Content-Length and resumes with HTTP
  Range requests, because a dropped connection can otherwise end the stream
  silently (a first local run did, caching only 60% of items).
- Items are shown as `<title> by <creator>` (Books: the `author` name, else
  the first contributor in `store`; Music: the first artist in `store`).
  Album titles alone rarely identify an album. Placeholders ("Various
  Artists", "Rated: ...") give no creator. Reference rankers read the same
  string.
- Genre: the first label on the category path found in a fixed list
  (`BOOK_GENRES`, 31 Amazon top-level genres; `MUSIC_GENRES`, 24 genres plus
  three aliases used under deals sections). Store sections (deals, record
  labels, box sets, calendars) are skipped, so an item filed under one takes
  the genre one level down when present.
- Users: the v9 rule (`sha256("0:<user_id>")[:8] < fraction * 2^32`, then
  5-core until stable) with the fraction frozen per domain so each subset is
  about the size of Movies & TV: Books 0.10, Music 0.30.
- Everything downstream is unchanged from v7-v11.

| | `amazon_books` | `amazon_music` |
|---|---|---|
| Raw ratings / users / items | 9,488,297 / 776,370 / 495,063 | 1,552,764 / 123,876 / 89,370 |
| Subset ratings / users / items | 263,357 / 24,426 / 22,281 | 269,882 / 22,653 / 22,132 |
| Items with a genre / a creator | 99.6% / 99.7% | 98.5% / 96.1% (of titled raw items) |
| Gaps over one day | 81.1% | 54.6% |
| Clean sessions (validation / test) | 20,471 (6,122 / 14,349) | 19,744 (5,976 / 13,768) |
| L3 pools genre-matched | 99.6% | 97.9% |
| Retriever recall@20 | 13.6% | 10.0% |
| Subset ratings SHA-256 (prefix) | `799cb04f` | `f1e1393c` |

### Shortcut checks (test split, 3,000 sessions, MRR; random = 0.180)

| Ranker | Books L1 / L2 / L3 | Music L1 / L2 / L3 |
|---|---|---|
| popularity | 0.372 / 0.178 / 0.183 | 0.330 / 0.180 / 0.183 |
| genre_overlap | 0.318 / 0.308 / 0.180 | 0.329 / 0.321 / 0.178 |
| newest_first | 0.275 / 0.274 / 0.279 | 0.254 / 0.262 / 0.252 |
| title_overlap | 0.444 / 0.426 / 0.393 | 0.415 / 0.401 / 0.373 |
| item_knn | 0.554 / 0.495 / 0.441 | 0.491 / 0.468 / 0.417 |

The ladder behaves as designed: popularity is at random from L2 and genre
overlap at random at L3. The recency shortcut is present at every level,
strongest on Books. Title overlap is high because titles carry the author
or artist and users return to the same creator; the LLM is judged against
these baselines (hybrid test), not against random.

### Runs and comparisons

As in Addenda v8-v10: Experiment 1 and Experiment 2 on both domains in one
Kaggle kernel (`build_kernel.py --domains amazon_books,amazon_music`), one
domain per GPU, with the pre-registered comparisons per experiment, domain
and ladder level.

## Addendum v13: Ladder Level L4 (Recency-Matched)

Recorded 2026-09-30, before any run on L4.

### Why

Every Amazon ladder level so far lets "the newest item wins" beat random
(newest-first MRR 0.25-0.28 at L1-L3, Addendum v11), because targets are
each user's latest interaction and the pools do not match on time. L4
removes that construction artifact while keeping every real taste signal.

### Definition (`benchmark.build_recency_matched_pools`, pool `recency_matched`)

- L3 plus release time: negatives share the target's attribute (genre or
  platform) and are matched on two keys at once, training popularity and
  first training timestamp (never seen counts as newest).
- For each key a count `k` in 0..19 is drawn once, independently, and
  exactly `k` negatives rank below the target on that key, so the target's
  rank on each key is uniform and popularity and newest-first both score at
  random. How the two counts combine (e.g. less popular and newer) adapts
  to the available items; the counts are never redrawn, since redrawing
  until a draw fits skews the target's rank.
- Items with exactly the target's popularity count are not used as
  negatives: counts are small integers, and partial ties pull the target's
  tie-averaged rank toward the middle of the pool (both popularity rankers
  then score below random).
- A session gets an L4 pool only when it is matchable: at least 19 eligible
  items on each side of the target on both keys, and an arrangement of the
  drawn counts exists. Other sessions have no L4 pool (no fallback, which
  would reintroduce the shortcuts). L1-L3 pools are unchanged byte for byte.

Coverage of clean sessions: Books 76.4%, Music 74.7%, TV 66.6%, Film 82.1%,
MovieLens 69.5%, Games 75.4%. Rebuilding the MovieLens and Games benchmarks
locally reproduced every candidate list used in the Kaggle ladder pilots
(4,400 of 4,400 sessions), so L1-L3 are unchanged for them too.
On Kaggle, L4 runs on the matchable part of the ladder sessions, so it has
proportionally fewer sessions than L1 and L3.

### Shortcut checks (test split, MRR; random = 0.180)

| Domain | popularity | inverse pop. | newest-first | genre overlap | title overlap | item-KNN |
|---|---|---|---|---|---|---|
| Books L4 | 0.184 | 0.177 | 0.179 | 0.179 | 0.408 | 0.445 |
| Music L4 | 0.177 | 0.179 | 0.178 | 0.177 | 0.387 | 0.409 |
| TV L4 (n = 1,355) | 0.171 | 0.172 | 0.180 | 0.135 | 0.548 | 0.563 |
| Film L4 | 0.182 | 0.171 | 0.178 | 0.139 | 0.264 | 0.349 |
| MovieLens L4 (n = 1,006) | 0.188 | 0.168 | 0.192 | 0.188 | 0.189 | 0.273 |
| Games L4 | 0.191 | 0.163 | 0.185 | 0.213 | 0.298 | 0.342 |

L4 is the only level with no shortcut flag in any domain. Item-KNN, title
overlap and sequential transitions keep their L3 values: they are real
taste and co-purchase signals and remain the baselines the LLM must beat.
Genre overlap is below random on TV and Film from L3 on (0.135-0.143),
because many targets have several genres and negatives need share only one.

### L5 considered and rejected

A fifth level matched on each session's item-KNN score as well was built
and checked (KNN at 0.178), then dropped before any run. Co-purchase
similarity is genuine personalization, not a construction artifact, so
removing it answers a different question. It could also cover only 33-53%
of sessions, dropping exactly those where the target is clear from
co-purchases, so its population would no longer match the other levels.

## Addendum v14: Context Ensemble Without the Co-Purchase Context

Recorded 2026-09-30. Pre-registered secondary analysis for Film, TV, Books
and Music (before any run); exploratory for Games and MovieLens, whose
context trials were already seen.

### Why

In Experiment 2, `context_collab_v2` shows each candidate's co-purchase
link to the history, the signal item-KNN uses. It was the best member and
the validation-preselected single prompt at every Games level. Two
questions follow: how much of the context ensemble's accuracy is that one
co-purchase context, and does ensembling the three semantic contexts
(`content`, `crowd`, `personal`) help on its own?

### Method (`evaluate_debiased.py --exclude-member context_collab_v2`)

The same trials, with no new model calls. Everything data-driven (slot
priors, preselected single prompt) is refitted on the remaining three
members, and RWRA is recomputed from them (`rwra_recomputed`); recomputing
it from all four members reproduces the recorded RWRA exactly (1,099 Games
sessions, maximum score difference 0). The two primary comparisons are
tested and Holm-corrected as in Addendum v7, as their own family. The
Kaggle job writes `*_debiased_without_context_collab_v2.json` for every
context run.

### Exploratory results (test split, MRR; random = 0.180)

| Games | Single | Naive | Debiased | RWRA | Debiased vs single (Holm p) |
|---|---|---|---|---|---|
| L1, all four | 0.303 | 0.331 | 0.327 | 0.332 | +0.024 (0.44) |
| L1, without collab | 0.217 | 0.237 | 0.243 | 0.238 | +0.026 (0.20) |
| L2, all four | 0.263 | 0.249 | 0.259 | 0.250 | -0.003 (0.81) |
| L2, without collab | 0.192 | 0.193 | 0.204 | 0.197 | +0.012 (0.40) |
| L3, all four | 0.229 | 0.229 | 0.218 | 0.230 | -0.011 (0.83) |
| L3, without collab | 0.183 | 0.194 | 0.186 | 0.193 | +0.002 (0.98) |

On Games most of the context ensemble's accuracy comes from the
co-purchase context: without it the ensemble drops to 0.243 / 0.204 / 0.186
and is at random by L3. Ensembling the three semantic contexts gives small
gains over the best of them that are not significant at any level.
MovieLens is near random with or without it (0.169-0.254).

Agreement weighting also works against the strongest member here: RWRA
weights each member by its agreement with the others, and the co-purchase
context is the one that disagrees, so RWRA cannot favour it.

## Addendum v15: Five Amazon Domains, Experiment 2b, Full L1-L4 Plan

Recorded 2026-09-30, before any Experiment 2b run and before any L4 run.

### Domains

The study runs on five Amazon-Reviews-2023 domains with real purchase
order: `amazon_games`, `amazon_film`, `amazon_tv`, `amazon_books`,
`amazon_music`, each on all four ladder levels (L1 random, L2
popularity-matched, L3 attribute-matched, L4 recency-matched).

Retired from runs (code kept so earlier results in `results/` can be
reproduced; `build_kernel.py` prints a note when one is built):

- `movielens`: ratings were entered in bulk (Addendum v9), so it has no
  next-item signal; its pilots showed no significant ensemble gain at any
  level. It remains a reported contrast from the earlier runs.
- `amazon_movies`: the combined Movies & TV domain, replaced by Film and TV.
- Experiment `context` (the four v8 contexts including `context_collab_v2`).

`run_ensemble.py` no longer defaults to MovieLens: `--domain`, `--examples`
and `--candidates` are required.

### Experiment 2b: semantic contexts (`build_kernel.py --experiment context2b`)

Why: Experiment 2 asks how the LLM responds to semantic item context.
`context_collab_v2` passes co-purchase statistics (the item-KNN signal), not
a description of the item. On Games, Film and TV it was the only context
above random and carried almost all of the context ensemble's accuracy
(Addendum v14), hiding the effect of the semantic contexts. This is a
design reason about what the experiment measures; the Experiment 2 results
with it stay reported as they are.

Members (baseline wording, `prompts.SEMANTIC_CONTEXT_SET`):
`context_content_v2`, `context_crowd_v2`, `context_personal_v2` (unchanged
from v8) and two new ones, each the title plus three fields:

- `context_subgenre_v2`: `subgenre` (deepest category label below the
  item's genre, or product type for games), `family` (labels between the
  genre and the sub-genre), `in_your_history` (how many of the user's last
  50 items share that sub-genre; a history item does not count itself).
- `context_format_v2`: `format` (how the item is sold), `price` (listed
  price), `price_level` (low/mid/high: tertiles of all priced catalog items).

Fields come from the dataset only (`item_metadata.py` and the domain
modules) and are stored with every item in `<domain>_items.jsonl`:
sub-genre from the category path below the genre (store sections such as
"CDs $7 - $10", deals and "General" are skipped); format from the store's
"Format:" label, else the book binding in `details`, the Kindle or Audible
storefront, a DVD/Blu-ray tag in the title, or Prime Video; for games the
product type (game, accessory, console, VR hardware) or "digital code".
Prices are those in the 2023 metadata snapshot, not at rating time. The
Books and Music metadata caches are now `*_meta_compact_v2.jsonl`, which
add the raw inputs of format and price.

Coverage over each domain's items:

| Domain | Sub-genre | Format | Price |
|---|---|---|---|
| Games | 34% (accessory types; games carry no genre labels) | 90% | 67% |
| Film | 6% | 100% (DVD 76%, Blu-ray 23%) | 90% |
| TV | 2% | 100% (DVD 96%) | 94% |
| Books | 99% | 97% (Kindle 37%, Hardcover 27%, Paperback 25%) | 75% |
| Music | 81% | 99% (Audio CD 96%) | 94% |

Expected limits, stated before any run: the sub-genre context is close to
title-only on Film and TV, and format carries little variation on Film, TV
and Music; Books is the domain where both new contexts are informative.

Comparisons: as in Addendum v8, per domain and ladder level on the test
split. Primary (Holm across the two): RWRA vs the validation-preselected
single context, and RWRA vs the naive mean. The debiased-ensemble
comparisons (Addendum v7) are reported as their own Holm family; each
member's test MRR and member agreement are descriptive. Comparing the
ensemble with each single context is not pre-registered and, if reported,
is labelled exploratory.

### Adding the fields changed no pool

Rebuilding every domain with the new item fields left all 35 pool and
example files byte-identical (five domains x L1-L4, retrieval, top-popular,
clean examples), so L1-L4 are the pools already checked in Addenda
v10-v13.

### Line endings and subset fingerprints

`write_subset` now writes Unix line endings on every platform. On Windows,
pandas had written "\r\n", so the laptop fingerprints recorded in Addenda
v10 and v12 (`6c8306d9`, `829a86b6`, `799cb04f`, `f1e1393c`) differed from
Kaggle's although the data were identical (confirmed for Film and TV:
removing the "\r" reproduces Kaggle's `b8f7dd5b` and `5829ac25`). The
fingerprints to check from now on, identical on both machines:

| Domain | Film | TV | Books | Music |
|---|---|---|---|---|
| Subset ratings SHA-256 (prefix) | `b8f7dd5b` | `5829ac25` | `2e41eaa4` | `ef5cee23` |

### Run plan

Two experiments x five domains x L1-L4, shuffled candidates, model
`qwen2.5:3b-instruct`, the session sizes of Addendum v8 (500 L2 sessions;
300 for L1 and L3; L4 on the matchable part of those 300). Six kernels:

    python kaggle/build_kernel.py --username <you> --experiment <strategy|context2b> --domains <pair> --pools random,popularity_matched,attribute_matched,recency_matched --conditions shuffled

with `<pair>` one of `amazon_film,amazon_tv`, `amazon_books,amazon_music`
and `amazon_games` (a single domain is split over both GPUs). Estimated from
the earlier runs (about 12.5 s per session per GPU with 8 members): about
5-6 h per paired strategy kernel, 6-7 h per paired context2b kernel (five
members), about half that for Games alone; roughly 30 GPU hours in total.
