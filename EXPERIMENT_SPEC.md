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
