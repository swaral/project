# Experiment 3: every prompt, ensemble, model, domain and level

Test split, title only -> grounded (co-purchase context). Values are MRR (random = 0.180),
except agreement, which is mean pairwise Kendall tau-b between the eight prompts (higher = less drift).
`*` = p < 0.05: agreement and RWRA Holm-corrected in the primary family, the debiased ensemble in the
secondary family, each prompt Holm-corrected across the eight; the naive mean is exploratory and uncorrected.
Item-KNN is the non-LLM reference on the same candidates. Generated from `grounded_vs_title/` and the
trial files; the same numbers are in `experiment3_full_table.csv`.

## 3B, Games

| Method | L1 random (n=191) | L2 popularity (n=323) | L3 attribute (n=190) |
|---|---|---|---|
| **agreement (tau-b)** | 0.010 → 0.115* | 0.007 → 0.088* | 0.007 → 0.082* |
| baseline_scores_v1 | 0.219 → 0.274* | 0.205 → 0.266* | 0.203 → 0.256 |
| next_step_v2 | 0.191 → 0.317* | 0.185 → 0.278* | 0.187 → 0.217 |
| long_term_taste_v2 | 0.216 → 0.306* | 0.200 → 0.252* | 0.177 → 0.183 |
| closest_match_v2 | 0.228 → 0.275 | 0.195 → 0.259* | 0.199 → 0.216 |
| rule_out_rank_v2 | 0.217 → 0.301* | 0.199 → 0.256* | 0.180 → 0.218 |
| preference_enjoy_v2 | 0.209 → 0.321* | 0.215 → 0.269* | 0.196 → 0.264* |
| preference_pick_now_v2 | 0.189 → 0.281* | 0.188 → 0.243* | 0.197 → 0.248 |
| skip_risk_v2 | 0.242 → 0.322* | 0.204 → 0.257* | 0.203 → 0.228 |
| **naive mean** | 0.315 → 0.441* | 0.247 → 0.341* | 0.234 → 0.288 |
| **RWRA** | 0.315 → 0.441* | 0.244 → 0.347* | 0.257 → 0.296 |
| **debiased ensemble** | 0.304 → 0.461* | 0.265 → 0.368* | 0.250 → 0.292 |
| item-KNN (reference) | 0.582 | 0.463 | 0.381 |

## 3B, Film

| Method | L1 random (n=423) | L2 popularity (n=423) | L3 attribute (n=420) |
|---|---|---|---|
| **agreement (tau-b)** | 0.004 → 0.098* | 0.006 → 0.088* | 0.003 → 0.089* |
| baseline_scores_v1 | 0.194 → 0.248* | 0.187 → 0.237* | 0.187 → 0.232* |
| next_step_v2 | 0.187 → 0.253* | 0.190 → 0.230 | 0.196 → 0.242* |
| long_term_taste_v2 | 0.179 → 0.236* | 0.206 → 0.241 | 0.173 → 0.224* |
| closest_match_v2 | 0.192 → 0.252* | 0.180 → 0.232* | 0.195 → 0.252* |
| rule_out_rank_v2 | 0.200 → 0.251* | 0.189 → 0.227* | 0.197 → 0.234* |
| preference_enjoy_v2 | 0.190 → 0.261* | 0.211 → 0.252* | 0.192 → 0.253* |
| preference_pick_now_v2 | 0.182 → 0.251* | 0.201 → 0.236 | 0.193 → 0.226 |
| skip_risk_v2 | 0.206 → 0.209 | 0.195 → 0.195 | 0.203 → 0.195 |
| **naive mean** | 0.215 → 0.315* | 0.227 → 0.300* | 0.226 → 0.296* |
| **RWRA** | 0.220 → 0.321* | 0.231 → 0.301* | 0.228 → 0.299* |
| **debiased ensemble** | 0.219 → 0.317* | 0.232 → 0.303* | 0.216 → 0.302* |
| item-KNN (reference) | 0.406 | 0.394 | 0.365 |

## 3B, TV

| Method | L1 random (n=411) | L2 popularity (n=411) | L3 attribute (n=410) |
|---|---|---|---|
| **agreement (tau-b)** | 0.011 → 0.107* | 0.011 → 0.100* | 0.010 → 0.100* |
| baseline_scores_v1 | 0.209 → 0.312* | 0.213 → 0.286* | 0.206 → 0.290* |
| next_step_v2 | 0.209 → 0.324* | 0.220 → 0.326* | 0.235 → 0.296* |
| long_term_taste_v2 | 0.238 → 0.326* | 0.217 → 0.310* | 0.221 → 0.310* |
| closest_match_v2 | 0.222 → 0.292* | 0.246 → 0.299* | 0.224 → 0.274* |
| rule_out_rank_v2 | 0.204 → 0.292* | 0.205 → 0.327* | 0.214 → 0.277* |
| preference_enjoy_v2 | 0.239 → 0.337* | 0.241 → 0.301* | 0.220 → 0.282* |
| preference_pick_now_v2 | 0.206 → 0.291* | 0.213 → 0.286* | 0.206 → 0.297* |
| skip_risk_v2 | 0.203 → 0.223 | 0.207 → 0.215 | 0.203 → 0.231* |
| **naive mean** | 0.281 → 0.428* | 0.297 → 0.418* | 0.287 → 0.399* |
| **RWRA** | 0.288 → 0.429* | 0.308 → 0.422* | 0.288 → 0.401* |
| **debiased ensemble** | 0.312 → 0.451* | 0.348 → 0.428* | 0.313 → 0.409* |
| item-KNN (reference) | 0.604 | 0.592 | 0.571 |

## 7B, Games

| Method | L1 random (n=191) | L2 popularity (n=326) | L3 attribute (n=191) |
|---|---|---|---|
| **agreement (tau-b)** | 0.086 → 0.201* | 0.095 → 0.217* | 0.109 → 0.231* |
| baseline_scores_v1 | 0.246 → 0.325* | 0.236 → 0.289* | 0.237 → 0.258 |
| next_step_v2 | 0.264 → 0.295 | 0.223 → 0.311* | 0.180 → 0.263* |
| long_term_taste_v2 | 0.270 → 0.303 | 0.249 → 0.313* | 0.197 → 0.240 |
| closest_match_v2 | 0.287 → 0.345 | 0.252 → 0.300* | 0.245 → 0.253 |
| rule_out_rank_v2 | 0.268 → 0.294 | 0.220 → 0.259* | 0.195 → 0.242 |
| preference_enjoy_v2 | 0.280 → 0.329 | 0.267 → 0.340* | 0.222 → 0.272 |
| preference_pick_now_v2 | 0.226 → 0.372* | 0.209 → 0.272* | 0.203 → 0.265* |
| skip_risk_v2 | 0.281 → 0.325 | 0.243 → 0.247 | 0.210 → 0.230 |
| **naive mean** | 0.406 → 0.500* | 0.335 → 0.366 | 0.262 → 0.275 |
| **RWRA** | 0.402 → 0.508* | 0.335 → 0.366 | 0.261 → 0.273 |
| **debiased ensemble** | 0.408 → 0.500* | 0.338 → 0.393* | 0.253 → 0.306* |
| item-KNN (reference) | 0.582 | 0.462 | 0.384 |

## 7B, Film

| Method | L1 random (n=217) | L2 popularity (n=358) | L3 attribute (n=217) |
|---|---|---|---|
| **agreement (tau-b)** | 0.104 → 0.234* | 0.097 → 0.230* | 0.101 → 0.232* |
| baseline_scores_v1 | 0.276 → 0.318 | 0.229 → 0.302* | 0.249 → 0.276 |
| next_step_v2 | 0.254 → 0.294 | 0.242 → 0.287 | 0.239 → 0.253 |
| long_term_taste_v2 | 0.262 → 0.277 | 0.235 → 0.279 | 0.250 → 0.290 |
| closest_match_v2 | 0.267 → 0.312 | 0.217 → 0.266* | 0.257 → 0.271 |
| rule_out_rank_v2 | 0.245 → 0.298 | 0.237 → 0.277 | 0.242 → 0.253 |
| preference_enjoy_v2 | 0.285 → 0.308 | 0.232 → 0.281* | 0.253 → 0.309 |
| preference_pick_now_v2 | 0.260 → 0.326 | 0.234 → 0.282* | 0.258 → 0.261 |
| skip_risk_v2 | 0.265 → 0.279 | 0.235 → 0.256 | 0.249 → 0.267 |
| **naive mean** | 0.348 → 0.384 | 0.314 → 0.351 | 0.345 → 0.348 |
| **RWRA** | 0.351 → 0.384 | 0.313 → 0.355* | 0.347 → 0.346 |
| **debiased ensemble** | 0.361 → 0.395 | 0.306 → 0.354* | 0.360 → 0.337 |
| item-KNN (reference) | 0.444 | 0.407 | 0.390 |

## 7B, TV

| Method | L1 random (n=214) | L2 popularity (n=343) | L3 attribute (n=214) |
|---|---|---|---|
| **agreement (tau-b)** | 0.101 → 0.208* | 0.106 → 0.214* | 0.117 → 0.211* |
| baseline_scores_v1 | 0.300 → 0.332 | 0.304 → 0.343 | 0.300 → 0.318 |
| next_step_v2 | 0.322 → 0.343 | 0.328 → 0.347 | 0.344 → 0.340 |
| long_term_taste_v2 | 0.307 → 0.346 | 0.308 → 0.321 | 0.308 → 0.337 |
| closest_match_v2 | 0.386 → 0.355 | 0.342 → 0.352 | 0.291 → 0.318 |
| rule_out_rank_v2 | 0.263 → 0.321 | 0.262 → 0.325* | 0.268 → 0.310 |
| preference_enjoy_v2 | 0.340 → 0.371 | 0.309 → 0.347 | 0.332 → 0.333 |
| preference_pick_now_v2 | 0.308 → 0.336 | 0.302 → 0.323 | 0.324 → 0.354 |
| skip_risk_v2 | 0.309 → 0.329 | 0.268 → 0.327* | 0.263 → 0.290 |
| **naive mean** | 0.475 → 0.449 | 0.468 → 0.445 | 0.418 → 0.432 |
| **RWRA** | 0.472 → 0.450 | 0.466 → 0.447 | 0.413 → 0.432 |
| **debiased ensemble** | 0.483 → 0.454 | 0.467 → 0.462 | 0.426 → 0.436 |
| item-KNN (reference) | 0.590 | 0.573 | 0.549 |
