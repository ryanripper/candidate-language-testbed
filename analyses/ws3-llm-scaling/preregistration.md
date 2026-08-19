# Preregistration — Workstream 3: LLM ideological scaling for behavioral inference

*Filled from `ws0/preregistration_TEMPLATE.md` and locked before touching
`ws0/sealed_truth.parquet`. Protocol established 2026-07-20; standing habit
per WS0. Operationalizes §4 of `extensions-execution-plan.md`. Pilot scope
per decision D1 (resolved 2026-07-25): the in-session annotator runs on the
frozen 150-candidate subsample; the full 910 × 5 API run happens only if the
pilot clears the pre-registered bar.*

**Date locked:** 2026-07-26
**Session seed:** 20260726 (per project convention: seed = run date)

## 1. Question

Can ask-and-average LLM positioning (the Political Analysis manifesto
method: score anonymized text bundles repeatedly, average) recover the
planted `true_ideology` from tweet bundles at the manifesto-literature
level (r ≈ 0.90) — and do the resulting scores carry held-out predictive
power for *behavior* (retweet-source choice, topic attention, framing
intensity), in both directions per Ryan's scope: validate against
behavior first, then predict it?

## 2. Methods being compared

### 2.1 The new instrument: LLM ask-and-average (main condition)

- **Candidates:** the frozen `subsample_150` from `ws0/splits.json` (never
  re-split).
- **Bundles:** per candidate and repetition, n = 25 tweets sampled
  **without replacement from that candidate's split-A tweets only**
  (`ws0/tweet_split_ab.parquet`; split B is reserved for behavior).
  Retweets are included, matching the corpus-wide retweets-as-speech
  assumption. If a candidate has fewer than 25 split-A tweets, the bundle
  uses all of them (blind census: 21 of 150 candidates, minimum 8; these
  are flagged `small_bundle` and their repetitions differ only in
  presentation order — their across-rep SD understates sampling
  variability and is excluded from the stability–error diagnostic).
- **Repetitions:** m = 5 fresh samples per candidate; sampling RNG seed =
  `20260726 * 10 + rep` (rep ∈ 1…5); presentation order shuffled with the
  same seed. Candidate score = mean of the 5 bundle scores; across-rep SD
  = instrument stability.
- **Identifier stripping (main condition — mandatory per the party-cue
  bias literature):** strip the `RT @Handle:` routing prefix (retweet body
  kept); mask any remaining @mentions to `@user`; mask explicit party
  labels (Democrat/Democratic/Dem(s)/Republican/GOP, case-insensitive) to
  `[party]`; strip the candidate's own name/handle if present. Blind
  census: the corpus contains **no** party-label tokens and **no**
  @mentions outside the RT prefix, so in practice stripping = removing the
  20 org handles; the rules are still stated (and applied) so the
  procedure ports to real data unchanged.
- **Scoring prompt:** frozen before any scoring as
  `prompts/scoring_prompt_v1.md`. Each bundle is presented as a numbered
  anonymous bundle of tweets by a single U.S. political figure from the
  2022 midterm cycle; the annotator places the author on a −1.0 (very
  liberal) … +1.0 (very conservative) scale, two decimals, full range
  encouraged, JSON output, no rationale text.
- **Annotator (per D1):** Claude in-session (`claude-fable-5`), zero cost.
  The 750 main-condition bundles are shuffled with seed 20260726 and
  dealt into batches of 25; each batch is scored by a fresh agent that
  sees only bundle IDs and cleaned text — never candidate IDs, never
  another batch, never any truth. Because every candidate is fictional,
  LLM training-data contamination is impossible by construction.
- **Parse/repair rule:** malformed or missing scores are re-requested once
  for the affected bundles (fresh agent, same batch prompt); still-missing
  scores are dropped and the candidate's mean uses the surviving reps
  (count reported).

### 2.2 Ablations (Stage A, secondary — pre-registered here)

- **(a) Bundle vs tweet-level:** on a 30-candidate subset (seeded draw,
  seed 20260726, from the 150; composition reported), every tweet in the
  rep-1 main bundle is scored **individually** on the same scale (batches
  of 125 tweets, same blinding); candidate score = mean of its ~25 tweet
  scores. Comparison: axis recovery r on the same 30 candidates,
  bundle-rep-1 vs tweet-level-mean, plus the correlation between the two.
- **(b) Cue bias:** for all 150 candidates, repetitions 1–2 are re-scored
  from the **same tweet samples** with cues left intact (`RT @OrgHandle:`
  prefixes preserved; no masking). Cue-condition score = mean of the 2
  reps; the paired stripped-condition comparator is the mean of stripped
  reps 1–2 (same tweet sets — the contrast isolates text treatment).
  The deltas (Δr vs truth at unseal; paired per-candidate score shifts,
  overall and by observable party) *are* the cue-bias measurement.

### 2.3 Comparison instruments (all frozen before this workstream)

| Instrument | Source | Note |
|---|---|---|
| TF-IDF+SVD PC1 | `ws0/baselines/axis_scores.csv`, `tfidf_partisan_score` | frozen axis r = 0.9738 (full 910) |
| WS1 best ST axis | `sentence-transformer-analysis/outputs/blind_axis_scores.csv`, Tier A (Model2Vec) centered partisan axis, oriented | full-sample r = 0.9001; WS1's *pre-registered primary* (Tier B) was 0.7212 — both facts reported |
| Behavioral signal | mean retweet-source ideology over **split-A** retweets; org ideologies are planted truth, read at unseal | generator ceiling r = 0.980 (full corpus, all tweets) |
| Oracle | `true_ideology` itself | Stage C ceiling only |

**Known asymmetry (stated now):** the frozen TF-IDF and WS1 axes were fit
on the full corpus, i.e. they have seen split-B text; the LLM instrument
sees split-A only. Any Stage C leakage therefore favors the *baselines*,
making a strong LLM showing conservative — the direction is noted wherever
Stage C is reported.

## 3. Primary metric

**Pearson r of the main-condition ask-and-average score vs `true_ideology`
over the 150 subsample candidates** — `metrics.axis_recovery`,
`pearson_r` field, computed once at the single unseal step (§7). No sign
flip is permitted post hoc; the scale is defined with +1 = conservative,
and the blind D/R orientation check (§6) happens before unseal.

## 4. Comparison set / baselines

From `ws0/baselines/baseline_validation.csv` and §2.3: TF-IDF axis r =
0.9738 (the incumbent bar), w2v 0.8854, WS1 Tier A 0.9001 / Tier B 0.7212
(all full-910 figures; each is **recomputed on the same 150 candidates**
at unseal so all instruments are compared on identical support), and the
behavioral ceiling 0.980. The manifesto-literature figure r ≈ 0.90 is the
external transfer bar.

## 5. Success criterion & decision rule

- **Stage B success bar (pre-registered, plan §4):** main-condition LLM
  r ≥ 0.90 on the 150 → "the ask-and-average method transfers to tweet
  bundles."
- **D1 gate:** r ≥ 0.90 → recommend the full 910 × 5 Anthropic-API run
  (Ryan's key, single-digit-to-tens of dollars); r < 0.90 → the pilot is
  the result, no API spend, and the write-up reports where the method
  loses signal.
- **Stage C headline (pre-registered):** the *ranking of instruments on
  held-out behavioral prediction* — primary Stage C metric = C1 mean
  per-retweet log-loss on split B (§7). If LLM scores predict behavior on
  par with the oracle while embeddings lag (or vice versa), that ordering
  is the paper.
- A negative Stage B result does not abort Stage C: whatever score the
  pilot produced is still entered into the behavioral horse race.

## 6. Secondary / diagnostic metrics

- Spearman ρ (same unseal step); across-rep SD distribution; stability →
  |error| relationship (Pearson r over non-`small_bundle` candidates).
- Miss anatomy: |error| (after per-instrument affine calibration to truth)
  regressed on |true_ideology| (are moderates harder?), log10 split-A
  volume, and split-A retweet share.
- Blind-safe checks run **before** unseal: D/R separation of the LLM score
  (point-biserial), score-distribution/use-of-range plot, and the §8
  confound screen.
- Ablation deltas per §2.2; cue-condition D/R separation shift.
- Agreement of the LLM score with each frozen instrument (blind-safe
  Pearson, computable pre-unseal; reported for the record).

## 7. Unseal plan

- `ws0/sealed_truth.parquet` is read **once**, in
  `scripts/06_unseal_validate.py`, only after all LLM scores (main +
  ablations) are aggregated and written to `outputs/` and the blind
  diagnostics are on disk. WS3 reads `true_ideology` (candidate-level,
  Stage B and Stage C oracle) and `true_framing` (tweet-level, split-B
  rows only, C3 target). `true_topic` is **not** read — WS2 settled the
  topic layer; C2 uses `topic-modeling-bakeoff/outputs/assignments_llm_refined.npy`
  (K = 13) per the WS2 decision rule.
- The 20 planted org ideologies are parsed from
  `synthetic-candidate-tweets/generate_synthetic_candidates.py` at the
  same step (they are generator truth, unseal-scoped).
- Splits: `subsample_150` → all LLM-cost-bound scoring (§2.1–2.2);
  `tweet_split_ab.parquet` → instrument estimation on A, behavior
  measured on B. Never re-split.
- **Stage C (post-unseal, labeled as such — instrument scores themselves
  are frozen blind beforehand):**
  - **C1 retweet-source choice.** For each instrument s ∈ {LLM, TF-IDF,
    WS1-TierA, oracle}: choice model over the 20 orgs,
    P(org j | candidate i) = softmax over j of −β · |a·s_i + b − ι_j|.
    (a, b, β) fit by maximum likelihood on **split-A** retweets of the 150
    (scipy minimize, multi-start; β ≥ 0), frozen, then evaluated on
    **split-B** retweets: mean per-retweet log-loss (primary) and top-1
    accuracy. Nulls: uniform (ln 20) and split-A org base rates.
    Candidates with zero split-B retweets (blind census: 2 of 150) are
    excluded from C1 and counted.
  - **C2 topic attention.** 13-dim split-B topic-share vector per
    candidate from `assignments_llm_refined`. Per topic, linear regression
    share ~ calibrated score fit on **split-A** shares, predicted for
    split B, clipped at 0 and renormalized. Metric: mean Jensen–Shannon
    divergence to observed split-B shares (lower = better) vs the
    constant split-A-grand-mean baseline and the oracle; per-topic
    r(score, split-B share) as diagnostic.
  - **C3 framing intensity.** Target: candidate-level mean split-B framing
    direction, `true_framing` mapped {liberal-coded: −1, neutral: 0,
    conservative-coded: +1} (exact label strings resolved at unseal;
    mapping rule fixed here). Metric: Pearson r(instrument score, target),
    overall and within each of the three refined policy topics with the
    highest WS2 within-topic distance validity (topics 0, 1, 2 in
    `stagec_validity_refined.csv`), tweets assigned by
    `assignments_llm_refined`. Candidates need ≥ 5 split-B tweets in a
    topic to enter its within-topic figure.
- Everything after the unseal is validation/behavioral evaluation, not
  design; no prompt, sampling, cleaning, or aggregation choice may change
  after truth is visible.

## 8. Confound gate (adapted — the instrument is a declared 1-D score, not
a discovered space)

There are no PCs to diagnose: the LLM score axis is declared, not
discovered, so the WS1/WS2 top-10-PC screen does not apply. The
pre-registered analogue:

- **Blind (before unseal):** Pearson r of the LLM score with split-A
  retweet share, log10 split-A tweet volume, and split-A topic-mix entropy
  (from `assignments_llm_refined`). |r| ≥ 0.6 with retweet share or
  volume flags a style contamination (topic entropy is expected to carry
  legitimate ideological signal and is reported, not gated).
- **Post-unseal:** the same covariates regressed on the calibrated error
  (score − truth) — the 07-20 lesson (retweet-style artifacts) checked in
  error space, where "legitimate ideology correlate" is no longer a
  defense. Any |r| ≥ 0.6 finding triggers a reported (not silently
  corrected) sensitivity reanalysis excluding high-retweet-share
  candidates.
