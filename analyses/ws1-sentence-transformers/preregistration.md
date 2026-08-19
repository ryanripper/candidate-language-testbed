# Preregistration — Workstream 1: Sentence-transformer embeddings

*Filled from `ws0/preregistration_TEMPLATE.md` and locked before touching
`ws0/sealed_truth.parquet`. Protocol established 2026-07-20; standing habit
per WS0.*

**Date locked:** 2026-07-25
**Session seed:** 20260725

## 1. Question

Does context-aware, sentence-level embedding beat the frozen lexical
baseline (TF-IDF+SVD PC1, r = 0.9738) for ideological *position recovery* —
and, separately, does it produce more valid *distances* (frozen bars:
TF-IDF 0.6238, corrected word2vec 0.5921)? These are scored as two distinct
questions per the 07-20 lesson that a method can win one and lose the other.

## 2. Methods being compared

All embeddings are one vector per tweet over the full 104,601-row
`ws0/blind_corpus.parquet` (retweets embedded as their text, consistent
with the retweets-as-speech choice). Candidates ordered as in
`ws0/baselines/candidate_metadata.csv`.

**Tiers (E1.1):**

- **Tier A** — Model2Vec static: `minishlab/potion-base-8M` (256-d).
- **Tier B** — Sentence-transformer: `sentence-transformers/all-MiniLM-L6-v2`
  (384-d, mean pooling, model-default normalization).
- **Tier C (conditional; D2 blind gate)** — `BAAI/bge-small-en-v1.5`
  (384-d). Runs **iff** the blind gate in §2a passes. If the gate fails,
  Tier C is not run and not scored in this workstream.

**Anisotropy variants (E1.2)**, defined at the tweet-vector level:

- `raw` — vectors as produced by the model.
- `centered` — corpus mean subtracted.
- `whitened` — PCA whitening of centered vectors (components with
  eigenvalue > 1e-10; ε = 1e-8).
- `corrected` — centered minus any style directions found by the confound
  gate (§8). If no style direction is found, `corrected` ≡ `centered`.

**Candidate representations (E1.3):**

- **Centroid** — per-candidate mean tweet vector; pairwise **cosine
  distance**; computed for all four variants.
- **Distributional** — candidates as clouds of tweet vectors; pairwise
  **energy distance** (Euclidean; unbiased within-cloud terms) and
  **MMD** (RBF kernel, unbiased within-cloud terms; bandwidth σ = median
  pairwise Euclidean distance among a 2,000-tweet random sample, seed
  20260725, per tier × variant); computed for `centered` and `corrected`
  only. No per-candidate subsampling — full clouds.

**Axes:** per tier, top-10 PCs of the candidate-centroid matrix are
computed for the `centered` and `corrected` spaces;
`metrics.identify_partisan_axis` (blind-safe, observable D/R labels)
selects the partisan PC, `metrics.orient_axis` fixes the sign (R > D).
**The pre-declared primary instrument is the Tier B (MiniLM)
corrected-space partisan axis.** All other tier × space axes are secondary.

Anything not listed here does not get scored post hoc.

### 2a. Blind Tier-C gate (decision point D2, resolved 2026-07-25)

Ryan's D2 decision: conditional inclusion via a blind gate. Tier C runs iff
**either** of the following blind-safe diagnostics holds for Tier B:

1. max over Tier B's pre-registered distance matrices of the blind
   between/within party ratio (`metrics.within_between_ratio`) ≥ **1.3255**
   (the frozen TF-IDF reference in `baselines/baseline_validation.csv`); or
2. |point-biserial corr| of Tier B's corrected-space partisan axis with the
   observable D/R label > the same quantity computed from the frozen
   `baselines/axis_scores.csv` `tfidf_partisan_score`.

The gate is evaluated in script 04, strictly before unsealing. If it
passes, Tier C runs through the identical E1.1–E1.4 pipeline before
script 05 executes.

## 3. Primary metric

**Pearson r of the Tier B corrected-space partisan axis vs
`true_ideology`** — `metrics.axis_recovery`, `pearson_r` field. One number
decides the axis question.

## 4. Comparison set / baselines

From `ws0/baselines/baseline_validation.csv` (WS0 canonical values):

- TF-IDF partisan axis r = **0.9738** (the bar to clear; primary comparison).
- word2vec partisan axis r = 0.8854 (secondary reference).
- Distance validity: TF-IDF **0.6238**; corrected w2v **0.5921**;
  raw w2v 0.2781.
- Between/within ratio: TF-IDF **1.3255**; corrected w2v **1.3517**;
  raw w2v 1.2588.

## 5. Success criterion & decision rule

Copied from `extensions-execution-plan.md` §2 and binding:

- **ST axis r > 0.9738** → contextual embeddings earn a place in the
  real-data pipeline as primary instrument.
- **ST axis r ≤ 0.9738 but** best pre-registered distributional distance
  validity > max(centroid distance validity, TF-IDF 0.6238) → adopt
  sentence transformers as *distance* instrument, keep TF-IDF as *axis*
  instrument.
- **Neither** → publishable negative result ("on short, topically-planted
  political text, lexical baselines remain sufficient"); Model2Vec becomes
  the default embedding for the real-data sweep on cost grounds.

## 6. Secondary / diagnostic metrics

Reported, not decision-driving: axis recovery for all secondary axes
(Tier A, Tier C if run, centered-space variants, Spearman ρ);
`distance_validity` and `within_between_ratio` for every pre-registered
tier × variant × representation distance matrix; whitening effect
(raw vs centered vs whitened centroid distances); confound-gate regression
table (top-10 PCs × 3 covariates); embedding wall-clock times.

## 7. Unseal plan

- `ws0/sealed_truth.parquet` is read **once**, in
  `scripts/05_validate.py`, after scripts 01–04 have produced every score
  and distance matrix. Only the per-candidate `true_ideology` (constant
  within candidate) is used; `true_topic` / `true_framing` are not read
  by WS1.
- Splits: none of `ws0/splits.json` is consumed by WS1 (no LLM-cost-bound
  step, no behavioral prediction); the A/B split and subsample_150 stay
  reserved for WS2/WS3.

## 8. Confound gate (mandatory; E1.4)

Before any distance or axis claim, for each tier: regress the top-10 PCs
of the centered centroid space on three observable covariates —
**retweet share** (`share_retweets` in candidate metadata), **log10 tweet
volume** (`n_tweets`), and **topic-mix entropy** (blind proxy: per-tweet
k-means cluster, k = 15, seed 20260725, on a blind TF-IDF+SVD tweet
representation built with the WS0 recipe — min_df = 5, sublinear TF,
100 SVD components; entropy of each candidate's cluster distribution).

**Style-axis criterion (pre-declared):** a PC with max |Pearson r| ≥ 0.6
against any covariate is a style axis and is projected out of the tweet
vectors (`metrics.project_out`) to form the `corrected` space —
**unless** that PC is the blind-identified partisan PC *and* its |D/R
point-biserial| exceeds its max covariate |r| (partisan signal is not a
confound). 07-20 precedent: w2v PC1 was a retweet-style artifact
(r = 0.96 with retweet share); projecting it out raised distance validity
0.281 → 0.597.
