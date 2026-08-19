# Preregistration — [Workstream N: title]

*Copy this file into the workstream folder as `preregistration.md` and fill it
in COMPLETELY before touching `ws0/sealed_truth.parquet`. Date it. The point
is that every evaluative choice is on record before truth can influence it
(protocol established 2026-07-20; made a standing habit at WS0).*

**Date locked:** YYYY-MM-DD
**Session seed:** [run date, e.g. 20260725]

## 1. Question

One sentence: what is this workstream trying to find out?

## 2. Methods being compared

List every method/instrument that will be scored, with its exact
configuration (model names, hyperparameters, K-selection rule, prompt
version, …). Anything not listed here doesn't get scored post hoc.

## 3. Primary metric

The single pre-declared metric that decides the outcome, and the exact
implementation used (cite the `ws0/metrics.py` function).

## 4. Comparison set / baselines

Which frozen references from `ws0/baselines/baseline_validation.csv` this
result is judged against (e.g. TF-IDF axis r = 0.974; corrected w2v distance
validity = 0.597).

## 5. Success criterion & decision rule

- Success bar: [e.g. ARI ≥ 0.60; axis r > 0.974]
- Decision rule: [what happens to the pipeline under each outcome —
  including what a negative result means]

## 6. Secondary / diagnostic metrics

Metrics reported but not decision-driving (coherence, stability SDs,
within/between ratios, …).

## 7. Unseal plan

- What is read from `sealed_truth.parquet`, at which script/step, exactly once.
- Which splits from `ws0/splits.json` are used, and for what
  (subsample_150 → LLM-cost-bound steps; tweet A/B → estimate on A,
  evaluate behavior on B).

## 8. Confound gate (mandatory for any new embedding/score space)

Before any distance or axis claim: regress top-10 PCs on observable
covariates (retweet share, tweet volume, topic-mix entropy); name the
correction applied if a style axis is found (07-20 lesson: w2v PC1 was a
retweet-style artifact, r = 0.96 with retweet share).
