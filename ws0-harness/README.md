# WS0 — Shared Harness

Built 2026-07-25 per §1 of `../docs/plans/extensions-execution-plan.md`. Every
workstream (WS1 sentence-transformers, WS2 topic bake-off, WS3 LLM scaling)
reads identical inputs from this folder — nothing is recomputed downstream.

## Contents

| File | Plan step | What |
|---|---|---|
| `01_seal_corpus.py` → `blind_corpus.parquet`, `sealed_truth.parquet`, `seal_manifest.json` | 0.1 | Blind working corpus (13 observable columns, 104,601 rows) and hash-stamped sealed truth (`true_topic`, `true_framing`, `true_ideology`) |
| `metrics.py` | 0.2 | Shared metrics: axis recovery, distance validity, within/between ratio, ARI/NMI, NPMI + c_v coherence, topic diversity, Mantel, Procrustes, confound helpers. `python metrics.py` runs self-tests |
| `03_build_splits.py` → `splits.json`, `tweet_split_ab.parquet` | 0.3 | Frozen evaluation splits (see below) |
| `02_freeze_baselines.py` → `baselines/` | 0.4 | Frozen 07-20 instruments: score arrays, PCA scores, three 910×910 distance matrices, validation references |
| `04_verify_harness.py` → `baselines/baseline_validation.csv` | gate | End-to-end verification; the canonical baseline comparison table |
| `preregistration_TEMPLATE.md` | habit | Fill in per workstream **before** unsealing |

## Rules of the road

1. **Read `blind_corpus.parquet` only.** `sealed_truth.parquet` is opened
   once per workstream, at the validation step declared in that workstream's
   `preregistration.md` (written first, dated).
2. **Compare against `baselines/baseline_validation.csv`**, not against
   re-derived numbers. Key frozen references: TF-IDF axis r = 0.974
   (the bar to clear), corrected w2v distance validity = 0.597,
   between/within ratio = 1.35.
3. **Use the frozen splits.** `subsample_150` (in `splits.json`) for any
   LLM-cost-bound step; `tweet_split_ab.parquet` for behavioral prediction
   (estimate on A, evaluate on B). Never re-split.
4. **Confound gate** before any distance claim in a new space: diagnose
   top PCs against retweet share / volume / topic entropy
   (`metrics.identify_partisan_axis`, `metrics.project_out`).

## Splits (seed 20260725, deterministic)

- **A/B tweet split** — per candidate, 50/50; originals and retweets split
  separately so split B always retains retweets for the retweet-source
  choice model (WS3 C1). On odd counts B gets the extra retweet, A the
  extra original.
- **Stratified subsample (n=150)** — party × chamber × within-party tercile
  of the *blind* TF-IDF partisan score; proportional allocation, min 1 per
  non-empty stratum.

## Baseline provenance note

The plan's step 0.4 said "copy, no recomputation", but the 07-20 session
committed only summary CSVs — score arrays and distance matrices were never
written to the project folder. `02_freeze_baselines.py` regenerated them
with the original recipe and seed (20260720; two reproducibility fixes:
`workers=1`, deterministic hashfxn) and `04_verify_harness.py` confirmed
they reproduce the frozen 07-20 validation numbers (see
`baselines/baseline_validation.csv`, which also adds the TF-IDF distance
references the plan assigns to WS0.4). The arrays in `baselines/` are now
the canonical frozen baselines.

## Reproducing from scratch

```bash
python 01_seal_corpus.py      # needs the source csv.gz staged/available
python metrics.py             # self-tests
python 02_freeze_baselines.py # ~10 min (word2vec retrain, single-threaded)
python 03_build_splits.py
python 04_verify_harness.py   # must print HARNESS VERIFIED
```
