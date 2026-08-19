# WS0 — Building the Shared Harness

*Write-up of Workstream 0 from [extensions-execution-plan.md](../docs/plans/extensions-execution-plan.md), executed and verified 2026-07-25. Session seed 20260725. All artifacts live in `ws0/`; companion reference is `ws0/README.md`.*

## Why a harness at all

The three extension workstreams — sentence-transformer embeddings (WS1), the LLM-augmented topic bake-off (WS2), and LLM ideological scaling for behavioral inference (WS3) — are designed to be *comparable*: each one's results are judged against the same frozen baselines, on the same corpus, using the same metric implementations. That comparability is easy to lose. If each workstream re-tokenizes the corpus, re-derives the TF-IDF baseline, or re-rolls its own evaluation splits, then a difference between two methods can no longer be attributed to the methods. WS0 exists to make that failure mode structurally impossible: every downstream session reads identical inputs from one folder, and nothing in that folder changes again.

There was also one genuinely time-sensitive piece. The evaluation splits (step 0.3) had to be fixed *before* any workstream scores anything — a held-out claim ("scores estimated on split A predict behavior in split B") is only credible if the split predates the scoring. Everything else in WS0 was consolidation; that one piece could not be retrofitted.

## What was built

**0.1 — The seal.** `01_seal_corpus.py` splits the synthetic corpus (104,601 tweets, 910 fictional 2022-cycle candidates) into two hash-stamped parquet files: `blind_corpus.parquet`, carrying the thirteen observable columns every workstream may read, and `sealed_truth.parquet`, carrying the planted ground truth (`true_topic`, `true_framing`, `true_ideology`). `seal_manifest.json` records sha256 hashes of both, so any script can verify it is reading the canonical blind corpus. The protocol, unchanged since the 2026-07-20 analysis: design decisions are made blind; truth is opened once per workstream, at a validation step declared in writing beforehand.

**0.2 — One metrics implementation.** `metrics.py` consolidates the scoring code that previously lived inline in the 07-20 scripts (axis recovery, pairwise distance validity, within/between-party ratio, blind partisan-axis identification, confound projection) and adds what the new workstreams need: ARI/NMI for topic recovery, NPMI coherence (self-contained) and c_v coherence (gensim), topic diversity, a permutation Mantel test, and Procrustes agreement for the synthesis stage. Functions that touch truth are marked UNSEAL-ONLY in the docstrings. The module ships with a self-test suite (`python metrics.py`) covering every function against constructed cases with known answers — perfect recovery, planted separation, shuffled labels, rotated configurations.

**0.3 — Frozen evaluation splits.** Two splits, both deterministic under seed 20260725, both blind:

- *Tweet-level A/B split* (`tweet_split_ab.parquet`), for WS3's prediction direction: within each candidate, originals and retweets are split 50/50 *separately*, so split B — where behavior is measured — always retains retweets for the retweet-source choice model. On odd counts, B gets the extra retweet and A the extra original. Result: 52,295 A / 52,306 B, maximum per-candidate imbalance of one tweet, and every candidate who has any retweets at all has at least one in B.
- *Stratified candidate subsample* (n = 150, in `splits.json`), for LLM-cost-bound steps: strata are party × chamber × within-party tercile of the **blind** TF-IDF partisan-axis score, with proportional allocation and a minimum of one candidate per non-empty stratum. Within-party terciles keep moderates and extremes represented inside each party rather than letting the tercile cut simply reproduce the party split. Composition: 73 R / 71 D / 6 I; 139 House / 11 Senate.

**0.4 — Frozen baselines.** `ws0/baselines/` now holds the canonical instruments every workstream compares against: the per-candidate score arrays for both 07-20 methods (`axis_scores.csv`, `candidate_vectors.npz`, `pca_scores.npz`) and three full 910×910 distance matrices (raw word2vec, style-corrected word2vec, TF-IDF), plus observables metadata and manifests.

**The pre-registration habit.** `preregistration_TEMPLATE.md` turns the 07-20 workflow-outline practice into a standing form: question, methods, primary metric, comparison set, success criterion and decision rule, unseal plan, and the mandatory confound gate. Each workstream copies it, fills it in, and dates it before touching sealed truth.

## One deviation from the plan, and why

The plan's step 0.4 said *"no recomputation; copy the existing score arrays and distance matrices from `embeddings-pca-analysis/outputs/`."* That turned out to be impossible: the 07-20 session committed only summary CSVs to the project folder — the `.npz` score arrays and `.npy` distance matrices were written to the working container and lost when that session ended.

The baselines were therefore **regenerated** from the blind corpus using the 07-20 recipe exactly — identical tokenization, identical hyperparameters, original seed 20260720 — with two deliberate changes made for reproducibility and recorded in `baselines/manifest.json`: word2vec now trains single-threaded (`workers=1`; the original used 4 threads, which is non-deterministic, and the original arrays no longer exist to match anyway) and with an explicit crc32 hash function (so results don't depend on `PYTHONHASHSEED`). The regenerated arrays are now the canonical frozen baselines; any future regeneration from `02_freeze_baselines.py` will reproduce them bit-for-bit given the same library versions.

Regeneration raised an obvious question: how do we know the reconstruction is faithful? That is what the verification gate answers.

## Verification

`04_verify_harness.py` runs four blocks and exits nonzero on any failure. Block 1 checks seal integrity (no `true_*` columns in the blind corpus, hashes match the manifest, 1:1 row alignment). Block 2 runs the metrics self-tests. Block 4 checks the splits (full single coverage, hash match, both halves present for every candidate, no candidate stripped of retweets in B, subsample size and balance).

Block 3 is a **sanctioned unseal**: it opens sealed truth solely to confirm the regenerated baselines reproduce the already-published 07-20 validation numbers, and to compute the TF-IDF distance references the plan explicitly assigns to WS0.4. No design decision was informed by truth. The reproduction:

| Measure | 07-20 frozen | WS0 regenerated |
|---|---|---|
| TF-IDF partisan axis (PC1) vs true_ideology, r | +0.9738 | **+0.9738** |
| word2vec partisan axis (PC2) vs true_ideology, r | +0.8856 | +0.8854 |
| word2vec PC1 (style axis) vs true_ideology, r | +0.1709 | +0.1750 |
| word2vec PC1 vs retweet share, r | ~0.96 | +0.9596 |
| Raw w2v distance validity | +0.2811 | +0.2781 |
| Corrected w2v distance validity | +0.5966 | +0.5921 |
| w2v between/within ratio, raw → corrected | 1.26 → 1.35 | 1.259 → 1.352 |

The TF-IDF axis reproduces exactly (it is deterministic given library versions); the word2vec figures land within retraining noise of the originals, and the qualitative structure — partisan axis on PC2, PC1 a retweet-style confound — reproduces precisely. All 19 checks passed: `HARNESS VERIFIED`.

Two **new reference numbers** came out of the sanctioned unseal, recorded alongside the reproductions in `baselines/baseline_validation.csv`:

- **TF-IDF distance validity = +0.6238**
- **TF-IDF between/within ratio = 1.326**

The first one matters for WS1's framing: TF-IDF's distances already beat corrected word2vec's 0.597. The distance bar the sentence-transformer workstream has to clear is **0.624**, not 0.597 — the "dumb" baseline is stronger on distances than previously bookkept, which is consistent with the 07-20 lesson that the lexical baseline is the one to beat.

## How the workstreams consume this

The rules of the road (also in `ws0/README.md`): read `blind_corpus.parquet` only; compare against `baselines/baseline_validation.csv` rather than re-derived numbers; use the frozen splits and never re-split; run the confound gate (top-PC diagnosis against retweet share, volume, topic entropy) before any distance claim in a new space; and file a dated `preregistration.md` before unsealing. WS1 starts by copying the template and launching the Model2Vec/MiniLM embedding runs (E1.1–E1.3); its success criteria are already fixed — axis r > 0.974, distance validity > 0.624.

One small empirical note for WS3's C1 model, surfaced by the split build: five candidates posted zero retweets across the whole cycle. They will simply contribute no observations to the retweet-source choice likelihood — worth remembering when interpreting per-candidate prediction results.

## Reproducing

```bash
cd ws0
python 01_seal_corpus.py       # needs the source csv.gz
python metrics.py              # self-tests
python 02_freeze_baselines.py  # ~10 min, single-threaded word2vec
python 03_build_splits.py
python 04_verify_harness.py    # must print HARNESS VERIFIED
```
