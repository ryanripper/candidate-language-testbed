# WS4 preanalysis: a static-embedding bake-off (word2vec, GloVe, fastText, doc2vec)

**Date:** 2026-08-07 · **Status:** informal preanalysis for the WS4 supervised extension — truth-visible, quick standalone run, NOT a certified WS0/blind result. Numbers here are directional and must not be quoted alongside the frozen baselines without this caveat.

## What this is

Before drafting the WS4 plan (supervised prediction of planted ideology), Ryan asked for a very simple preanalysis: take the classic static-embedding family — word2vec and GloVe, plus their peers fastText and doc2vec — train each on the synthetic 2022-cycle corpus, and run the project's standard PCA and distance analyses. TF-IDF+SVD is included as the fifth column because it is the project's reigning simple-methods champion (frozen r = .974).

Setup: same corpus (104,601 tweets, 910 candidates, retweets included as candidate speech), same tokenizer and candidate-vector construction as the 07-20 analysis (frequency-weighted token average, 100 dims). word2vec was run with the exact canonical recipe (seed 20260720, single-threaded, crc32 hashfxn) as a reproduction anchor; GloVe used the Stanford C tool (word+context vectors, 25 iterations); fastText matched word2vec's hyperparameters plus subwords; doc2vec (PV-DBOW) tagged every tweet with its candidate_id, so it learns one vector per candidate directly. Ground truth stayed visible throughout — the "partisan axis" is simply the PC that best correlates with true ideology, an upper bound rather than a blind identification.

**Anchor check (verification):** the word2vec run reproduced the canonical baseline to the third decimal everywhere — axis r = .885 on PC2, distance validity .278 raw → .592 corrected, between/within ratio 1.259 → 1.352 — and TF-IDF landed on .974 exactly. Everything else in the table is therefore on the same footing as the frozen numbers, up to the truth-visible caveat.

## Results

| model | best single PC (r vs truth) | style axis | dist. validity raw → corrected | b/w ratio raw → corrected | CV ridge probe r (100d) |
|---|---|---|---|---|---|
| word2vec | PC2, **.885** | PC1 (r=.96 w/ RT share) | .278 → .592 | 1.259 → 1.352 | .969 |
| GloVe | PC2, **.497** | PC1 (r=−.90) | .182 → .344 | 1.206 → 1.249 | .966 |
| fastText | PC2, **.880** | PC1 (r=−.95) | .273 → .566 | 1.259 → 1.347 | .971 |
| doc2vec | PC1, **.968** | PC2 (r=.94) | .392 → .419 | 1.097 → 1.100 | .973 |
| TF-IDF+SVD | PC1, **.974** | PC2 (r=.95) | .624 → .768 | 1.326 → 1.355 | .971 |

(Style correction = projecting out the PC most aligned with retweet share, the generalization of the 07-20 PC1 correction. Probe = 5-fold candidate-level cross-validated ridge on the full 100-dim vectors, out-of-fold r; generator ceiling ≈ .973.)

## Five findings

**1. The retweet-style confound is universal, not a word2vec quirk.** All five spaces devote a dominant PC to retweet share (|r| .90–.96 with the observable covariate). In the three word-vector models it is PC1; in doc2vec and TF-IDF it is PC2, behind the partisan axis. The 07-20 lesson — check dominant PCs against behavioral covariates before making distance claims — generalizes across the entire embedding family.

**2. word2vec and fastText are near-twins.** Axis recovery (.885 vs .880), distance validity (.592 vs .566 corrected), and ratio (1.35 both) all agree. Subword information adds nothing here — unsurprising on a synthetic corpus with a small (823-word) template vocabulary and no misspellings, but it means fastText earns no slot of its own in WS4.

**3. GloVe looks bad on one axis but fine under supervision — signal smearing.** Its best single PC manages only r = .497, yet the probe recovers .966. The per-PC table shows why: ideology correlations of .50/.42/.36 spread across PCs 2, 4, and 6 rather than concentrating. GloVe's global count-based objective distributes the partisan direction across several components on this small corpus. Lesson for WS4: single-axis PCA numbers understate what a supervised reader can extract from a space.

**4. doc2vec is the surprise of the family — an ideology axis at PC1 r = .968,** statistically tied with TF-IDF, with only 5% explained variance (its variance spectrum is flat, so the axis is nearly pure). But its *distance* structure is the weakest (b/w ratio 1.10, correction barely helps): candidate-tagged PV-DBOW concentrates ideology on one clean direction while the remaining 99 dims carry idiosyncratic noise that dilutes pairwise cosine distances. Strong axis, weak geometry — the two diagnostics genuinely measure different things.

**5. Under supervision, every space saturates at the ceiling.** All five probes land at r = .966–.973 against a generator ceiling of ≈ .973. This is the cleanest possible motivation for WS4's core question: *how much does supervision add over near-ceiling unsupervised recovery?* Answer on the testbed, at n = 910 with light ridge supervision: it erases every unsupervised deficit, including GloVe's 47-point axis gap. It also sharpens the WS4 design problem — with all feature spaces at ceiling, the interesting comparisons move to label efficiency (how few labeled candidates suffice), robustness, and cross-family ensembles, not raw accuracy.

Incidental note: applying the style correction to TF-IDF (not done in the frozen runs) lifts its distance validity from .624 to .768 — worth folding into any future distance-based work as an improved simple baseline, pending a certified rerun.

## What this recommends for the WS4 plan

Carry TF-IDF+SVD (champion baseline), word2vec (canonical anchor), and doc2vec (cheap, near-ceiling, directly candidate-level) as feature spaces; drop fastText (redundant with word2vec) and GloVe (no unique strengths here). Design the headline experiments around label efficiency and candidate-level splits, since raw accuracy saturates. Keep the style-PC projection as a standard preprocessing option and test whether supervision makes it redundant (the probe results suggest it does — the ridge sees past the confound without help).

## Files

- `scripts/01_prepare.py` … `05_figures.py` — numbered pipeline (seeds: 20260720 for the w2v/TF-IDF anchors, 20260807 elsewhere)
- `outputs/validation_summary.csv`, `outputs/pca_correlations.csv`, `outputs/linear_probe.csv` — the tables above
- `outputs/candidate_vectors_all.npz`, `outputs/pca_scores_all.npz` — 910×100 matrices and PC scores per model, reusable as WS4 feature spaces
- `figures/fig1_axis_vs_truth.png` — best axis vs planted ideology, five panels
- `figures/fig2_axis_recovery.png` — single-axis recovery bars
- `figures/fig3_distance_validity.png` — distance validity raw vs style-corrected
- `figures/fig4_probe_vs_axis.png` — unsupervised axis vs supervised probe (the WS4 bridge)
