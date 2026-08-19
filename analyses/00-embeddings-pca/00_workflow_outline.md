# Workflow Outline — Candidate Language Embeddings, PCA, and Distance Analysis

**Project:** Data Science Skills and Methods — extension of prior congressional-candidate tweet research
**Data:** `synthetic_candidate_tweets_2022.csv.gz` (910 fictional candidates, 104,601 tweets, 2022 midterm cycle, planted ground truth)
**Date:** 2026-07-20
**Pipeline:** five numbered scripts in `scripts/`, run in order; figures land in `figures/`, tables and intermediate artifacts in `outputs/`

---

## Design decisions (agreed before running)

| Decision | Choice | Rationale |
|---|---|---|
| Embedding method | word2vec (skip-gram) trained on the corpus, **plus** TF-IDF + SVD baseline | Mirrors the original research; the sparse baseline shows what "simple" buys you |
| Retweets | Included as candidate speech | Consistent with the prior methodological stance (a retweet is endorsed language) |
| Ground truth | Blind first, then validate | Analysis run without touching `true_*` columns; unsealed only in step 5 |
| Deliverables | Markdown article + numbered .py scripts + PNG figures | Reproducible and Medium-ready |

## Step 0 — Environment

Python 3 with `pandas`, `gensim` (4.4), `scikit-learn`, `scipy`, `matplotlib`. A fixed random seed (20260720) is used everywhere randomness enters (word2vec init, SVD, PCA).

## Step 1 — Corpus preparation (`01_prepare_corpus.py`)

1. Load the flat CSV (104,601 rows).
2. **Seal the ground truth**: write per-candidate `true_ideology` to `ground_truth_SEALED.csv`, then drop all `true_*` columns from the working data.
3. Build a candidate metadata table from observable fields only (party, chamber, state, incumbency, tweet count, retweet share).
4. Tokenize: lowercase; strip URLs and @mentions; unpack hashtags into plain words; keep simple word tokens.
5. Emit corpus stats and a tokenized JSONL corpus.

*Output check:* 910 candidates; 1.44M tokens; 26.5% retweets; median 99 tweets per candidate.

## Step 2 — Embeddings (`02_embeddings.py`)

1. Train word2vec skip-gram (100 dims, window 5, min_count 5, 10 epochs) on all tweets.
2. **Candidate vector = mean of the word vectors of every token the candidate tweeted** (retweets included). This is the "very straightforward" aggregation: no weighting, no document model.
3. Baseline: concatenate each candidate's tokens into one document → TF-IDF (sublinear tf) → TruncatedSVD to the same 100 dims.
4. Sanity check: nearest neighbors of seed words (`border` → wall, patrol, agents…) confirm the space learned topical structure.

## Step 3 — PCA + visualization (`03_pca_visualize.py`)

1. PCA (10 components) on both candidate matrices.
2. **Do not assume PC1 is the interesting axis.** Identify the *partisan axis* empirically: the PC most correlated with the (observable) D/R label, sign-oriented so R is positive.
3. Diagnose the remaining dominant axes against observable covariates (tweet count, retweet share).
4. Figures: scree plot; PC1×PC2 scatter by party for both methods; partisan-axis histograms by party; scatters colored by chamber and incumbency (artifact checks).

*Key findings:* TF-IDF partisan axis = PC1 (|r|=.94 with party). word2vec partisan axis = **PC2** (|r|=.86); word2vec's PC1 (23% of variance) tracks **retweet share** at r=.96 — a style confound, not content.

## Step 4 — Distance analysis (`04_distances.py`)

1. Pairwise cosine distances between word2vec candidate vectors (**raw**).
2. **Style-corrected** variant: project out the retweet-style PC1 from centered vectors, recompute distances.
3. For each variant: 10 most-alike pairs, 10 most-unalike pairs, within- vs between-party mean distance, per-candidate mean distance to the field ("loners" and "centrists").
4. Figures: distance heatmap sorted by partisan axis (corrected); same-party vs cross-party distance histograms, raw vs corrected.

*Key findings:* Raw most-unalike list is contaminated by same-party pairs (style artifact). After correction every most-unalike pair is cross-party, and the between/within separation is visibly sharper.

## Step 5 — Validation against sealed ground truth (`05_validate.py`)

1. Re-join `ground_truth_SEALED.csv`.
2. Correlate each method's partisan axis with `true_ideology` (Pearson + Spearman).
3. Pairwise check: does cosine distance track |true ideology gap|, raw vs corrected?
4. Figure: partisan axis vs true ideology, both methods.

*Results:*

| Measure | r |
|---|---|
| TF-IDF PC1 vs true ideology | **+0.974** (ρ = .939) |
| word2vec PC2 vs true ideology | +0.886 (ρ = .856) |
| word2vec PC1 vs true ideology | +0.171 |
| raw distance vs \|ideology gap\| | +0.281 |
| corrected distance vs \|ideology gap\| | **+0.597** |

## Lessons carried forward (for the real-data phases)

1. **The partisan axis need not be PC1** — identify it against observables instead of assuming.
2. **Averaged embeddings absorb style**: retweet share dominated the word2vec space. Any real pipeline should diagnose dominant PCs against behavioral covariates and correct before measuring distances.
3. **Simple baselines are competitive**: TF-IDF + SVD essentially hit the generator's ceiling (the corpus was built with corr(framing, ideology) ≈ .973).
4. The blind-then-validate protocol worked and should be the template for testing fancier methods (Model2Vec, BERTopic distances, LLM positioning) on this testbed before touching real corpora.

## File map

```
embeddings-pca-analysis/
├── 00_workflow_outline.md          (this file)
├── article_medium_draft.md         (Medium-style writeup)
├── scripts/
│   ├── 01_prepare_corpus.py
│   ├── 02_embeddings.py
│   ├── 03_pca_visualize.py
│   ├── 04_distances.py
│   └── 05_validate.py
├── figures/fig1..fig7 (.png)
└── outputs/  (metadata, pair tables, mean distances, validation_results.csv)
```
