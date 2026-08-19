# candidate-language-testbed

**How well do text-based instruments recover a politician's ideological position — and how would you know?**

Ryan Ripper — 2026

On real congressional tweets you cannot answer that question cleanly: there is no ground truth, only proxies (DW-NOMINATE, donor scores, expert surveys) that are themselves contested. This project builds a **synthetic corpus with planted ground truth** — 910 fictional 2022-cycle candidates, 104,601 tweets, each candidate generated from a known latent ideology and a known topic mix — and then runs six families of instruments against it under a blind protocol, so that "this method recovers ideology at r = .97" is a measurement rather than a claim.

The design question is not *which method wins*. It is **which methods fail, how, and whether you could have detected the failure without the answer key** — because on real data you never have the answer key.

## Data

The corpus is generated, not collected. Everything — names, handles, organizations — is fictional, and the generator is deterministic under `SEED = 20260719`.

| | |
| --- | --- |
| Candidates | 910 (447 R, 433 D, 30 I) |
| Chambers | 841 House, 69 Senate |
| Tweets | 104,601 (26.5% retweets) |
| Tokens | 1,439,390 across a raw vocabulary of 828 (harness tokenizer: lowercased, URL/mention-stripped, `[a-z][a-z']+`) |
| Median tweets per candidate | 99 |
| Topics | 10 policy topics, plus `campaign_logistics` (uncorrelated with ideology) and `retweet_source` |

`data/synthetic-candidate-tweets/generate_synthetic_candidates.py` plants the structure the instruments are then asked to recover:

- Each candidate gets a `true_ideology` score in [−1, 1] (liberal → conservative), drawn from party-conditional distributions.
- Each candidate gets a Dirichlet topic mix over 11 topics — the 10 policy topics plus `campaign_logistics`, which is uncorrelated with ideology and carries the largest alpha.
- Tweet text is assembled from topic-specific phrase banks, where the probability of drawing liberal- vs. conservative-coded framing is a logistic function of `true_ideology` — so lexical choice encodes the latent score.
- Retweets are drawn from a pool of fictional org/pundit accounts, each with its own ideology; candidates preferentially retweet nearby accounts, modeling retweets-as-endorsed-speech.

The flat corpus ships as `synthetic_candidate_tweets_2022.csv.gz` with 16 columns. `ws0-harness/01_seal_corpus.py` splits it into a **13-column blind working view** and a **sealed truth file** holding `true_ideology`, `true_topic`, and `true_framing`. Both views, and the source file, are SHA-256 stamped in `ws0-harness/seal_manifest.json` (sealed 2026-07-25).

## Method

Every workstream follows the same discipline, enforced by a shared harness:

1. **Seal the truth.** `ws0-harness/01_seal_corpus.py` produces the blind view and the hash-stamped sealed truth file.
2. **Pre-register.** Each workstream writes `preregistration.md` — hypotheses, metrics, decision rules, and the bar to clear — *dated, before unsealing*.
3. **Analyze blind.** Including a mandatory **confound gate**: diagnose the dominant principal components against observable covariates (retweet share, volume, topic entropy) before making any distance claim.
4. **Unseal once.** A single validation script per workstream. Results are reported against the frozen baseline table, not against re-derived numbers.

Anything discovered after the unseal is labelled **EXPLORATORY** and stays labelled that way downstream. Several headline numbers in this repo carry that label.

## Results

Axis recovery — Pearson r against planted ideology, n = 910 unless noted:

| Instrument | r vs truth | Notes |
|---|---|---|
| Behavioral (retweet-source ideology) | **.977** | generator ceiling ≈ .98 |
| TF-IDF + SVD | **.974** | the frozen baseline, and the bar to beat |
| LLM ask-and-average | **.970** | n = 150 pilot; **zero corpus training** |
| Model2Vec (static distilled) | .900 | best of the embedding family |
| word2vec | .878 | |
| MiniLM sentence transformer | .721 | style-contaminated axis |

Distance validity (do pairwise text distances reproduce ideological distances?): Model2Vec corrected **.640** *(exploratory)* ≈ TF-IDF **.624** > word2vec .592 ≫ MiniLM .397. Within the retweet-content topic slice alone: **.849**.

Topic recovery: in the blind bake-off, **nobody cleared the .60 ARI bar** — the LLM entrant won at .289. Once an *observable* retweet-routing convention was applied (decided before modeling, not after), the refined LLM taxonomy at K = 13 reached ARI **.890**.

Held-out behavior prediction (retweet-source choice, estimated on split A, evaluated on split B): oracle 2.317 < TF-IDF 2.348 < LLM 2.368 < Model2Vec 2.430, against nulls at ≈ 2.98–3.00.

### The four findings that transfer

**1. Direction is much easier than distance.** Four instrument families place candidates on a left–right axis at r ≥ .90, yet no high-dimensional text geometry gets above .66 agreement with the true pairwise-distance structure. Recovering *who is to the left of whom* and recovering *how far apart they are* are different problems with a large difficulty gap between them.

**2. The retweet-style confound is universal.** Every embedding space tested — word2vec, GloVe, fastText, doc2vec, MiniLM, Model2Vec, TF-IDF — devotes a dominant principal component to **retweet share**, an observable covariate, at |r| = .90–.96. It is PC1 in the word-vector models and PC2 in doc2vec and TF-IDF. This is detectable blind, which is the point: the confound gate catches it before any substantive claim is made. Correcting for it is what lifts word2vec distance validity from .278 to .592.

**3. Instrument errors cluster into two families.** Partialling out the oracle and correlating the residuals reveals a **content family** (LLM–behavioral .58, LLM–TF-IDF .55) and a **style family** (word2vec–MiniLM .61, MiniLM–Model2Vec .47), with cross-family residuals at .15–.38. Practical consequence: an ensemble gains almost nothing from a second instrument *within* a family. Buy diversity *across* families.

**4. The "far from whom, on what" decomposition can be estimated truth-free.** Replacing the oracle with LLM pilot scores reproduces the WS2 signal-tier ladder almost exactly (tier-order Spearman ρ = **.97**) — LLM taxonomy topics + LLM positional scores + per-topic centroid distances, with no ground truth anywhere in the pipeline. That is precisely the configuration available on real data.

## Requirements

Python 3.10+, plus `requirements.txt`:

```
numpy>=1.26                   pandas>=2.0             scipy>=1.11
scikit-learn>=1.3             matplotlib>=3.8         pyarrow>=14.0
gensim>=4.3                   sentence-transformers>=2.2
model2vec>=0.3                umap-learn>=0.5         hdbscan>=0.8
```

The original runs did not record a lockfile, so these are floors at the versions the pipeline was developed against rather than exact pins. **Results are seed-pinned, not version-pinned** — minor numerical drift across library versions is possible.

## Setup

```bash
git clone https://github.com/ryanripper/candidate-language-testbed.git
cd candidate-language-testbed
pip install -r requirements.txt
```

## Usage

Build the harness first — the analyses read its outputs, and several of its artifacts are gitignored because they are regenerable:

```bash
cd ws0-harness
python 01_seal_corpus.py        # rebuilds the blind/sealed corpus views
python metrics.py               # self-tests
python 02_freeze_baselines.py   # ~10 min, word2vec retrain, single-threaded
python 03_build_splits.py
python 04_verify_harness.py     # must print HARNESS VERIFIED
```

Then run any analysis folder's `scripts/` in numeric order. Every script is seed-pinned; word2vec in particular is sensitive to library version and thread count, and the canonical recipe is seed 20260720, `workers=1`, deterministic `hashfxn`.

The LLM-dependent steps (WS2 topic labelling, WS3 scoring) call an external model API and are the only steps that are not free to re-run. Their outputs are committed so the downstream analyses reproduce without re-scoring; `analyses/ws3-llm-scaling/outputs/raw_scores.tar.gz` is the audit archive.

## Project structure

```
├── data/synthetic-candidate-tweets/   generator + the 104,601-tweet corpus
├── ws0-harness/                       sealed corpus, frozen baselines, shared
│                                      metrics, frozen evaluation splits
├── analyses/
│   ├── 00-embeddings-pca/             pilot: word2vec + PCA + distances (07-20)
│   ├── ws1-sentence-transformers/     MiniLM / Model2Vec — pre-registered
│   │                                  negative result
│   ├── ws2-topic-bakeoff/             LDA / NMF / LSA / BERTopic-style / LLM
│   ├── ws3-llm-scaling/               LLM ask-and-average ideological scaling
│   ├── ws4-preanalysis/               static-embedding bake-off (w2v, GloVe,
│   │                                  fastText, doc2vec) — informal, truth-visible
│   └── synthesis/                     agreement matrix, Mantel/Procrustes,
│                                      divergence cases, consolidated table
└── docs/
    ├── writeups.md                    index of every stage write-up + article draft
    ├── plans/                         research plan and execution plan
    ├── skills/                        2026 methods survey the project draws on
    ├── reference/                     supervised/unsupervised workflow guide
    └── NOTES.md, NOTES-readout.md     working notes and their read-out
```

Each analysis folder carries its own `scripts/` (numbered pipeline), `outputs/`, and `figures/`, plus a stage write-up. The supporting documents vary by stage:

| Folder | `README` | `preregistration` | `REGENERATE` | Write-up |
| --- | :-: | :-: | :-: | --- |
| `00-embeddings-pca` | | | | `00_workflow_outline.md` |
| `ws1-sentence-transformers` | ✓ | ✓ | ✓ | `ws1-writeup.md` |
| `ws2-topic-bakeoff` | ✓ | ✓ | ✓ | `ws2-writeup.md` |
| `ws3-llm-scaling` | ✓ | ✓ | | `ws3-writeup.md` |
| `ws4-preanalysis` | ✓ | | ✓ | `ws4-preanalysis-writeup.md` |
| `synthesis` | ✓ | | | `synthesis-writeup.md` |

The gaps are deliberate rather than oversights: `00-embeddings-pca` is the pre-harness pilot and predates the protocol, while `ws4-preanalysis` and `synthesis` are respectively truth-visible and post-hoc, so neither had a blind result to pre-register.

### Not included in this repository

Roughly 106 MB of derived binary artifacts are excluded to keep the repo cloneable: WS1's 16 distance matrices and candidate representations, WS2's two per-topic distance tensors, the WS0 sealed-corpus parquets and baseline distance matrices, and WS4's feature matrices. `ws0-harness/`, `ws1-sentence-transformers/`, `ws2-topic-bakeoff/`, and `ws4-preanalysis/` each carry a `REGENERATE.md` naming the exact script that rebuilds each file; `.gitignore` also lists every excluded path against the script that produces it. All CSV/JSON results, all figures, all scripts, all manifests, and the source corpus are committed.

Also omitted: a technical writing sample PDF and an annotated companion, both job-application artifacts rather than research outputs.

### A note on paths and historical documents

This research was produced in a sandboxed working directory, not a git repo. Converting it required a folder reorganization and a set of path fixes — scripts had hardcoded sandbox absolute paths and located each other by their old directory names. Every one of those edits is itemized in [docs/repo-restructure-notes.md](docs/repo-restructure-notes.md), along with a folder mapping table. No seed, hyperparameter, metric, or decision rule changed.

The **pre-registrations and write-ups were deliberately not edited.** Several are dated documents written *before* the corresponding unseal, and their evidential value depends on not being rewritten afterward. They therefore refer to folders by their original names — use the mapping table when following a path mentioned in one of them.

## Standing caveats

Read these before quoting any number above.

- **Template-generated text flatters every recovery number.** Planted sentences recur verbatim across candidates. These are *transfer* results — a method that fails here would fail on real tweets; a method that succeeds here has only earned a real-data trial.
- **The frozen baselines saw split-B text**, so leakage mildly favors the baselines.
- **WS3 annotator agents and the orchestrator share a model family** (blinded, and reported as such).
- **Score-derived distance matrices are rank-1 by construction**, so the .93–.95 agreement among score-based geometries is partly bookkeeping.
- **LLM coverage is the n = 150 stratified pilot only.** The full 910 × 5 scale-up cleared its pre-registered bar (r = .970 ≥ .90) and was recommended, but was deferred on 2026-07-27 and is not funded.
- **WS4 preanalysis is truth-visible and informal** — directional only, and must not be quoted alongside the frozen blind numbers without this caveat.

## Status

WS0–WS3 and the synthesis stage are complete. WS4 (supervised prediction of planted ideology) is at the preanalysis stage: its bake-off found that **every feature space saturates the generator ceiling under light ridge supervision** (out-of-fold r = .966–.973 against a ceiling of ≈ .973), which moves the interesting WS4 questions away from raw accuracy toward label efficiency, robustness, and cross-family ensembles.

## Tech stack

NumPy, pandas, SciPy, scikit-learn (PCA, TF-IDF/SVD, ridge, clustering metrics), gensim (word2vec, doc2vec), sentence-transformers (MiniLM), Model2Vec (static distilled embeddings), UMAP + HDBSCAN (BERTopic-style topic entrant), matplotlib (figures), PyArrow (parquet I/O for the sealed corpus), and an external LLM API for the WS2 labelling and WS3 scoring steps.

## Author

Ryan Ripper — 2026
