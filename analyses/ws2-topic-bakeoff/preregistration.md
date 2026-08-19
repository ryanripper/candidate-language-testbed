# Preregistration — Workstream 2: LLM-augmented topic modeling (five-way blind bake-off)

*Filled from `ws0/preregistration_TEMPLATE.md` and locked before touching
`ws0/sealed_truth.parquet`. Protocol established 2026-07-20; standing habit
per WS0. Operationalizes §3 of `extensions-execution-plan.md`.*

**Date locked:** 2026-07-26
**Session seed:** 20260726 (per project convention: seed = run date)

## 1. Question

From NOTES.md (07/20/26): *"Is AI good enough to identify themes? Does topic
modeling become obsolete?"* — answered as an experiment: which of five topic
instruments (three classical, one embedding-clustering, one direct-LLM)
best recovers the planted `true_topic` partition of 104,601 blind tweets,
and are the recovered themes interpretable?

## 2. Methods being compared

All entrants read `ws0/blind_corpus.parquet` only, assign **every one of the
104,601 tweets** a topic label, and choose their own number of topics K
blind (that ambiguity is part of what is being tested).

**Shared lexical preprocessing** (entrants 1–3 + all coherence scoring):
strip the `RT @Handle:` prefix from retweets (the org handle is a routing
artifact, not theme language; the retweet *body* is kept — retweets-as-speech,
consistent with WS0/WS1); lowercase; tokens = `[a-z]{3,}` after removing
apostrophes+digits; sklearn ENGLISH_STOP_WORDS ∪ {"rt", "amp", "icymi"};
document frequency floor min_df = 5.

**Shared K-selection rule** (entrants 1–3): sweep K ∈ {5, 8, 10, 12, 15,
20, 25, 30, 40}; pick the K maximizing mean NPMI coherence
(`metrics.npmi_coherence`, top-10 terms) computed on a fixed 10,000-tweet
evaluation subsample drawn once with seed 20260726. Ties break toward
smaller K.

| # | Entrant | Exact configuration |
|---|---|---|
| 1 | **LDA** | gensim `LdaMulticore`, bag-of-words on shared tokens, passes = 2, chunksize = 10,000, random_state = 20260726, per-K; assignment = argmax topic probability per tweet |
| 2 | **NMF** | sklearn `NMF(init="nndsvda", random_state=20260726, max_iter=400)` on TF-IDF (shared tokens, sublinear TF, min_df = 5); assignment = argmax of W row |
| 3 | **LSA** | TF-IDF as above → `TruncatedSVD(100, random_state=20260726)` → row-L2-normalize → `KMeans(n_clusters=K, n_init=5, random_state=20260726)`; assignment = cluster. (The incumbent family — TF-IDF+SVD already won WS1.) |
| 4 | **BERTopic** | Tier B MiniLM tweet embeddings (regenerated via WS1 `01_embed_corpus.py B`, frozen encoder) → UMAP(n_neighbors=15, n_components=5, metric="cosine", min_dist=0.0, random_state=20260726) → HDBSCAN(min_cluster_size=200, metric="euclidean", cluster_selection_method="eom") → c-TF-IDF topics. **Outlier handling (pre-declared):** outlier rate reported explicitly; if > 50% of tweets are outliers, fall back to min_cluster_size = 60; if still > 50%, fall back to k-means mode (K from the entrant-3 sweep). Remaining outliers are assigned to their nearest topic embedding centroid (cosine) so the entrant labels all tweets; the pre-fallback outlier rate stays reported. |
| 5 | **Direct LLM theming** | Two-stage per plan §3: (i) stratified 2,000-tweet sample (proportional by party × is_retweet, seed 20260726); Claude (this session, per D1) reads the sample and proposes a theme taxonomy — **taxonomy step run twice** on two disjoint 1,000-tweet halves, agreement reported, taxonomies merged by the pre-stated union-then-dedupe rule (merge two themes iff they would label the same tweets; Claude documents the merge); (ii) Claude labels all 2,000 sampled tweets with the merged taxonomy; labels propagate corpus-wide by nearest label-centroid (cosine) in Tier B MiniLM space, centroids = mean embedding of the sample tweets carrying each label. |

The LLM entrant, the LLM judge (§6), and Stage B augmentation all run
in-session (decision D1, resolved 2026-07-25: pilot-grade LLM work is
Claude in-session at zero cost).

## 3. Primary metric

**ARI of tweet-level assignments vs sealed `true_topic`** over all 104,601
tweets — `metrics.ari_nmi`, `ari` field. One number per entrant; computed
once, at the single unseal step (§7). For BERTopic the primary score uses
the all-tweets assignment (after nearest-centroid outlier reassignment);
ARI on the non-outlier subset is secondary/diagnostic.

## 4. Comparison set / baselines

No frozen topic-recovery baseline exists (this bake-off creates it). The
relevant frozen references from `ws0/baselines/baseline_validation.csv`
enter at Stage C only: TF-IDF distance validity **0.6238**, corrected w2v
**0.5921**, between/within ratios 1.3255 / 1.3517; plus WS1's exploratory
best distance instrument (Tier A Model2Vec corrected centroid cosine,
distance validity **0.640** — labeled exploratory in WS1 and cited as such).

## 5. Success criterion & decision rule

- **Success bar (pre-registered, plan §3):** at least one entrant reaches
  **ARI ≥ 0.60**. If none does, the finding is that tweet-length text under
  this generator defeats unsupervised topic recovery — also informative,
  and reported as such.
- **Winner rule:** highest ARI wins. If the top two are within 0.03 ARI,
  the higher mean blinded-judge interpretability score (§6) wins. The
  winner (accuracy + interpretability jointly, per plan) is the **sole
  topic instrument carried to real data** and to Stage C / WS3 C2.
- The NOTES.md question is answered by where entrant 5 lands relative to
  entrants 1–4 on ARI *and* interpretability.

## 6. Secondary / diagnostic metrics

- NMI (same unseal step); per-entrant chosen K; wall-clock.
- NPMI coherence (`metrics.npmi_coherence`) and c_v coherence
  (`metrics.cv_coherence`, gensim) on the 10k evaluation subsample;
  topic diversity (`metrics.topic_diversity`, top-25).
- BERTopic pre-fallback outlier rate.
- LLM taxonomy-stability: agreement between the two independent taxonomy
  runs (greedy label matching, Jaccard over sample assignments).
- **Blinded LLM-judge interpretability** (BLIND-SAFE, run before unseal):
  every entrant's topics rendered identically as top-10 c-TF-IDF terms +
  3 medoid example tweets, pooled across entrants, shuffled with seed
  20260726, method identity hidden; Claude scores each topic 1–5 on three
  rubric items (specific / coherent / nameable); topic score = mean of the
  three; entrant score = mean over its topics. Judged in one pass.
  Honest-broker note for the write-up: contestant 5 and the judge are the
  same model; the blind (shuffled, provenance-hidden, term-list-rendered)
  is the mitigation, and this limitation is reported.

## 7. Unseal plan

- `ws0/sealed_truth.parquet` is read **once**, in
  `scripts/07_unseal_validate.py`, after every entrant's corpus-wide
  assignment, all coherence/diversity scores, and the judge scores are on
  disk. WS2 reads `true_topic` (tweet-level) and `true_ideology`
  (per-candidate, for Stage C distance validity). `true_framing` is not
  read.
- Splits: no WS0 split is consumed. The 2,000-tweet LLM sample is WS2-local
  (seed 20260726) because `subsample_150` is a *candidate* split reserved
  for WS3's cost-bound scoring, while entrant 5 needs tweet-level
  stratification. A/B split stays reserved for WS3.
- **Stage B (post-unseal, labeled as such):** LLM augmentation of the
  winner — names + descriptions for each topic; judge-flagged topics
  (interpretability ≤ 2.5) get a merge/split refinement pass; if the winner
  missed the 0.60 bar or its topics are too broad, a **seeded/guided rerun**
  (anchor terms from the entrant-5 taxonomy) is run and the ARI delta
  reported. Everything after the unseal is explicitly *mitigation
  measurement*, not blind discovery, and is labeled so in the article.
- **Stage C (topic-conditioned distances):** built blind from the winner's
  assignments before unseal where possible; validity scored at the same
  single unseal step. Space: Tier A Model2Vec tweet vectors,
  corpus-centered; WS2 replicates the confound gate (§8) independently and
  projects out any style axis to form a corrected space. Per-(candidate,
  topic) centroids for candidates with ≥ 5 tweets in the topic; within-topic
  cosine distance matrices over qualifying candidates. At unseal: per-topic
  distance validity (`metrics.distance_validity` on the qualifying
  submatrix) vs the overall corrected figure; the pre-registered claim
  being tested is that the generator plants framing *conditionally on
  topic*, so the best within-topic validity should exceed the overall
  corrected validity of the same space. Deliverable: "far from whom, on
  what" heatmap — 15 candidate pairs (5 most-distant cross-party, 5
  most-distant same-party, 5 random, chosen by the blind overall corrected
  distances, seed 20260726) × topic.

## 8. Confound gate (mandatory for the Stage C space)

Before any Stage C distance claim: top-10 PCs of the candidate-centroid
matrix (Tier A centered) regressed on retweet share, log10 tweet volume,
and topic-mix entropy (entropy now computed from the *winner's* topic
assignments — an upgrade over WS1's k-means proxy). Style-axis criterion
identical to WS1 §8: max |r| ≥ 0.6 with any covariate → projected out of
the tweet vectors, unless that PC is the blind partisan PC and its |D/R
point-biserial| exceeds its max covariate |r|. WS1 precedent: Tier A PC1
was a retweet-style axis (r = −0.97) — expected to recur and be removed.
