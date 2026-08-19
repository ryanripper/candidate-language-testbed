# Extensions Execution Plan — Sentence-Transformer Embeddings, LLM-Augmented Topic Modeling, and LLM Ideological Scaling for Behavioral Inference

*Prepared 2026-07-25. Operationalizes §2 (embedding layer), §4 (LLM-augmented topic modeling), and §5 (LLM positioning) of [candidate-language-research-extension-plan.md](candidate-language-research-extension-plan.md) as three executable workstreams on the synthetic testbed, with the real-data phases sketched as a follow-on. Scope per Ryan (2026-07-25): testbed-first, execution-level detail, and behavioral inference in both directions — scores validated against behavior, then used to predict it.*

---

## 0. Why this plan, and what it builds on

The 2026-07-20 embeddings/PCA analysis established three things this plan inherits directly:

1. **The bar to clear.** TF-IDF+SVD PC1 recovers `true_ideology` at **r = 0.974** — essentially the generator's ceiling (lexical framing ↔ ideology r = 0.973). Corpus-trained word2vec managed r = 0.886, and only on PC2. Any new method must be judged against the "dumb" baseline, not against word2vec.
2. **The confound lesson.** Word2vec's PC1 (23% of variance) was a retweet-style artifact (r = 0.96 with retweet share). Projecting it out doubled distance validity (corr with |ideology gap|: 0.281 → 0.597). Every new embedding space gets the same diagnosis-before-distances treatment: check dominant PCs against behavioral covariates (retweet share, volume, topic-mix entropy) before making any distance claim.
3. **The protocol.** Blind-then-validate: all design choices are made with `true_topic` / `true_framing` / `true_ideology` sealed; ground truth is unsealed once, at a pre-registered validation step. Deliverables follow the established convention — numbered `.py` scripts, a markdown article, `figures/`, `outputs/`.

**Testbed:** `synthetic-candidate-tweets/synthetic_candidate_tweets_2022.csv.gz` — 910 candidates, 104,601 tweets+retweets, 2022 cycle, ~26.5% retweets from 20 fictional org accounts with known ideologies. Behavioral ceiling for reference: mean retweet-source ideology correlates with `true_ideology` at **r = 0.980**.

**Session seed:** 20260725 (per project convention: seed = run date). Each workstream gets its own subfolder in the project root.

A quiet advantage worth exploiting in write-ups: because every candidate is fictional, **LLM training-data contamination is impossible by construction** — the standard objection to LLM-based political scaling (arXiv 2511.13238) simply cannot apply here. That makes the testbed unusually clean for the WS3 questions.

---

## 1. Workstream 0 — Shared harness (prerequisite, ~quarter session)

Mostly **consolidation of the 2026-07-20 embeddings analysis**, not new work — the blind protocol, the baselines, and the core metrics all already exist in `embeddings-pca-analysis/`. WS0 lifts them into one shared location so all three workstreams read identical inputs. The only genuinely new artifact is 0.3, and it is the one that cannot be retrofitted later.

| Step | Status | What | Output |
|---|---|---|---|
| 0.1 | *Consolidate* — seal/unseal mechanics proven on 07-20; re-run the seal against a fresh working copy | Strip `true_*` into `sealed_truth.parquet`, hash-stamped; blind working copy of the corpus | `ws0/blind_corpus.parquet`, `ws0/sealed_truth.parquet` |
| 0.2 | *Consolidate + extend* — r-vs-truth, distance validity, within/between ratio exist inline in the 07-20 scripts; lift into one importable module and add what's new (ARI/NMI, coherence NPMI/c_v, Mantel + Procrustes) | Shared metrics module | `ws0/metrics.py` |
| 0.3 | **New** — must exist before any scoring | Fixed evaluation splits: (a) tweet-level A/B split per candidate (for WS3 prediction), (b) stratified candidate subsample of ~150 (party × chamber × ideology-proxy strata) for anything LLM-cost-bound | `ws0/splits.json` |
| 0.4 | *Consolidate* — no recomputation; copy the existing score arrays and distance matrices from `embeddings-pca-analysis/outputs/` into one canonical location | Frozen TF-IDF+SVD and word2vec baselines so all workstreams compare against identical arrays | `ws0/baselines/` |

**Pre-registration:** before unsealing anything, each workstream writes a short `preregistration.md` naming its primary metric, comparison set, and success criterion. The 07-20 analysis effectively did this via its `00_workflow_outline.md`; WS0 just makes it the standing habit. Cheap, and it makes the eventual write-up (and job-application framing) much stronger.

---

## 2. Workstream 1 — Sentence-transformer embeddings

**Question:** Does context-aware, sentence-level embedding beat the r = 0.974 lexical baseline for position recovery — and, separately, does it produce more valid *distances*?

Those are different questions; the 07-20 analysis showed a method can score well on the axis and poorly on distances.

### Experiments

- **E1.1 — Embed the corpus at three tiers.**
  - Tier A: Model2Vec static (`potion-base-8M`) — seconds on CPU, the "fast sweep" tier.
  - Tier B: full sentence-transformer (`all-MiniLM-L6-v2`; ~30–60 min CPU for 104.6k tweets — fine in-session).
  - Tier C (optional, decision point D2): a stronger model (`bge-small-en-v1.5` or similar) to see if quality scales.
  - One embedding per tweet; retweets embedded as their text (consistent with the retweets-as-speech choice).
- **E1.2 — Anisotropy correction.** Mean-center; run distances raw vs centered vs whitened. Report all three — this is a known failure mode of transformer spaces and a nice methods point.
- **E1.3 — Candidate representation, centroid vs distribution.**
  - Centroid: per-candidate mean vector → PCA → axis diagnosis → distances (mirrors the original design).
  - Distributional: candidates as clouds of tweet vectors; pairwise **energy distance** and **MMD** (RBF kernel, median heuristic). This is the genuinely new capability sentence-level embedding unlocks — test whether distribution-to-distribution distances beat centroid cosine on distance validity.
- **E1.4 — Confound diagnosis (mandatory gate).** Regress top 10 PCs of each space on retweet share, tweet volume, and topic-mix entropy (topic proxy: WS2 clusters or hashtag-free TF-IDF clusters). Project out any style axis before reporting corrected distances, exactly as on 07-20.
- **E1.5 — Unseal & validate.** Primary metric: best-axis r vs `true_ideology`. Secondary: distance validity and within/between ratio vs the frozen baselines (0.597 and 1.35 corrected word2vec; TF-IDF equivalents from WS0.4).

### Success criteria & decision rule

- ST axis r > 0.974 → contextual embeddings earn a place in the real-data pipeline as primary instrument.
- ST axis r ≤ 0.974 but distributional distance validity > centroid/TF-IDF distance validity → adopt as *distance* instrument, keep TF-IDF as *axis* instrument.
- Neither → a publishable negative result ("on short, topically-planted political text, lexical baselines remain sufficient"), and Model2Vec becomes the default for the real-data sweep on cost grounds.

**Deliverables:** `sentence-transformer-analysis/` — scripts 01–05, figures (scree/axis diagnosis, corrected vs raw distance validity, tier comparison), `article_draft.md`, validation table extending the 07-20 Table 1.

---

## 3. Workstream 2 — LLM-augmented topic modeling

**Question:** From NOTES.md (07/20/26): *"Is AI good enough to identify themes? Does topic modeling become obsolete?"* — answered as an experiment rather than an opinion, then used to build the "far from whom, **on what**" decomposition.

### Stage A — Blind topic-recovery bake-off (the Phase-2 opener from the extension plan)

Five entrants, all choosing their own number of topics blind (that ambiguity is part of what's being tested):

| Entrant | Spec |
|---|---|
| LDA | gensim, K chosen by coherence sweep (K ∈ 5…40) |
| NMF | sklearn on TF-IDF, same K sweep |
| LSA | TF-IDF+SVD cluster assignment (the incumbent family — already won once) |
| BERTopic | MiniLM embeddings (reused from E1.1) → UMAP → HDBSCAN → c-TF-IDF; outlier handling reported explicitly |
| Direct LLM theming | LLM reads a stratified ~2,000-tweet sample → proposes a theme taxonomy → labels the sample; labels propagated corpus-wide by embedding nearest-centroid. Two-stage on purpose: it makes LLM theming corpus-scale without 104k calls |

**Scoring (pre-registered):** primary — ARI and NMI of tweet-level assignments vs sealed `true_topic`. Secondary — NPMI coherence, topic diversity, and an interpretability score (blinded LLM-as-judge rubric on topic labels: specific/coherent/nameable, 1–5). This directly quantifies the NOTES.md fear of themes that are "too broad, too inclusive, difficult to interpret."

### Stage B — LLM augmentation of the winner

- LLM labeling + description of each topic; merge/split refinement pass (judge flags incoherent topics → seeded re-run).
- If discovered topics are too broad: **seeded/guided topic modeling** against a fixed issue list as the mitigation, measuring the recovery improvement.

### Stage C — Topic-conditioned candidate comparison

- Per-topic candidate centroids (WS1's best embedding space) → within-topic distances.
- **Validation at unseal:** the generator plants framing *conditionally on topic*, so within-topic distance validity should exceed the overall 0.597-style figures. Report the per-topic decomposition: which topic dimensions carry the partisan signal, which are noise.
- Deliverable figure: the "far from whom, on what" matrix — a candidate-pair × topic heatmap of distances; this is the artifact most worth showing an employer.

**Success criterion:** at least one method reaches ARI ≥ 0.60 vs planted topics (else the finding is that tweet-length text under this generator defeats unsupervised topic recovery — also informative). Decision rule: bake-off winner (accuracy + interpretability jointly) is the sole topic instrument carried to real data.

**Deliverables:** `topic-modeling-bakeoff/` — scripts, bake-off scoreboard table, coherence/ARI scatter, topic-conditioned distance heatmap, `article_draft.md` answering the NOTES.md questions head-on.

---

## 4. Workstream 3 — LLM ideological scaling for behavioral inference

**Question:** Can ask-and-average LLM positioning (Political Analysis method, ~.90 vs expert benchmarks on manifestos) recover ideology from tweets — and do those scores then carry real predictive power for *behavior*? Both directions, per Ryan: validate against behavior first, then predict it.

### Stage A — Ask-and-average positioning

- Per candidate: sample n = 25 tweets (retweets included, matching the corpus-wide assumption), **strip handles, names, party labels, and org identifiers** (the party-cue bias literature makes this mandatory), present as an anonymous bundle; LLM places the author on a −1…+1 liberal–conservative scale. Repeat m = 5 with fresh samples; the score is the mean, with SD as an instrument-stability measure.
- Run on the WS0.3 stratified ~150-candidate subsample first (pilot); full 910 pending decision point D1.
- Ablations: (a) tweet-level scoring then averaging vs bundle scoring; (b) party cues stripped vs left in — the delta *is* the cue-bias measurement, a finding in itself.

### Stage B — Validation direction (scores earn trust)

Unseal and compare, all on the same candidates:

| Instrument | Reference value |
|---|---|
| LLM ask-and-average | this experiment |
| TF-IDF+SVD PC1 | r = 0.974 (frozen) |
| WS1 best sentence-transformer axis | from WS1 |
| Behavioral signal (mean retweet-source ideology) | r = 0.980 (generator ceiling) |

Also: stability (across-repetition SD vs |error|), and where the LLM misses, *who* it misses on (moderates? low-volume candidates? high-retweet-share candidates?).

### Stage C — Prediction direction (scores infer behavior)

Using the A/B tweet split from WS0.3 — scores estimated on split A only, behavior measured on split B:

- **C1 — Retweet-source choice.** Model each candidate's split-B retweet sources as a choice among the 20 orgs: P(org) ∝ softmax(−β·|score − org ideology|). Compare held-out log-loss / top-1 accuracy using LLM scores vs TF-IDF scores vs WS1 scores vs `true_ideology` (oracle ceiling). The gap to oracle measures how much behavioral inference each instrument leaves on the table.
- **C2 — Topic attention.** Predict split-B topic shares (WS2's winning assignments) from the ideology score; the generator tilts each candidate's Dirichlet topic mix ideologically, so a good score should recover attention patterns, not just word choice.
- **C3 — Framing intensity.** Predict split-B lexical framing scores within topics.

**Success criteria:** Stage B — LLM r ≥ 0.90 (the manifesto-literature figure) counts as "method transfers to tweet bundles." Stage C — instrument ranking on held-out behavioral prediction is the headline result; if LLM scores predict behavior on par with the oracle while embeddings lag (or vice versa), that ordering is the paper.

**Deliverables:** `llm-scaling-analysis/` — prompt templates + scoring scripts, instrument-comparison table, behavioral-prediction results, cue-bias ablation figure, `article_draft.md`.

---

## 5. Synthesis — instrument agreement & the measurement story

Once all three workstreams have unsealed:

- Candidate-score correlation matrix across every instrument (TF-IDF, w2v, ST tiers, LLM, behavioral) + Mantel/Procrustes agreement of distance matrices.
- **Divergence case studies:** the 5–10 candidates where instruments disagree most, read qualitatively — which connects directly to the NOTES.md quant-qual bridging entry (numbers flag *where* to look; reading the tweets says *why*).
- A consolidated validation table spanning all three workstreams — the natural extension of the job-application writing sample (`technical_writing_sample.pdf` is built from a script; a second results section slots in cleanly).

---

## 6. Decision points for Ryan

- **D1 — LLM annotator & budget (blocks WS2 Stage-B judge + WS3 full run).** Options: (a) Claude in-session on the ~150-candidate subsample — zero cost, fully runnable now, pilot-grade n; (b) Anthropic API with your key for the full 910 × 5 runs — roughly 4½k calls of ~1k tokens; on a small model this is single-digit dollars, on a large one tens of dollars; (c) both — pilot in-session, scale via API only if the pilot clears the bar. **Recommended: (c).** *Resolved 2026-07-25: (c). Pilot ran 2026-07-26 and cleared the bar (r = 0.970 ≥ 0.90) — but as of **2026-07-27 the full 910 × 5 API scale-up is DEFERRED by Ryan** (pinned, not funded for now). Synthesis proceeds on the n=150 pilot scores; the gate result stands if the run is revisited.*
- **D2 — Embedding Tier C.** Add a stronger sentence-transformer to E1.1, or hold at Model2Vec + MiniLM? (Cost is ~1–2 h extra CPU; recommend adding it only if MiniLM materially beats TF-IDF.)
- **D3 — Write-up shape.** Three articles (one per workstream, matching the existing convention) vs one combined "three instruments" paper. Recommend deciding after WS1 results.
- **D4 — Writing-sample update.** Extend `technical_writing_sample.pdf` with these results as they land, or keep it frozen for current applications?

## 7. Sequencing

Dependencies: WS2 reuses WS1's E1.1 embeddings; WS3 Stages B/C compare against WS1/WS2 outputs but Stage A can run any time after WS0.

| Session | Work |
|---|---|
| 1 | WS0 consolidation + WS1 E1.1–E1.3 (embeddings running while the harness is assembled) |
| 2 | WS1 E1.4–E1.5 + article; WS2 Stage A bake-off launched |
| 3 | WS2 Stages B–C + article; WS3 Stage A pilot (in-session annotator) |
| 4 | WS3 Stages B–C (+ full API run if D1 approves) + article |
| 5 | Synthesis, agreement analysis, divergence case studies, writing-sample update per D4 |

## 8. Risks & mitigations

- **BERTopic/HDBSCAN outlier flood** (common on tweets): report outlier rate as a metric; fall back to reduced `min_cluster_size` or k-means-mode BERTopic; the bake-off design means one entrant failing is a result, not a blocker.
- **LLM theming taxonomy drift** (sample-dependent taxonomies): fix the sample by seed; run the taxonomy step twice and report agreement.
- **Blind-protocol leakage:** prior sessions' summary statistics (e.g., r = 0.974) are known, but per-candidate/per-tweet truth stays sealed; the seal script + pre-registration keep design decisions honest, as on 07-20.
- **Container compute:** MiniLM over 104.6k tweets is the heaviest step (≲1 h CPU); everything else is minutes-scale. No GPU needed.

## 9. Real-data follow-on (sketch only, per scope decision)

Each workstream's decision rule names one surviving instrument. Those three survivors — best embedding tier, best topic method, LLM scaling if validated — then run on the frozen historical corpora from the extension plan's Phase 0 (congresstweets archive, CampaignView, congress-press; bulk download, no scraping), validated against DW-NOMINATE for incumbents, with the retweet-weight sensitivity analysis (λ-sweep) and eventually the Bluesky Jetstream live pipeline. Nothing in this plan needs rework to get there: the metrics module, pre-registration habit, and instrument-agreement framework port unchanged — only the gold standard changes, from planted truth to roll-call behavior.
