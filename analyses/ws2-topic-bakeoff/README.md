# WS2 — LLM-Augmented Topic Modeling (five-way blind bake-off)

Run 2026-07-26, seed 20260726, per §3 of `../../docs/plans/extensions-execution-plan.md`.
Blind protocol: `preregistration.md` locked first; `ws0-harness/sealed_truth.parquet`
read exactly once (`scripts/08_unseal_validate.py`).

## Headline results

- **Stage A (blind):** no entrant reached the pre-registered ARI ≥ 0.60 bar.
  Winner: **direct LLM theming**, ARI 0.289 / NMI 0.630 (LDA 0.156, LSA 0.138,
  NMF 0.044, BERTopic 0.023 with a 763-topic shatter). Truth: **12 topics**.
- **Stage B (post-unseal mitigation, labeled):** the gap was two conventions —
  retweets (26.5% of corpus = one true topic `retweet_source`) and visit-genre
  tweets. Observable retweet routing alone: ARI 0.289 → **0.765**; full refined
  instrument (K=13): **ARI 0.890**. Same rule rescues all entrants; ranking
  preserved.
- **Stage C:** pre-registered within-topic > overall claim **held** (0.704 >
  0.640). Refined decomposition: retweet-content dv **0.849** > policy topics
  0.23–0.57 > campaign-process **0.004**.

## Pipeline (scripts/, in order)

| Script | Stage | What |
|---|---|---|
| `01_prepare_tokens.py` | A | shared preprocessing, TF-IDF, 10k coherence subsample |
| `02_classical_entrants.py` | A | LDA / NMF / LSA with blind NPMI K-sweeps |
| `03_llm_sample.py` | A | stratified 2,000-tweet sample, halves H1/H2 |
| `04_llm_propagate.py` | A | LLM labels → corpus via MiniLM nearest-centroid |
| `05_bertopic_entrant.py` | A | UMAP→HDBSCAN→c-TF-IDF (BERTopic pipeline components), pre-registered outlier ladder |
| `06_judge_prep.py`, `06b_coherence_table.py` | A | blinded judge packet (≤40 topics/entrant sample — documented deviation), uniform coherence table |
| `07_stagec.py` | C | confound gate + per-topic distance matrices (blind machinery) |
| `08_unseal_validate.py` | — | **single unseal**: scoreboard, winner, Stage C validity |
| `09_stageb_augment.py` | B | mitigation ladder L0→L3 + supervised merge ceiling |
| `10_figures.py`, `10b_fig5_refined.py` | — | figures 1–5 |
| `11_reconstruct_exploratory.py` | — | post-unseal reconstruction (2026-08 audit) of `exploratory_rt_routing.csv` + `stagec_validity_refined.csv`, originally ad-hoc session computations |

Naming note: preregistration §7 names the unseal script `07_unseal_validate.py`;
it shipped as `08_unseal_validate.py` because `07_stagec.py` took the 07 slot.
Same script, same single-unseal discipline — the number is the only deviation.

LLM stages ran in-session per decision D1: two independent taxonomy agents
(19/22 theme agreement), eight labeling agents (2,000 tweets), two blinded
judge agents (152 topics).

## Key outputs (outputs/)

`scoreboard.csv`, `decision.json` (winner=llm), `stageb_ladder.csv`,
`stagec_validity.csv` (+`_refined`), `exploratory_rt_routing.csv`,
`assignments_*.npy` (5 entrants + `llm_refined`), `llm_taxonomy.json`,
`judge_scores.csv` + `judge_key.json`, `coherence_diversity.csv`,
`stagec_*.npz` (distance matrices, reusable by WS3/synthesis).

Tweet-level MiniLM/Model2Vec embeddings are NOT committed (regenerable via
`../ws1-sentence-transformers/scripts/01_embed_corpus.py`).

## Notes for downstream workstreams

- **WS3 C2 (topic attention)** should use `assignments_llm_refined.npy`
  (K=13, includes the retweet-content topic) as the topic instrument.
- The Tier A retweet-style PC1 confound replicated a **4th** time
  (`stagec_confound_llm.csv`, r ≈ −0.97 with retweet share) and was projected
  out before all Stage C distances.
- Real-data lesson: fix content-routing conventions (shared/quoted content;
  genre vs subject) explicitly before topic modeling — the choice dominated
  every modeling decision here.
