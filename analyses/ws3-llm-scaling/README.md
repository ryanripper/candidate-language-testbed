# WS3 — LLM Ideological Scaling for Behavioral Inference (pilot)

Run 2026-07-26, seed 20260726, per §4 of `../../docs/plans/extensions-execution-plan.md`.
Blind protocol: `preregistration.md` locked first; `ws0-harness/sealed_truth.parquet`
read exactly once (`scripts/06_unseal_validate.py`). Pilot scope per decision
D1: in-session annotator on the frozen `subsample_150`; full 910 × 5 API run
gated on the pre-registered r ≥ 0.90 bar.

## Headline results

- **Stage A/B (ask-and-average, identifiers stripped, n=25 × m=5):**
  pilot **r = 0.970** vs `true_ideology` on the 150 — **clears the
  pre-registered 0.90 bar** (manifesto-literature figure). On identical
  support: behavioral 0.977 (n = 146 — four pilot candidates have no
  split-A retweets; all other rows n = 150), TF-IDF 0.974, **LLM 0.970**,
  WS1 Model2Vec 0.899, w2v 0.878. The LLM matches the lexical ceiling to two decimals
  without any corpus-specific training. Median across-rep SD 0.089;
  calibrated RMSE 0.163; both confound screens clean (max |r| ≈ 0.11).
- **Cue-bias ablation:** leaving the 20 org handles in changed almost
  nothing — Δr = +0.004, mean |paired shift| = 0.055, slight outward
  (polarizing) drift for Democrats (−0.038). When the text itself is this
  discriminative, cues are redundant; stripping cost nothing.
- **Bundle vs tweet-level ablation (30-cand subset):** 0.953 vs 0.951 —
  bundle scoring and tweet-averaging are equivalent here.
- **Stage C (estimate on split A, predict split B):**
  - **C1 retweet-source choice (primary):** held-out log-loss —
    oracle 2.317 < TF-IDF 2.348 < **LLM 2.368** < WS1 2.430 ≪ nulls
    2.98–3.00. Every text instrument recovers ~93% of the oracle's edge
    over null; the LLM leaves only 0.05 nats on the table.
  - **C2 topic attention:** ~null for everyone (JS 0.149–0.151 vs null
    0.158) — **even the oracle barely beats the null**; a 1-D ideology
    score does not predict topic mix under this generator's Dirichlet
    noise. Reported as the designed negative it is.
  - **C3 framing intensity:** TF-IDF 0.955 > oracle 0.945 > LLM 0.933 >
    WS1 0.873 (TF-IDF > oracle = shared lexical method variance with the
    target; caveat in article).
- **D1 gate: CLEARED** → recommendation is the full 910 × 5 API run with
  Ryan's key (`outputs/decision.json`).
  **Status 2026-07-27: DEFERRED by Ryan** — the scale-up run is pinned,
  not funded for now. Synthesis (§5 of the plan) proceeds on the n=150
  pilot scores in `unsealed_pack.parquet`; the gate result stands if the
  run is revisited later.

## Pipeline (scripts/, in order)

| Script | Stage | What |
|---|---|---|
| `01_build_bundles.py` | A | blind bundles: 150×5 main (stripped), 150×2 cue-intact, 680 tweet-level items; 42 agent batch files |
| *(scoring)* | A | 42 fresh in-session agents (per D1), one batch each, blind to identity/truth; raw JSON in `outputs/raw_scores/` |
| `02_collect_scores.py` | A | parse, QC (0 missing / 0 out-of-range), ask-and-average aggregation |
| `03_blind_diagnostics.py` | — | BLIND-SAFE: D/R separation, instrument agreement, confound screen, cue shifts |
| `06_unseal_validate.py` | B | **single unseal** (name fixed by preregistration §7 — hence the 04–05 gap): Stage B table, miss anatomy, decision gate, Stage C pack |
| `07_behavior_prediction.py` | C | C1 softmax choice model, C2 topic shares, C3 framing (reads `unsealed_pack.parquet`, not truth) |
| `08_figures.py` | — | figures 1–5 |

## Key outputs (outputs/)

`scores_main.csv` (the instrument), `scores_bundles.csv`, `scores_cue.csv`,
`scores_tweetlevel.csv`, `scoring_qc.json`, `blind_diagnostics.csv`,
`validation_results.csv`, `miss_anatomy.csv`, `decision.json`,
`c1_retweet_choice.csv`, `c2_topic_attention.csv` (+`c2_topic_r.csv`),
`c3_framing.csv`, `org_ideologies.json`, `unsealed_pack.parquet`,
`bundle_map.csv`, `batch_manifest.json`, `scores_tweetlevel_items.csv`,
`prompts/scoring_prompt_v1.md` (frozen), and `raw_scores.tar.gz` — every
raw agent JSON, the non-regenerable primary data (unlike the `batches/`
prompt files, which are deterministic from `01_build_bundles.py` and not
committed).

Provenance note (2026-08 audit): `scores_tweetlevel_items.csv` is an ad-hoc
in-session merge of `tweetlevel_map.csv` with the per-item scores — no
committed script produces or consumes it. It is kept as a convenience
denormalization; the canonical per-item record is `raw_scores.tar.gz` +
`tweetlevel_map.csv`, and the aggregate `scores_tweetlevel.csv` is what the
pipeline reads.

## Notes for synthesis / real data

- The synthesis stage now has all five instruments on common support:
  `unsealed_pack.parquet` carries LLM, TF-IDF, WS1, behavioral, truth per
  candidate (150).
- Scale-up cost note: the pilot used 1,730 in-session scorings
  (750 main + 300 cue + 680 tweet-level). Full 910 × 5 ≈ 4,550 bundles
  ≈ 2.7M input tokens — single-digit dollars on a small API model, which
  the tweet-vs-bundle equivalence suggests is worth testing first.
  (Deferred 2026-07-27 per Ryan — see D1 status above; synthesis uses the
  150-candidate pilot coverage.)
- Real-data caveats that do NOT transfer automatically: fictional
  candidates made training-data contamination impossible here; on real
  candidates, identifier stripping is load-bearing, not optional.
