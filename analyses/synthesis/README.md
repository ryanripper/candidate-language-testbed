# Synthesis — instrument agreement & the measurement story

Plan §5 of
[extensions-execution-plan.md](../../docs/plans/extensions-execution-plan.md), run
2026-07-27 (session seed 20260727). This is the stage that begins *after*
all three workstreams unsealed: no new blind claims are made here, so
there is no preregistration file — every truth read is downstream of the
workstreams' already-executed preregistered validations.

**D1 status (governs everything here):** the WS3 pilot cleared the
pre-registered bar (r = .970 ≥ .90) and `ws3-llm-scaling/outputs/
decision.json` recommends the full 910 × 5 API run — but Ryan **deferred
(pinned) the scale-up on 2026-07-27**; it is not funded for now. The LLM
instrument therefore exists only on the n = 150 pilot subsample, and all
five-instrument comparisons run on that support. The gate result stands
on the record if the run is revisited; scripts 01–04 pick up a full-corpus
LLM score file with no structural changes.

## Scripts (run in order from the project root)

| Script | What it does | Outputs |
|---|---|---|
| `scripts/01_assemble_instruments.py` | One master candidate × instrument table on both supports (910 / 150); recomputes the full-corpus behavioral score from retweet sources; reproduction checks against every frozen number | `instruments_910.csv`, `instruments_150.csv` |
| `scripts/02_agreement_matrix.py` | Score-level agreement (Pearson + Spearman) on both supports, plus the error-correlation matrix with the oracle partialled out | `agreement_*.csv`, `error_correlation_150.csv`, fig 1 |
| `scripts/03_distance_agreement.py` | Mantel tests (999 perms) and Procrustes similarity of 2-D MDS maps across distance instruments; WS2 × WS3 topic-conditioned tie-in | `mantel_*.csv`, `procrustes_150.csv`, `topic_conditioned_agreement.csv`, fig 2 |
| `scripts/04_divergence_cases.py` | Disagreement index over six instruments; top-8 case table; qualitative tweet packet (the NOTES.md quant-qual bridge) | `divergence_index.csv`, `divergence_cases.csv`, `divergence_case_tweets.md`, fig 3 |
| `scripts/05_consolidated_table.py` | Consolidated validation table spanning WS0–WS3 + synthesis, every value read from the source output files | `consolidated_validation.csv`, fig 4 |

## Headlines

1. Score-level instruments that passed their gates form one tight club:
   oracle, behavioral, TF-IDF and LLM all pairwise r ≥ .97 (n = 150).
2. Errors come in two families — content-driven (LLM/TF-IDF/behavioral,
   residual r ≈ .38–.58 after removing the oracle) and style-driven
   (w2v/MiniLM, residual r ≈ .61). Agreement ≠ independence.
3. Distance is harder than direction for every text-space instrument
   (Mantel r vs oracle geometry .44–.66, vs .72–.97 axis correlations).
   Score-derived |gap| matrices are rank-1 and inflate by construction —
   labeled as such wherever compared.
4. The WS2 three-tier signal ladder (retweet-content ≫ policy ≫
   campaign-process) is reproduced by the truth-free LLM instrument at
   tier-order ρ = .97 — the "far from whom, on what" recipe needs no
   ground truth.
5. All eight top-divergence cases are embedding-family departures, via
   two nameable mechanisms (retweet-style overshoot; process-diet /
   cross-pressure sign flips). Reading the tweets resolves every case —
   the quant-qual loop the NOTES.md entry asked for.

Caveats inherited from the workstreams (template text flatters recovery;
annotator/orchestrator share a model family; frozen baselines saw
split-B text; small-bundle SDs understated) apply to every number cited
here and are restated in `synthesis-writeup.md` §Caveats.

Per D4 (2026-07-25), `technical_writing_sample.pdf` stays frozen — the
consolidated table lives here, not there.
