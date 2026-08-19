# WS3 Session Write-Up — LLM Ideological Scaling for Behavioral Inference

*Session 2026-07-26 (same day as WS2), seed 20260726. This is the working
narrative of what was done and decided; the polished piece is
`article_draft.md`, the operational summary is `README.md`.*

## What was run

Workstream 3 per §4 of `../../docs/plans/extensions-execution-plan.md`, pilot scope per
decision D1 (resolved 2026-07-25: in-session annotator on the frozen
150-candidate subsample; API scale-up only if the pilot clears r ≥ 0.90).

Order of operations, blind protocol intact end to end:

1. **Harness verification.** SHA-256 of `blind_corpus.parquet` and
   `sealed_truth.parquet` matched `ws0/seal_manifest.json` before anything
   ran.
2. **Preregistration locked** (`preregistration.md`, dated 2026-07-26)
   before any scoring and before unseal: design (n=25 × m=5, split-A only,
   identifier stripping), primary metric, 0.90 bar, both ablations, the
   full Stage C protocol including the C1 softmax functional form and
   fitting rule, and the adapted confound gate. One blind census informed
   the prereg (allowed: observable data): all identifier cues in this
   corpus live in `RT @OrgHandle:` prefixes — no party words, no other
   @mentions anywhere.
3. **Bundle build** (`01`): 750 main + 300 cue-intact + 680 tweet-level
   items, dealt into 42 batch files; candidate↔bundle mapping kept out of
   every batch file.
4. **Scoring**: 42 fresh in-session agents, one batch each, blinded to
   identity/mapping/truth; raw JSON preserved in `outputs/raw_scores/`
   (full audit trail). QC: 1,730/1,730 scores parsed, zero missing, zero
   out-of-range. A session-limit interruption mid-run cost nothing —
   completed batches were on disk; the remaining 12 ran after reset.
5. **Blind diagnostics** (`03`) saved to disk *before* unseal: D/R
   point-biserial 0.955; blind agreement with the frozen TF-IDF axis
   0.975; confound screen clean (max |r| 0.11); cue-vs-stripped paired
   shifts computed blind.
6. **Single unseal** (`06`; the 04–05 numbering gap exists because the
   preregistration fixed this script's name): Stage B validation, miss
   anatomy, decision gate, and the truth-derived pack that lets Stage C
   (`07`) run without re-reading `sealed_truth.parquet`. The 20 planted
   org ideologies were parsed from the generator at the same step
   (unseal-scoped by prereg).
7. **Stage C** (`07`) and figures (`08`).

## Results in one breath

Pilot r = **0.970** (bar 0.90: **cleared**; D1 → recommend API scale-up;
*deferred by Ryan 2026-07-27 — see closing section*).
Same-support comparison: behavioral 0.977 > TF-IDF 0.974 > LLM 0.970 >
WS1 0.899 > w2v 0.878. Cue ablation Δr +0.004 (cues redundant; mild
polarizing shift for D candidates). Tweet-level ≈ bundle (0.951/0.953).
Stage C: C1 ranking oracle > TF-IDF > LLM > WS1 ≫ nulls (LLM captures
~93% of the oracle's edge); C2 is a designed negative (even the oracle
barely beats the null — scalar ideology doesn't predict topic mix); C3
strong for all, with TF-IDF > oracle exposing shared lexical method
variance with the text-derived target.

## Decisions taken during the session (all pre-registerable choices were pre-registered; these are execution notes)

- **Batch sizes** (25 bundles / 50 cue bundles / 125 tweets per agent) were
  fixed in the prereg-frozen build script before scoring; rep-to-batch
  assignment randomized (seed 20260726) so a candidate's five reps landed
  in different agents' hands.
- **NaN framing rows** (unframed tweets) are excluded from C3 targets
  rather than counted as neutral — resolved at unseal per the prereg's
  "exact label strings resolved at unseal" clause; labels found were
  con/lib/neu plus NaN.
- **No re-splitting anywhere**: subsample_150 for all scoring,
  tweet A/B for estimate-vs-behavior, per WS0 rules.

## What this closes and what it opens

This completes the third of three workstreams. All three instruments the
plan set out to test now have verdicts on identical support: sentence
transformers (WS1: negative, Model2Vec as cost default), LLM topic
instruments (WS2: taxonomy + propagation with routing rules), LLM scaling
(WS3: transfers at 0.97, matches but does not beat the lexical baseline,
predicts behavior near-oracle). Remaining per the plan: the **synthesis
stage** (§5) — instrument-agreement matrix, Mantel/Procrustes, divergence
case studies — for which `outputs/unsealed_pack.parquet` already carries
all five instruments per candidate; and the **D1 scale-up decision**,
which now sits with Ryan: the gate is cleared, the full 910 × 5 API run is
recommended, cost single-digit-to-tens of dollars with his key.

**Update 2026-07-27:** Ryan has **deferred the D1 scale-up** — pinned,
not funded for now. The gate result and recommendation stand on the
record for whenever it is revisited; the synthesis stage proceeds on the
150-candidate pilot coverage in `unsealed_pack.parquet` and will run in
a separate session.

## Honest-broker notes (repeat wherever WS3 is cited)

Annotator agents and orchestrator share a model family (blinded, but
reported — same caveat as WS2's judge). Template text flatters recovery;
0.97 is a transfer result, not a real-tweet forecast. Frozen baselines saw
split-B text (leakage favors baselines; direction pre-registered). The
cue-stripping rules were largely vacuous on this corpus (stated, applied,
and load-bearing only on real data). Small-bundle candidates' repetition
SDs understate sampling noise (flagged, excluded from the stability
diagnostic).
