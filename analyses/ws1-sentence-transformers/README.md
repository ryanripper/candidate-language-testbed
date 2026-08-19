# WS1 — Sentence-Transformer Embeddings

Executes §2 of `../../docs/plans/extensions-execution-plan.md` (E1.1–E1.5) on the shared
WS0 harness. Built 2026-07-25, session seed 20260725. All design choices
were locked in `preregistration.md` before `ws0-harness/sealed_truth.parquet` was
opened (once, in script 05).

## Pipeline

| Script | Step | What |
|---|---|---|
| `scripts/01_embed_corpus.py {A\|B\|C}` | E1.1 | One vector per tweet. Tier A = Model2Vec `potion-base-8M` (256-d); Tier B = `all-MiniLM-L6-v2` (384-d); Tier C = `bge-small-en-v1.5`, run only if the blind D2 gate passes |
| `scripts/02_candidate_representations.py` | E1.2–E1.3 | Anisotropy variants (raw / centered / whitened), per-candidate centroids, centroid PCA, blind partisan-axis identification |
| `scripts/03_confound_gate.py` | E1.4 | Mandatory confound gate: top-10 PCs vs retweet share / log volume / blind topic-entropy proxy; style PCs projected out → `corrected` space |
| `scripts/04_distances.py` | E1.3 | Centroid cosine distances (4 variants) + distributional energy & RBF-MMD distances (centered, corrected; full clouds); blind between/within diagnostics; evaluates the D2 Tier-C gate |
| `scripts/05_validate.py` | E1.5 | **The unseal.** Axis recovery, distance validity, decision rule vs frozen baselines |

Run order: `01 A`, `01 B`, `02`, `03`, `04` (then `01 C`, `02`, `03`, `04`
again iff `outputs/tierC_gate.json` says so), `05`.

## Key outputs

- `outputs/validation_results.csv` — the WS1 validation table (extends the
  07-20 Table 1; frozen baselines appended for side-by-side reading).
- `outputs/decision.json` — the preregistered decision rule, applied.
- `outputs/tierC_gate.json` — the blind D2 gate result.
- `outputs/blind_diagnostics.csv`, `outputs/confound_regressions.csv` —
  blind-phase diagnostics.
- `outputs/D_tier{X}_{variant}_{rep}.npy` — 910×910 float32 distance
  matrices, candidate order = `ws0-harness/baselines/candidate_metadata.csv`.
- `figures/fig1`–`fig4`, `article_draft.md`.

## Not committed

Tweet-level embedding arrays (`intermediate/emb_tier*.npz`, ~100–160 MB
each) are regenerable from script 01 (frozen pretrained encoders — no
training, deterministic) and are not stored in the project folder.
