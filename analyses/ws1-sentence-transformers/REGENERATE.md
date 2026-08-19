# Regenerating the excluded WS1 artifacts

The 16 candidate-distance matrices and the candidate representation files are
excluded from version control (~62 MB). All validation results, diagnostics, PCA
scores, and figures that depend on them **are** committed, so the WS1 conclusions
can be read without rebuilding anything.

Requires a verified WS0 harness first (see `../../ws0-harness/REGENERATE.md`).
Run from inside this folder, in order:

| Step | Command | Produces |
|---|---|---|
| 1 | `python scripts/01_embed_corpus.py` | `intermediate/emb_tier{A,B}.npz` — tweet-level embeddings, never committed at any point (~104k × dim, and cheap to recompute) |
| 2 | `python scripts/02_candidate_representations.py` | `outputs/centroids_tier{A,B}.npz` *(excluded)*, `outputs/pca_tier{A,B}.npz` *(committed)*, `outputs/blind_axis_scores.csv` *(committed)* |
| 3 | `python scripts/03_confound_gate.py` | `outputs/corrected_tier{A,B}.npz` *(excluded)*, plus the committed confound-regression and topic-entropy CSVs |
| 4 | `python scripts/04_distances.py` | the 16 `outputs/D_tier{A,B}_{variant}_{rep}.npy` matrices *(excluded)*, plus committed `blind_diagnostics.csv` and `tierC_gate.json` |

Step 5 (`05_validate.py`) opens the sealed truth. It was run once, per
preregistration §7; its outputs are committed. Re-running it does not invalidate
anything — the blind protocol was already discharged — but be aware you are
running it with the answer key visible, which the original run was not.

Tier A is MiniLM (sentence-transformers); tier B is Model2Vec. Both download
model weights on first use.
