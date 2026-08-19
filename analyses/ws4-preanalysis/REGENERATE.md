# Regenerating the excluded WS4 preanalysis artifacts

One file is excluded from version control: `outputs/candidate_vectors_all.npz`
(~3 MB) — the 910 × 100 candidate matrices for all five feature spaces. The PC
scores derived from it (`outputs/pca_scores_all.npz`) are committed, as are all
result tables and figures.

```bash
python scripts/01_prepare.py     # tokenize the corpus
python scripts/02_embed.py       # trains w2v / fastText / doc2vec / TF-IDF, runs GloVe
```

**GloVe needs a compiled binary.** It is not vendored here. Build it from the
Stanford C source and point the script at the build directory:

```bash
export GLOVE_BIN=/path/to/GloVe/build
python scripts/02_embed.py
```

Without `GLOVE_BIN` the script looks in `third_party/glove/build`. The other four
feature spaces are pure Python and need nothing extra.

**Caveat worth repeating:** this preanalysis is truth-visible and informal — it
was run to inform the WS4 design, not as a certified blind result. Its numbers
are directional and must not be quoted alongside the frozen WS0 baselines without
that qualification.
