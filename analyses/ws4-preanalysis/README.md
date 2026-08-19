# ws4-preanalysis

Informal, truth-visible static-embedding bake-off (word2vec / GloVe / fastText /
doc2vec, + TF-IDF+SVD anchor) on the synthetic 2022-cycle corpus — PCA and
distance analyses plus a cross-validated ridge probe, run 2026-08-07 as
groundwork for the WS4 supervised-prediction plan.

**Not a certified WS0 run**: ground truth was visible throughout; partisan axes
are truth-selected (upper bounds). The word2vec column reproduces the frozen
canonical baseline exactly (anchor check), so magnitudes are comparable with
that caveat stated.

Read `ws4-preanalysis-writeup.md` for design, results table, five findings, and
WS4 recommendations.

Pipeline (run in order from the project root; expects the corpus .csv.gz):

    scripts/01_prepare.py        tokenize (07-20 recipe), candidate table
    scripts/02_embed.py          train all models, build 910x100 matrices
    scripts/03_analyze.py        PCA + distance analyses -> summary CSVs
    scripts/04_linear_probe.py   5-fold ridge probe per feature space
    scripts/05_figures.py        figs 1-4

Not committed (regenerable): tokenized corpus, GloVe intermediates
(corpus_glove.txt, cooccurrence/shuffle bins, glove_vectors.txt).
GloVe training used the Stanford GloVe C tool (github.com/stanfordnlp/GloVe),
compiled locally; seeds 20260720 (anchors) / 20260807 (new models).
