"""
02_agreement_matrix.py — Synthesis step 2: instrument-agreement matrix
----------------------------------------------------------------------
Candidate-score correlation matrix across every instrument (plan §5,
first bullet), on both supports:

  * n=150 (pilot subsample) — the only support where the LLM instrument
    exists (D1 scale-up deferred by Ryan 2026-07-27): truth, behavioral
    (split-A, as WS3 used), LLM, TF-IDF, w2v, WS1 Tier A, WS1 Tier B.
  * n=910 (full corpus) — the six non-LLM instruments, behavioral =
    all-retweet mean source ideology.

Pearson below the diagonal, Spearman above (one CSV each, plus a
combined figure-ready CSV). Pairwise-complete observations (behavioral
is undefined for zero-retweet candidates: 4/150, 5/910).

Also reports a small "unique agreement" panel: partial correlation of
each instrument pair controlling for truth — how much two instruments
agree BEYOND both tracking the planted ideology. That is the synthesis
question: instruments can each validate ~.9 vs truth while their errors
are either shared (same blind spots) or independent (complementary).

Outputs:
  synthesis/outputs/agreement_150_pearson.csv / _spearman.csv
  synthesis/outputs/agreement_910_pearson.csv / _spearman.csv
  synthesis/outputs/error_correlation_150.csv   (residual r after truth)
  synthesis/figures/fig1_agreement_matrix.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "synthesis" / "outputs"
FIG = ROOT / "synthesis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

I150 = pd.read_csv(OUT / "instruments_150.csv")
I910 = pd.read_csv(OUT / "instruments_910.csv")

COLS_150 = ["truth", "behav_A", "llm_score", "tfidf", "w2v",
            "ws1_tierA", "ws1_tierB"]
COLS_910 = ["truth", "behavioral", "tfidf", "w2v", "ws1_tierA", "ws1_tierB"]
LABELS = {
    "truth": "True ideology (oracle)",
    "behav_A": "Behavioral (split A)",
    "behavioral": "Behavioral (all RTs)",
    "llm_score": "LLM ask-and-average",
    "tfidf": "TF-IDF+SVD PC1",
    "w2v": "word2vec partisan axis",
    "ws1_tierA": "WS1 Model2Vec axis",
    "ws1_tierB": "WS1 MiniLM axis",
}


def corr_matrices(df: pd.DataFrame, cols: list[str]):
    n = len(cols)
    P = np.full((n, n), np.nan)
    S = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            a, b = df[cols[i]].to_numpy(), df[cols[j]].to_numpy()
            ok = np.isfinite(a) & np.isfinite(b)
            P[i, j] = stats.pearsonr(a[ok], b[ok])[0]
            S[i, j] = stats.spearmanr(a[ok], b[ok])[0]
    return (pd.DataFrame(P, index=cols, columns=cols),
            pd.DataFrame(S, index=cols, columns=cols))


P150, S150 = corr_matrices(I150, COLS_150)
P910, S910 = corr_matrices(I910, COLS_910)
P150.to_csv(OUT / "agreement_150_pearson.csv")
S150.to_csv(OUT / "agreement_150_spearman.csv")
P910.to_csv(OUT / "agreement_910_pearson.csv")
S910.to_csv(OUT / "agreement_910_spearman.csv")

# ------------------------- error agreement: partial out the oracle (150)
est = ["behav_A", "llm_score", "tfidf", "w2v", "ws1_tierA", "ws1_tierB"]
resid = {}
t = I150["truth"].to_numpy()
for c in est:
    x = I150[c].to_numpy()
    ok = np.isfinite(x)
    beta = np.polyfit(t[ok], x[ok], 1)
    r_ = np.full_like(x, np.nan)
    r_[ok] = x[ok] - np.polyval(beta, t[ok])
    resid[c] = r_
E = np.full((len(est), len(est)), np.nan)
for i, a in enumerate(est):
    for j, b in enumerate(est):
        ok = np.isfinite(resid[a]) & np.isfinite(resid[b])
        E[i, j] = stats.pearsonr(resid[a][ok], resid[b][ok])[0]
Edf = pd.DataFrame(E, index=est, columns=est)
Edf.to_csv(OUT / "error_correlation_150.csv")

# ------------------------------------------------------------------ fig
def heat(ax, M, cols, title, note):
    n = len(cols)
    im = ax.imshow(M.values, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([LABELS[c] for c in cols], rotation=40, ha="right",
                       fontsize=8)
    ax.set_yticklabels([LABELS[c] for c in cols], fontsize=8)
    for i in range(n):
        for j in range(n):
            v = M.values[i, j]
            ax.text(j, i, f"{v:.2f}".lstrip("0") if v < 1 else "1.0",
                    ha="center", va="center", fontsize=7.5,
                    color="white" if v < 0.72 else "black")
    ax.set_title(title, fontsize=10)
    ax.text(0.5, -0.50, note, transform=ax.transAxes, ha="center",
            fontsize=7.5, style="italic")
    return im


fig, axes = plt.subplots(1, 3, figsize=(21, 5.6),
                         gridspec_kw={"wspace": 0.95})
heat(axes[0], P150, COLS_150,
     "Instrument agreement, Pearson r (n=150 pilot support)",
     "All five instrument families on identical candidates; behavioral n=146.")
heat(axes[1], P910, COLS_910,
     "Instrument agreement, Pearson r (n=910 full corpus)",
     "LLM absent: D1 scale-up deferred (Ryan, 2026-07-27); behavioral n=905.")
im = heat(axes[2], Edf.abs(), est,
          "|Error correlation| after removing oracle (n=150)",
          "Residual agreement once true ideology is partialled out —\n"
          "shared blind spots, not shared signal.")
axes[2].set_xticklabels([LABELS[c] for c in est], rotation=40, ha="right",
                        fontsize=8)
axes[2].set_yticklabels([LABELS[c] for c in est], fontsize=8)
fig.colorbar(im, ax=axes, shrink=0.75, label="correlation")
fig.suptitle("Synthesis fig. 1 — Five instrument families, one target: "
             "agreement matrix", fontsize=12, y=1.00)
fig.savefig(FIG / "fig1_agreement_matrix.png", dpi=200,
            bbox_inches="tight")
print(P150.round(3).to_string())
print()
print("Error correlations (150, oracle partialled out):")
print(Edf.round(3).to_string())
print("fig1 saved")
