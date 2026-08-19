"""
05_figures.py — WS4 preanalysis, step 5. Four comparison figures.

Model colors: fixed categorical assignment (validated palette, ΔE checks
passed; bars carry direct value labels). Party colors follow the project's
established convention (D blue / R red / I olive) for continuity with the
07-20 and WS1-WS3 figures.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path as _Path
import os as _os
_HERE = _Path(__file__).resolve().parent.parent   # this analysis folder
_ROOT = _HERE.parents[1]                          # repository root
OUT = str(_HERE / "outputs")
FIG = str(_HERE / "figures")

MODELS = ["w2v", "glove", "fasttext", "doc2vec", "tfidf"]
LABELS = {"w2v": "word2vec", "glove": "GloVe", "fasttext": "fastText",
          "doc2vec": "doc2vec", "tfidf": "TF-IDF+SVD"}
MCOLOR = {"w2v": "#2a78d6", "glove": "#eb6834", "fasttext": "#1baf7a",
          "doc2vec": "#eda100", "tfidf": "#e87ba4"}
PARTY_COLORS = {"D": "#3b6fb6", "R": "#c23b3b", "I": "#8a7d4a"}
CEILING = 0.973  # generator's lexical recoverability check

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})


def main() -> None:
    meta = pd.read_csv(f"{OUT}/candidate_table.csv")
    summ = pd.read_csv(f"{OUT}/validation_summary.csv").set_index("model")
    probe = pd.read_csv(f"{OUT}/linear_probe.csv").set_index("model")
    pz = np.load(f"{OUT}/pca_scores_all.npz", allow_pickle=True)
    y = meta["true_ideology"].to_numpy()

    # ---- Fig 1: partisan axis vs truth, per model ----
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), sharex=True)
    for ax, m in zip(axes, MODELS):
        P, k = pz[f"P_{m}"], int(pz[f"axis_{m}"])
        s = P[:, k]
        if np.corrcoef(s, y)[0, 1] < 0:  # orient axis for display only
            s = -s
        for party, c in PARTY_COLORS.items():
            msk = (meta["party"] == party).to_numpy()
            ax.scatter(y[msk], s[msk], s=7, alpha=0.6, c=c, linewidths=0,
                       label=party)
        r = summ.loc[m, "axis_pearson_r"]
        ax.set_title(f"{LABELS[m]}\nPC{int(summ.loc[m,'partisan_pc'])}, "
                     f"|r| = {abs(r):.3f}")
        ax.set_xlabel("true ideology")
        if m == "w2v":
            ax.set_ylabel("partisan-axis score")
            ax.legend(frameon=False, fontsize=8, title="party", title_fontsize=8)
    fig.suptitle("Best single PCA axis vs planted ideology (truth-visible identification)",
                 y=1.06)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig1_axis_vs_truth.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 2: single-axis recovery bars ----
    fig, ax = plt.subplots(figsize=(7, 3.8))
    vals = [abs(summ.loc[m, "axis_pearson_r"]) for m in MODELS]
    bars = ax.bar([LABELS[m] for m in MODELS], vals,
                  color=[MCOLOR[m] for m in MODELS], width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", fontsize=9)
    ax.axhline(CEILING, ls="--", lw=1, color="#555555")
    ax.text(-0.42, CEILING + 0.02, "generator ceiling .973", fontsize=8,
            ha="left", color="#555555")
    ax.set_ylim(0, 1.09)
    ax.set_ylabel("|r| best single PC vs true ideology")
    ax.set_title("Single-axis ideology recovery")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig2_axis_recovery.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 3: distance validity, raw vs style-corrected ----
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    x = np.arange(len(MODELS))
    raw = [summ.loc[m, "dist_validity_raw"] for m in MODELS]
    cor = [summ.loc[m, "dist_validity_corrected"] for m in MODELS]
    b1 = ax.bar(x - 0.19, raw, width=0.34, color=[MCOLOR[m] for m in MODELS],
                alpha=0.45, label="raw")
    b2 = ax.bar(x + 0.19, cor, width=0.34, color=[MCOLOR[m] for m in MODELS],
                label="style-corrected")
    for bars, vals in [(b1, raw), (b2, cor)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x, [LABELS[m] for m in MODELS])
    ax.set_ylabel("corr(cosine distance, |ideology gap|)")
    ax.set_title("Distance validity before and after removing the retweet-style axis")
    ax.legend(frameon=False)
    ax.set_ylim(0, 0.9)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig3_distance_validity.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 4: single axis vs supervised probe (the WS4 bridge) ----
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax1v = [abs(summ.loc[m, "axis_pearson_r"]) for m in MODELS]
    prv = [probe.loc[m, "probe_r_full100"] for m in MODELS]
    for i, m in enumerate(MODELS):
        ax.plot([i, i], [ax1v[i], prv[i]], color="#bbbbbb", lw=1.4, zorder=1)
        ax.scatter([i], [ax1v[i]], s=52, facecolors="white",
                   edgecolors=MCOLOR[m], linewidths=1.8, zorder=2)
        ax.scatter([i], [prv[i]], s=52, color=MCOLOR[m], zorder=2)
        ax.text(i + 0.1, prv[i] + 0.014, f"{prv[i]:.3f}", fontsize=8,
                va="bottom")
        if prv[i] - ax1v[i] < 0.05:  # points nearly coincide: push label down
            ax.text(i + 0.1, ax1v[i] - 0.028, f"{ax1v[i]:.3f}", fontsize=8,
                    va="top")
        else:
            ax.text(i + 0.1, ax1v[i] - 0.006, f"{ax1v[i]:.3f}", fontsize=8,
                    va="top")
    ax.axhline(CEILING, ls="--", lw=1, color="#555555")
    ax.text(3.5, CEILING - 0.055, "generator ceiling .973", fontsize=8,
            ha="center", va="top", color="#555555")
    ax.set_ylim(0.4, 1.06)
    ax.set_xticks(range(len(MODELS)), [LABELS[m] for m in MODELS])
    ax.set_ylim(0.4, 1.04)
    ax.set_ylabel("r vs true ideology")
    ax.set_title("Unsupervised best axis (open) vs cross-validated ridge probe (filled)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig4_probe_vs_axis.png", bbox_inches="tight")
    plt.close(fig)

    print("Wrote figs 1-4.")


if __name__ == "__main__":
    main()
