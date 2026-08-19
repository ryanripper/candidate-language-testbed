"""
03_pca_visualize.py
-------------------
Step 3 of the candidate-language pipeline (still blind to ground truth).

Runs PCA on the candidate vectors (word2vec main, TF-IDF baseline) and
produces the motivating visualizations:

  fig1_scree.png        - explained variance by component, both methods
  fig2_pca_party.png    - PC1 x PC2 scatter colored by party (w2v + tfidf)
  fig3_pc1_dist.png     - PC1 distribution by party (does language separate?)
  fig4_pca_facets.png   - PC1 x PC2 colored by chamber and incumbency
                          (checks PC1 isn't just chamber/incumbency)

Party labels are OBSERVABLE metadata (as they would be for real candidates),
so using them for coloring does not break the blind protocol; the sealed
true_ideology file remains untouched until step 5.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from pathlib import Path as _Path
import os as _os
_HERE = _Path(__file__).resolve().parent.parent   # this analysis folder
_ROOT = _HERE.parents[1]                          # repository root
OUT = str(_HERE / "outputs")
FIG = str(_HERE / "figures")
SEED = 20260720

PARTY_COLORS = {"D": "#3b6fb6", "R": "#c23b3b", "I": "#8a7d4a"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})

def scatter_by(ax, pcs, meta, field, colors, title, evr):
    for val, color in colors.items():
        m = (meta[field] == val).to_numpy()
        ax.scatter(pcs[m, 0], pcs[m, 1], s=12, alpha=0.65, c=color,
                   label=str(val), linewidths=0)
    ax.set_xlabel(f"PC1 ({evr[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({evr[1]:.0%} var)")
    ax.set_title(title)
    ax.legend(frameon=False, title=field)

def main() -> None:
    meta = pd.read_csv(f"{OUT}/candidate_metadata.csv")
    z = np.load(f"{OUT}/candidate_vectors.npz", allow_pickle=True)
    X_w2v, X_tfidf = z["X_w2v"], z["X_tfidf"]

    pca_w2v = PCA(n_components=10, random_state=SEED)
    P_w2v = pca_w2v.fit_transform(X_w2v)
    pca_tfidf = PCA(n_components=10, random_state=SEED)
    P_tfidf = pca_tfidf.fit_transform(X_tfidf)

    # Identify the PARTISAN AXIS empirically rather than assuming it is PC1.
    # Party is observable metadata, so this does not break the blind protocol:
    # among the top PCs, find the one most correlated with a D/R dummy, and
    # orient its sign so Republicans score positive. (Lesson from this corpus:
    # for word2vec the partisan axis turns out to be PC2, not PC1.)
    dr_mask = meta["party"].isin(["D", "R"]).to_numpy()
    party_dummy = (meta.loc[dr_mask, "party"] == "R").to_numpy().astype(float)
    partisan_axis = {}
    for P, name in [(P_w2v, "w2v"), (P_tfidf, "tfidf")]:
        cors = [abs(np.corrcoef(P[dr_mask, k], party_dummy)[0, 1]) for k in range(5)]
        k = int(np.argmax(cors))
        if np.corrcoef(P[dr_mask, k], party_dummy)[0, 1] < 0:
            P[:, k] *= -1
        partisan_axis[name] = k
        print(f"{name}: partisan axis = PC{k+1} "
              f"(|corr with party| by PC: {[f'{c:.2f}' for c in cors]})")

    np.savez_compressed(f"{OUT}/pca_scores.npz",
                        candidate_ids=z["candidate_ids"],
                        P_w2v=P_w2v, P_tfidf=P_tfidf,
                        evr_w2v=pca_w2v.explained_variance_ratio_,
                        evr_tfidf=pca_tfidf.explained_variance_ratio_,
                        partisan_axis_w2v=partisan_axis["w2v"],
                        partisan_axis_tfidf=partisan_axis["tfidf"])

    # ---- Fig 1: scree ----
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ks = np.arange(1, 11)
    ax.plot(ks, pca_w2v.explained_variance_ratio_, "o-", label="word2vec", color="#3b6fb6")
    ax.plot(ks, pca_tfidf.explained_variance_ratio_, "s--", label="TF-IDF + SVD", color="#8a7d4a")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("One dominant axis of variation in candidate language")
    ax.set_xticks(ks)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig1_scree.png"); plt.close(fig)

    # ---- Fig 2: PC1 x PC2 by party, both methods ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    scatter_by(axes[0], P_w2v, meta, "party", PARTY_COLORS,
               "word2vec (trained on corpus)", pca_w2v.explained_variance_ratio_)
    scatter_by(axes[1], P_tfidf, meta, "party", PARTY_COLORS,
               "TF-IDF + SVD baseline", pca_tfidf.explained_variance_ratio_)
    fig.suptitle("910 candidates in language space, colored by party", y=1.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_pca_party.png", bbox_inches="tight"); plt.close(fig)

    # ---- Fig 3: partisan-axis distributions by party (w2v) ----
    kw = partisan_axis["w2v"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for party in ["D", "I", "R"]:
        vals = P_w2v[(meta["party"] == party).to_numpy(), kw]
        ax.hist(vals, bins=40, alpha=0.55, color=PARTY_COLORS[party],
                label=f"{party} (n={len(vals)})")
    ax.set_xlabel(f"PC{kw+1} score (word2vec partisan axis)")
    ax.set_ylabel("Candidates")
    ax.set_title("The partisan axis separates the parties — with overlap in the middle")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_pc1_dist.png"); plt.close(fig)

    # Diagnose the dominant (non-partisan) w2v PC1 against OBSERVABLE
    # behavioral covariates — still blind-legal.
    for col in ["n_tweets", "share_retweets"]:
        r = np.corrcoef(P_w2v[:, 0], meta[col])[0, 1]
        print(f"diagnosis: corr(w2v PC1, {col}) = {r:+.3f}")

    # ---- Fig 4: is PC1 secretly chamber or incumbency? ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    scatter_by(axes[0], P_w2v, meta, "chamber",
               {"House": "#777777", "Senate": "#2a9d8f"},
               "Colored by chamber", pca_w2v.explained_variance_ratio_)
    scatter_by(axes[1], P_w2v, meta.assign(incumbent=meta["incumbent"].map({True: "Incumbent", False: "Challenger"})),
               "incumbent", {"Incumbent": "#e07b39", "Challenger": "#777777"},
               "Colored by incumbency", pca_w2v.explained_variance_ratio_)
    fig.suptitle("PC1 is not an artifact of chamber or incumbency", y=1.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_pca_facets.png", bbox_inches="tight"); plt.close(fig)

    print("Figures written to", FIG)

if __name__ == "__main__":
    main()
