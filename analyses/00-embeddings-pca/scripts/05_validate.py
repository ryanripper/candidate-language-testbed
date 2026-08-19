"""
05_validate.py
--------------
Step 5 of the candidate-language pipeline: UNSEAL the ground truth.

The synthetic corpus was generated with a planted true_ideology in [-1, 1]
per candidate. Everything up to this point was done blind. Now we score the
blind analysis:

  - Pearson/Spearman correlation between each method's partisan axis and
    true_ideology
  - the same for the non-partisan PC1 of word2vec (what was that axis?)
  - distance-level validation: does embedding distance track |ideology gap|?

Figure:
  fig7_validation.png - partisan axis vs true_ideology for both methods
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics.pairwise import cosine_distances

from pathlib import Path as _Path
import os as _os
_HERE = _Path(__file__).resolve().parent.parent   # this analysis folder
_ROOT = _HERE.parents[1]                          # repository root
OUT = str(_HERE / "outputs")
FIG = str(_HERE / "figures")

PARTY_COLORS = {"D": "#3b6fb6", "R": "#c23b3b", "I": "#8a7d4a"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})

def main() -> None:
    meta = pd.read_csv(f"{OUT}/candidate_metadata.csv")
    truth = pd.read_csv(f"{OUT}/ground_truth_SEALED.csv")
    df = meta.merge(truth, on="candidate_id", validate="1:1")

    pz = np.load(f"{OUT}/pca_scores.npz", allow_pickle=True)
    assert list(pz["candidate_ids"]) == list(df["candidate_id"])
    P_w2v, P_tfidf = pz["P_w2v"], pz["P_tfidf"]
    k_w2v, k_tfidf = int(pz["partisan_axis_w2v"]), int(pz["partisan_axis_tfidf"])

    y = df["true_ideology"].to_numpy()

    results = {}
    for name, score in [
        (f"word2vec PC{k_w2v+1} (partisan axis)", P_w2v[:, k_w2v]),
        (f"word2vec PC1 (dominant axis)", P_w2v[:, 0]),
        (f"TF-IDF PC{k_tfidf+1} (partisan axis)", P_tfidf[:, k_tfidf]),
    ]:
        r, _ = stats.pearsonr(score, y)
        rho, _ = stats.spearmanr(score, y)
        results[name] = (r, rho)
        print(f"{name:42s}  Pearson r = {r:+.3f}   Spearman rho = {rho:+.3f}")

    # ---- Distance-level validation (raw and style-corrected) ----
    z = np.load(f"{OUT}/candidate_vectors.npz", allow_pickle=True)
    D_raw = cosine_distances(z["X_w2v"])
    D_corr = np.load(f"{OUT}/distance_matrix_corrected.npy")
    iu = np.triu_indices(D_raw.shape[0], k=1)
    gap = np.abs(y[iu[0]] - y[iu[1]])
    r_dist, _ = stats.pearsonr(D_raw[iu], gap)
    r_dist_c, _ = stats.pearsonr(D_corr[iu], gap)
    print(f"\nPairwise: corr(raw cosine distance, |true ideology gap|)       = {r_dist:+.3f}")
    print(f"Pairwise: corr(corrected cosine distance, |true ideology gap|) = {r_dist_c:+.3f}")

    pd.DataFrame(
        [{"measure": k, "pearson_r": v[0], "spearman_rho": v[1]} for k, v in results.items()]
        + [{"measure": "raw cosine distance vs |ideology gap| (pairwise)",
            "pearson_r": r_dist, "spearman_rho": np.nan},
           {"measure": "corrected cosine distance vs |ideology gap| (pairwise)",
            "pearson_r": r_dist_c, "spearman_rho": np.nan}]
    ).to_csv(f"{OUT}/validation_results.csv", index=False)

    # ---- Fig 7: validation scatter ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)
    for ax, P, k, name in [
        (axes[0], P_w2v, k_w2v, "word2vec"),
        (axes[1], P_tfidf, k_tfidf, "TF-IDF + SVD"),
    ]:
        for party, color in PARTY_COLORS.items():
            m = (df["party"] == party).to_numpy()
            ax.scatter(y[m], P[m, k], s=12, alpha=0.65, c=color, label=party,
                       linewidths=0)
        r, _ = stats.pearsonr(P[:, k], y)
        ax.set_xlabel("true ideology (planted)")
        ax.set_ylabel(f"PC{k+1} score ({name} partisan axis)")
        ax.set_title(f"{name}:  r = {r:+.3f}")
        ax.legend(frameon=False, title="party")
    fig.suptitle("Unsealing the ground truth: did the blind analysis recover ideology?", y=1.03)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig7_validation.png", bbox_inches="tight"); plt.close(fig)

    # What is word2vec PC1 if not ideology? Correlate with observables.
    print("\nWhat is word2vec PC1? Correlations with observables:")
    for col in ["n_tweets", "share_retweets"]:
        r, _ = stats.pearsonr(P_w2v[:, 0], df[col])
        print(f"  corr(PC1, {col}) = {r:+.3f}")

    print("\nFigure written.")

if __name__ == "__main__":
    main()
