"""
04_distances.py
---------------
Step 4 of the candidate-language pipeline (still blind to ground truth).

Computes pairwise cosine distances between candidate vectors (word2vec,
retweets included), in two variants:

  RAW       - distances on the averaged word2vec vectors as-is
  CORRECTED - the dominant PC of the raw vectors is a STYLE confound
              (it tracks each candidate's retweet share, an observable
              covariate — diagnosed in step 3), so we project that
              component out before measuring distances.

Reports for each variant:
  - the 10 MOST ALIKE candidate pairs (smallest cosine distance)
  - the 10 MOST UNALIKE candidate pairs (largest cosine distance)
  - within-party vs between-party distance summary
  - per-candidate "loners": candidates farthest from everyone on average

Figures:
  fig5_distance_heatmap.png - corrected 910x910 distance matrix, rows/cols
                              sorted by the partisan axis (block structure)
  fig6_distance_dists.png   - within- vs between-party distance histograms,
                              raw vs corrected
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

from pathlib import Path as _Path
import os as _os
_HERE = _Path(__file__).resolve().parent.parent   # this analysis folder
_ROOT = _HERE.parents[1]                          # repository root
OUT = str(_HERE / "outputs")
FIG = str(_HERE / "figures")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
})

def pair_table(D, meta, idx_pairs):
    rows = []
    for i, j in idx_pairs:
        a, b = meta.iloc[i], meta.iloc[j]
        rows.append({
            "candidate_a": f"{a.candidate_name} ({a.party}-{a.state}, {a.chamber})",
            "candidate_b": f"{b.candidate_name} ({b.party}-{b.state}, {b.chamber})",
            "same_party": a.party == b.party,
            "cosine_distance": round(float(D[i, j]), 4),
        })
    return pd.DataFrame(rows)

def analyze(D, meta, label):
    """Report alike/unalike pairs and party structure for one distance matrix."""
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    flat = D[iu]

    order = np.argsort(flat)
    closest = [(iu[0][k], iu[1][k]) for k in order[:10]]
    farthest = [(iu[0][k], iu[1][k]) for k in order[-10:][::-1]]

    t_close = pair_table(D, meta, closest)
    t_far = pair_table(D, meta, farthest)
    t_close.to_csv(f"{OUT}/most_alike_pairs_{label}.csv", index=False)
    t_far.to_csv(f"{OUT}/most_unalike_pairs_{label}.csv", index=False)
    print(f"\n===== {label.upper()} =====")
    print("MOST ALIKE:\n", t_close.to_string(index=False))
    print("\nMOST UNALIKE:\n", t_far.to_string(index=False))

    party = meta["party"].to_numpy()
    same = party[iu[0]] == party[iu[1]]
    dr = np.isin(party[iu[0]], ["D", "R"]) & np.isin(party[iu[1]], ["D", "R"])
    w, b = flat[same & dr].mean(), flat[~same & dr].mean()
    print(f"\nMean cosine distance  within-party: {w:.4f}")
    print(f"Mean cosine distance between-party: {b:.4f}")
    print(f"Ratio between/within: {b/w:.2f}")

    mean_dist = D.sum(axis=1) / (n - 1)
    meta_out = meta.assign(mean_distance_to_field=mean_dist)
    meta_out.to_csv(f"{OUT}/candidate_mean_distances_{label}.csv", index=False)
    cols = ["candidate_name", "party", "state", "chamber", "mean_distance_to_field"]
    print("\nFarthest from the field:\n",
          meta_out.nlargest(5, "mean_distance_to_field")[cols].to_string(index=False))
    print("\nClosest to the center of the field:\n",
          meta_out.nsmallest(5, "mean_distance_to_field")[cols].to_string(index=False))
    return flat, same, dr

def main() -> None:
    meta = pd.read_csv(f"{OUT}/candidate_metadata.csv")
    z = np.load(f"{OUT}/candidate_vectors.npz", allow_pickle=True)
    X = z["X_w2v"]

    # ---- RAW distances ----
    D_raw = cosine_distances(X)

    # ---- STYLE-CORRECTED distances ----
    # Step 3's diagnosis: the dominant PC of the raw vectors tracks retweet
    # share (r ~ 0.96), an observable stylistic covariate, not content.
    # Project that single direction out of the centered vectors.
    Xc = X - X.mean(axis=0, keepdims=True)
    from sklearn.decomposition import PCA
    v1 = PCA(n_components=1, random_state=20260720).fit(Xc).components_[0]
    X_corr = Xc - np.outer(Xc @ v1, v1)
    D_corr = cosine_distances(X_corr)
    np.save(f"{OUT}/distance_matrix_corrected.npy", D_corr)

    flat_raw, same, dr = analyze(D_raw, meta, "raw")
    flat_corr, _, _ = analyze(D_corr, meta, "corrected")

    # ---- Fig 5: sorted heatmap (corrected distances) ----
    pz = np.load(f"{OUT}/pca_scores.npz", allow_pickle=True)
    axis_k = int(pz["partisan_axis_w2v"])
    sort_idx = np.argsort(pz["P_w2v"][:, axis_k])
    Ds = D_corr[np.ix_(sort_idx, sort_idx)]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(Ds, cmap="magma_r", interpolation="nearest")
    ax.set_title("Cosine distance after style correction,\ncandidates sorted by partisan axis")
    ax.set_xlabel("candidates (D side → R side)")
    ax.set_ylabel("candidates (D side → R side)")
    fig.colorbar(im, ax=ax, label="cosine distance", shrink=0.85)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig5_distance_heatmap.png"); plt.close(fig)

    # ---- Fig 6: within vs between distributions, raw vs corrected ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, flat, title in [(axes[0], flat_raw, "Raw word2vec distances"),
                            (axes[1], flat_corr, "Style-corrected distances")]:
        ax.hist(flat[same & dr], bins=60, alpha=0.6, density=True,
                color="#2a9d8f", label="same party")
        ax.hist(flat[~same & dr], bins=60, alpha=0.6, density=True,
                color="#7d5ba6", label="cross party")
        ax.set_xlabel("cosine distance")
        ax.set_ylabel("density")
        ax.set_title(title)
        ax.legend(frameon=False)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.suptitle("Removing the retweet-style axis sharpens the party signal", y=1.04)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig6_distance_dists.png", bbox_inches="tight"); plt.close(fig)

    print("\nFigures written.")

if __name__ == "__main__":
    main()
