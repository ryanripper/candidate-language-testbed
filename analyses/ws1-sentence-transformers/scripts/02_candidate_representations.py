"""
02_candidate_representations.py — WS1 E1.2 (variants) + E1.3 (centroids)
------------------------------------------------------------------------
BLIND. For every embedded tier found in intermediate/:

  * builds tweet-level anisotropy variants: raw / centered / whitened
    (PCA whitening, eigenvalue > 1e-10, eps = 1e-8) — preregistration §2
  * per-candidate centroids in each variant (rows ordered exactly as
    ws0-harness/baselines/candidate_metadata.csv)
  * PCA (10 comps) of the centroid matrix; blind partisan-axis
    identification + orientation via ws0-harness/metrics.py (observable D/R only)
  * figure: scree + per-PC blind D/R separation ("axis diagnosis")

Writes: outputs/centroids_tier{X}.npz, outputs/pca_tier{X}.npz,
        outputs/blind_axis_scores.csv (accumulates tiers),
        figures/fig1_axis_diagnosis.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
WS0 = HERE.parents[1] / "ws0-harness"
sys.path.insert(0, str(WS0))
from metrics import identify_partisan_axis, orient_axis  # noqa: E402

SEED = 20260725
EIG_FLOOR, EPS = 1e-10, 1e-8


def centroids(X: np.ndarray, codes: np.ndarray, n_cand: int) -> np.ndarray:
    S = np.zeros((n_cand, X.shape[1]), dtype=np.float64)
    np.add.at(S, codes, X)
    counts = np.bincount(codes, minlength=n_cand).astype(np.float64)
    return S / counts[:, None]


def main() -> None:
    meta = pd.read_csv(WS0 / "baselines" / "candidate_metadata.csv")
    cand_order = meta["candidate_id"].tolist()
    party = meta["party"].to_numpy()
    cid_index = {c: i for i, c in enumerate(cand_order)}

    tiers = sorted(p.stem[-1] for p in (HERE / "intermediate").glob("emb_tier?.npz"))
    print("tiers found:", tiers)

    axis_rows, diag = [], {}
    for tier in tiers:
        z = np.load(HERE / "intermediate" / f"emb_tier{tier}.npz",
                    allow_pickle=True)
        X = z["X"].astype(np.float32)
        codes = np.array([cid_index[c] for c in z["candidate_id"]])
        mu = X.mean(axis=0, keepdims=True)
        Xc = X - mu

        # tweet-level whitening basis is large; whiten via centroid path:
        # centroids of whitened tweets == whitening transform applied to
        # centered centroids (linear map), so compute the map from Xc.
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        eig = (s ** 2) / (Xc.shape[0] - 1)
        keep = eig > EIG_FLOOR
        W = Vt[keep].T / np.sqrt(eig[keep] + EPS)      # d x k whitening map

        C_raw = centroids(X, codes, len(cand_order))
        C_cen = C_raw - mu                              # centered centroids
        C_whi = C_cen @ W

        # PCA on centroid matrix (10 comps) + blind axis
        from sklearn.decomposition import PCA
        P = PCA(n_components=10, random_state=SEED).fit(C_cen)
        comps = P.components_                           # 10 x d
        scores = (C_cen - C_cen.mean(axis=0)) @ comps.T
        k = identify_partisan_axis(scores, party)
        axis = orient_axis(scores[:, k], party)
        dr = np.isin(party, ["D", "R"])
        y = (party[dr] == "R").astype(float)
        pc_dr = [abs(np.corrcoef(scores[dr, j], y)[0, 1]) for j in range(10)]
        print(f"tier {tier}: partisan axis PC{k+1}, blind |D/R corr| "
              f"{pc_dr[k]:.4f}, evr {P.explained_variance_ratio_.round(3)}")

        np.savez_compressed(HERE / "outputs" / f"centroids_tier{tier}.npz",
                            candidate_ids=np.array(cand_order),
                            C_raw=C_raw.astype(np.float32),
                            C_centered=C_cen.astype(np.float32),
                            C_whitened=C_whi.astype(np.float32),
                            corpus_mean=mu.astype(np.float32))
        np.savez_compressed(HERE / "outputs" / f"pca_tier{tier}.npz",
                            candidate_ids=np.array(cand_order),
                            P=scores.astype(np.float32),
                            components=comps.astype(np.float32),
                            explained_variance_ratio=P.explained_variance_ratio_,
                            partisan_axis=k, pc_dr_abscorr=np.array(pc_dr))
        for i, c in enumerate(cand_order):
            axis_rows.append({"candidate_id": c, "tier": tier,
                              "space": "centered", "axis_pc": k + 1,
                              "blind_axis_score": float(axis[i])})
        diag[tier] = {"evr": P.explained_variance_ratio_, "pc_dr": pc_dr,
                      "k": k}

    pd.DataFrame(axis_rows).to_csv(HERE / "outputs" / "blind_axis_scores.csv",
                                   index=False)

    # ---- figure 1: scree + axis diagnosis ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=150)
    colors = {"A": "#4269d0", "B": "#efb118", "C": "#ff725c"}
    names = {"A": "Model2Vec potion-8M", "B": "MiniLM-L6-v2",
             "C": "bge-small-en-v1.5"}
    x = np.arange(1, 11)
    for tier in tiers:
        axes[0].plot(x, diag[tier]["evr"], "o-", ms=4, lw=1.5,
                     color=colors[tier], label=names[tier])
        axes[1].plot(x, diag[tier]["pc_dr"], "o-", ms=4, lw=1.5,
                     color=colors[tier], label=names[tier])
        k = diag[tier]["k"]
        axes[1].scatter([k + 1], [diag[tier]["pc_dr"][k]], s=140,
                        facecolors="none", edgecolors=colors[tier], lw=1.8)
    axes[0].set(xlabel="principal component", ylabel="explained variance ratio",
                title="Scree (candidate centroids)", xticks=x)
    axes[1].set(xlabel="principal component",
                ylabel="|corr| with observable D/R label",
                title="Blind axis diagnosis (circled = selected)", xticks=x,
                ylim=(0, 1))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("WS1 — sentence-transformer spaces: variance vs partisan signal",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "fig1_axis_diagnosis.png",
                bbox_inches="tight")
    print("wrote outputs + figures/fig1_axis_diagnosis.png")


if __name__ == "__main__":
    main()
