"""
03_analyze.py — WS4 preanalysis, step 3.

For each model (w2v, glove, fasttext, doc2vec, tfidf):

  PCA        top-10 PCs of the candidate matrix; correlate every PC with
             true_ideology and with the known style covariates
             (share_retweets, log n_tweets). Partisan axis = PC with max
             |r| vs truth (truth-visible: this is the informal upper bound,
             not the blind-identification protocol).
  DISTANCES  cosine distance validity corr(D_ij, |ideology gap|) and
             between/within party ratio (D/R candidates only), raw and
             style-corrected (project out the PC most aligned with
             share_retweets — generalization of the 07-20 PC1 correction).

Outputs: pca_correlations.csv, validation_summary.csv, pca_scores_all.npz
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

from pathlib import Path as _Path
import os as _os
_HERE = _Path(__file__).resolve().parent.parent   # this analysis folder
_ROOT = _HERE.parents[1]                          # repository root
OUT = str(_HERE / "outputs")
MODELS = ["w2v", "glove", "fasttext", "doc2vec", "tfidf"]
N_PC = 10
SEED = 20260807


def dist_metrics(D, y, party):
    iu = np.triu_indices(D.shape[0], k=1)
    flat = D[iu]
    gap = np.abs(y[iu[0]] - y[iu[1]])
    dv, _ = stats.pearsonr(flat, gap)
    dr = np.isin(party[iu[0]], ["D", "R"]) & np.isin(party[iu[1]], ["D", "R"])
    same = party[iu[0]] == party[iu[1]]
    w = flat[same & dr].mean()
    b = flat[~same & dr].mean()
    return dv, b / w


def main() -> None:
    meta = pd.read_csv(f"{OUT}/candidate_table.csv")
    y = meta["true_ideology"].to_numpy()
    party = meta["party"].to_numpy()
    rt = meta["share_retweets"].to_numpy()
    logn = np.log(meta["n_tweets"].to_numpy())

    z = np.load(f"{OUT}/candidate_vectors_all.npz", allow_pickle=True)
    assert list(z["candidate_ids"]) == list(meta["candidate_id"])

    pc_rows, summary_rows = [], []
    scores = {"candidate_ids": z["candidate_ids"]}

    for m in MODELS:
        X = z[f"X_{m}"]
        Xc = X - X.mean(axis=0, keepdims=True)
        pca = PCA(n_components=N_PC, random_state=SEED)
        P = pca.fit_transform(Xc)
        evr = pca.explained_variance_ratio_

        r_ideo = np.array([stats.pearsonr(P[:, k], y)[0] for k in range(N_PC)])
        r_rt = np.array([stats.pearsonr(P[:, k], rt)[0] for k in range(N_PC)])
        r_ln = np.array([stats.pearsonr(P[:, k], logn)[0] for k in range(N_PC)])
        for k in range(N_PC):
            pc_rows.append({"model": m, "pc": k + 1, "evr": evr[k],
                            "r_true_ideology": r_ideo[k],
                            "r_share_retweets": r_rt[k],
                            "r_log_n_tweets": r_ln[k]})

        k_ideo = int(np.argmax(np.abs(r_ideo)))
        k_style = int(np.argmax(np.abs(r_rt)))
        axis_r = r_ideo[k_ideo]
        axis_rho = stats.spearmanr(P[:, k_ideo], y).statistic

        # raw distances
        dv_raw, ratio_raw = dist_metrics(cosine_distances(X), y, party)

        # style-corrected: project out the retweet-style PC (if distinct)
        if k_style != k_ideo:
            v = pca.components_[k_style]
            X_corr = Xc - np.outer(Xc @ v, v)
            corrected = True
        else:
            X_corr = Xc
            corrected = False
        dv_c, ratio_c = dist_metrics(cosine_distances(X_corr), y, party)

        summary_rows.append({
            "model": m,
            "partisan_pc": k_ideo + 1,
            "axis_pearson_r": axis_r,
            "axis_spearman_rho": axis_rho,
            "partisan_pc_evr": evr[k_ideo],
            "style_pc": k_style + 1,
            "style_pc_r_retweets": r_rt[k_style],
            "style_eq_partisan": not corrected,
            "dist_validity_raw": dv_raw,
            "dist_validity_corrected": dv_c,
            "ratio_bw_within_raw": ratio_raw,
            "ratio_bw_within_corrected": ratio_c,
        })
        scores[f"P_{m}"] = P
        scores[f"axis_{m}"] = k_ideo
        scores[f"style_{m}"] = k_style
        print(f"{m:9s} partisan=PC{k_ideo+1} r={axis_r:+.3f}  "
              f"style=PC{k_style+1} (r_rt={r_rt[k_style]:+.3f})  "
              f"dv {dv_raw:+.3f}->{dv_c:+.3f}  ratio {ratio_raw:.3f}->{ratio_c:.3f}")

    pd.DataFrame(pc_rows).to_csv(f"{OUT}/pca_correlations.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(f"{OUT}/validation_summary.csv", index=False)
    np.savez_compressed(f"{OUT}/pca_scores_all.npz", **scores)
    print("\nSaved pca_correlations.csv, validation_summary.csv, pca_scores_all.npz")


if __name__ == "__main__":
    main()
