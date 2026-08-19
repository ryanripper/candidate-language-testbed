"""
04_linear_probe.py — WS4 preanalysis, step 4 (bridge to WS4).

Simple supervised probe: 5-fold cross-validated ridge regression predicting
true_ideology from (a) each model's full 100-dim candidate vectors and
(b) its top-10 PCs. Out-of-fold Pearson r and R^2.

Motivation: the single-best-PC number (step 3) understates models whose
ideology signal is spread across several components (GloVe). The probe
measures TOTAL linearly-decodable signal — exactly the quantity WS4's
supervised prediction will exploit. Splits are by candidate (rows are
candidates), consistent with the WS4 design note that all splits must be
candidate-level.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from pathlib import Path as _Path
import os as _os
_HERE = _Path(__file__).resolve().parent.parent   # this analysis folder
_ROOT = _HERE.parents[1]                          # repository root
OUT = str(_HERE / "outputs")
MODELS = ["w2v", "glove", "fasttext", "doc2vec", "tfidf"]
SEED = 20260807
ALPHAS = np.logspace(-3, 4, 30)


def cv_probe(X, y, seed=SEED):
    oof = np.zeros_like(y)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        sc = StandardScaler().fit(X[tr])
        model = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), y[tr])
        oof[te] = model.predict(sc.transform(X[te]))
    r, _ = stats.pearsonr(oof, y)
    r2 = 1 - np.sum((y - oof) ** 2) / np.sum((y - y.mean()) ** 2)
    return r, r2


def main() -> None:
    meta = pd.read_csv(f"{OUT}/candidate_table.csv")
    y = meta["true_ideology"].to_numpy()
    z = np.load(f"{OUT}/candidate_vectors_all.npz", allow_pickle=True)

    rows = []
    for m in MODELS:
        X = z[f"X_{m}"]
        r_full, r2_full = cv_probe(X, y)
        P = PCA(n_components=10, random_state=SEED).fit_transform(
            X - X.mean(axis=0, keepdims=True))
        r_pc, r2_pc = cv_probe(P, y)
        rows.append({"model": m,
                     "probe_r_full100": r_full, "probe_R2_full100": r2_full,
                     "probe_r_top10pc": r_pc, "probe_R2_top10pc": r2_pc})
        print(f"{m:9s} full-100d: r={r_full:+.3f} R2={r2_full:.3f}   "
              f"top-10 PCs: r={r_pc:+.3f} R2={r2_pc:.3f}")

    pd.DataFrame(rows).to_csv(f"{OUT}/linear_probe.csv", index=False)
    print("\nSaved linear_probe.csv")


if __name__ == "__main__":
    main()
