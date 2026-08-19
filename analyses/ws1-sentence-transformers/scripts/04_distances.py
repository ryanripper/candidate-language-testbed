"""
04_distances.py — WS1 E1.3 distances + blind diagnostics + Tier-C gate
----------------------------------------------------------------------
BLIND. Per preregistration §2:

  * centroid cosine distances for raw / centered / whitened / corrected
  * distributional distances (energy, Euclidean; MMD, RBF with median-
    heuristic bandwidth on a 2,000-tweet sample, seed 20260725) for the
    `centered` and `corrected` spaces; unbiased within-cloud terms; full
    clouds, no subsampling
  * blind between/within party ratios for every matrix
  * evaluates the D2 blind Tier-C gate (preregistration §2a) — strictly
    before unsealing

Writes: outputs/D_tier{X}_{variant}_{rep}.npy (910x910 float32),
        outputs/blind_diagnostics.csv, outputs/tierC_gate.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
WS0 = HERE.parents[1] / "ws0-harness"
sys.path.insert(0, str(WS0))
from metrics import within_between_ratio  # noqa: E402

SEED = 20260725
BLOCK = 512
TFIDF_RATIO_BAR = None  # loaded from baseline_validation.csv below


def cosine_D(C: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(C, axis=1, keepdims=True)
    n[n == 0] = 1.0
    U = C / n
    D = 1.0 - U @ U.T
    np.fill_diagonal(D, 0.0)
    return np.clip(D, 0.0, None)


def cloud_distances(X: np.ndarray, codes: np.ndarray, n_cand: int,
                    sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Energy distance and MMD (RBF) between all candidate clouds.

    One pass of blocked gemms accumulates, for every candidate pair (i,j),
    the summed Euclidean distance and summed RBF kernel over ordered tweet
    pairs; energy/MMD then follow from segment means (unbiased within-cloud
    terms: ordered off-diagonal sums / n(n-1))."""
    order = np.argsort(codes, kind="stable")
    Xs = np.ascontiguousarray(X[order], dtype=np.float32)
    cs = codes[order]
    counts = np.bincount(cs, minlength=n_cand).astype(np.float64)
    bounds = np.concatenate([[0], np.cumsum(counts).astype(int)])
    starts = bounds[:-1]
    N = Xs.shape[0]
    sq_norms = (Xs ** 2).sum(axis=1)
    Sd = np.zeros((n_cand, n_cand))
    Sk = np.zeros((n_cand, n_cand))
    gamma = 1.0 / (2.0 * sigma ** 2)
    t0 = time.time()
    for lo in range(0, N, BLOCK):
        hi = min(lo + BLOCK, N)
        G = Xs[lo:hi] @ Xs.T
        sq = sq_norms[lo:hi, None] + sq_norms[None, :] - 2.0 * G
        np.clip(sq, 0.0, None, out=sq)
        d = np.sqrt(sq)
        k = np.exp(-gamma * sq)
        # column-reduce to candidates (columns grouped after sort)
        dcol = np.add.reduceat(d, starts, axis=1)
        kcol = np.add.reduceat(k, starts, axis=1)
        # row-reduce into (n_cand, n_cand)
        np.add.at(Sd, cs[lo:hi], dcol)
        np.add.at(Sk, cs[lo:hi], kcol)
    print(f"    pass done in {(time.time()-t0)/60:.1f} min", flush=True)

    nn = counts[:, None] * counts[None, :]
    M = Sd / nn                                    # mean cross distance
    Km = Sk / nn                                   # mean cross kernel
    n = counts
    w_d = np.diag(Sd) / (n * (n - 1))              # unbiased within (dist)
    w_k = (np.diag(Sk) - n) / (n * (n - 1))        # unbiased within (kernel)
    E = 2.0 * M - w_d[:, None] - w_d[None, :]
    np.fill_diagonal(E, 0.0)
    MMD2 = w_k[:, None] + w_k[None, :] - 2.0 * Km
    np.fill_diagonal(MMD2, 0.0)
    MMD = np.sqrt(np.clip(MMD2, 0.0, None))
    return np.clip(E, 0.0, None), MMD


def median_sigma(X: np.ndarray, rng: np.random.Generator) -> float:
    idx = rng.choice(X.shape[0], size=2000, replace=False)
    S = np.ascontiguousarray(X[idx], dtype=np.float32)
    a2 = (S ** 2).sum(axis=1)
    sq = a2[:, None] + a2[None, :] - 2.0 * (S @ S.T)
    np.clip(sq, 0.0, None, out=sq)
    iu = np.triu_indices(2000, k=1)
    return float(np.median(np.sqrt(sq[iu])))


def main() -> None:
    meta = pd.read_csv(WS0 / "baselines" / "candidate_metadata.csv")
    cand_order = meta["candidate_id"].tolist()
    party = meta["party"].to_numpy()
    cid_index = {c: i for i, c in enumerate(cand_order)}
    n_cand = len(cand_order)

    tiers = sorted(p.stem[-1] for p in (HERE / "intermediate").glob("emb_tier?.npz"))
    rows = []

    def record(tier, variant, rep, D):
        np.save(HERE / "outputs" / f"D_tier{tier}_{variant}_{rep}.npy",
                D.astype(np.float32))
        wb = within_between_ratio(D, party)
        rows.append({"tier": tier, "variant": variant, "rep": rep, **wb})
        print(f"  tier {tier} {variant}/{rep}: ratio {wb['ratio']:.4f}",
              flush=True)

    for tier in tiers:
        cz = np.load(HERE / "outputs" / f"centroids_tier{tier}.npz")
        corr = np.load(HERE / "outputs" / f"corrected_tier{tier}.npz")
        record(tier, "raw", "centroid_cosine", cosine_D(cz["C_raw"]))
        record(tier, "centered", "centroid_cosine", cosine_D(cz["C_centered"]))
        record(tier, "whitened", "centroid_cosine", cosine_D(cz["C_whitened"]))
        record(tier, "corrected", "centroid_cosine", cosine_D(corr["C_corrected"]))

        for variant, path in [
            ("centered", HERE / "intermediate" / f"emb_tier{tier}.npz"),
            ("corrected", HERE / "intermediate" / f"emb_tier{tier}_corrected.npz"),
        ]:
            z = np.load(path, allow_pickle=True)
            X = z["X"].astype(np.float32)
            if variant == "centered":
                X = X - X.mean(axis=0, keepdims=True)
            codes = np.array([cid_index[c] for c in z["candidate_id"]])
            rng = np.random.default_rng(SEED)
            sigma = median_sigma(X, rng)
            print(f"  tier {tier} {variant}: sigma={sigma:.4f}; "
                  f"energy+MMD pass over {X.shape}", flush=True)
            E, MMD = cloud_distances(X, codes, n_cand, sigma)
            record(tier, variant, "energy", E)
            record(tier, variant, "mmd", MMD)

    diag = pd.DataFrame(rows)
    diag.to_csv(HERE / "outputs" / "blind_diagnostics.csv", index=False)

    # ---- D2 blind Tier-C gate (preregistration §2a) ----
    bv = pd.read_csv(WS0 / "baselines" / "baseline_validation.csv")
    tfidf_ratio = float(bv.loc[bv["measure"].str.startswith(
        "TF-IDF between/within"), "ws0_value"].iloc[0])
    ax = pd.read_csv(WS0 / "baselines" / "axis_scores.csv")
    dr = np.isin(party, ["D", "R"])
    y = (party[dr] == "R").astype(float)
    tfidf_dr = abs(np.corrcoef(
        ax["tfidf_partisan_score"].to_numpy()[dr], y)[0, 1])
    b_ratio = diag.query("tier == 'B'")["ratio"].max()
    b_dr = float(np.load(HERE / "outputs" / "corrected_tierB.npz")["dr_abscorr"])
    gate = bool(b_ratio >= tfidf_ratio or b_dr > tfidf_dr)
    result = {"tierB_max_blind_ratio": float(b_ratio),
              "tfidf_ratio_reference": tfidf_ratio,
              "tierB_corrected_axis_dr_abscorr": b_dr,
              "tfidf_axis_dr_abscorr": float(tfidf_dr),
              "criterion_1_ratio": bool(b_ratio >= tfidf_ratio),
              "criterion_2_axis": bool(b_dr > tfidf_dr),
              "tier_C_runs": gate}
    with open(HERE / "outputs" / "tierC_gate.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print("TIER-C GATE:", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
