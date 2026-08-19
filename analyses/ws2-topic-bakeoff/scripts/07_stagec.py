"""
07_stagec.py — WS2 Stage C: topic-conditioned candidate distances (BLIND parts)
-------------------------------------------------------------------------------
Importable machinery + CLI. Everything here reads blind data only; the
distance validity numbers are computed later, at the single unseal step in
08_unseal_validate.py (preregistration §7).

Space (preregistration §7/§8): Tier A Model2Vec tweet vectors, corpus-
centered; WS2-replicated confound gate on the candidate-centroid PCA
(top-10 PCs vs retweet share, log10 volume, topic-mix entropy computed from
the given entrant's assignments); style PCs (max |r| >= 0.6, non-partisan)
projected out of tweet vectors -> corrected space.

Per-(candidate, topic) centroids for candidates with >= 5 tweets in that
topic; within-topic cosine distance matrices over qualifying candidates.

CLI: python 07_stagec.py <entrant>   -> writes outputs/stagec_<entrant>.npz
                                        + outputs/stagec_confound_<entrant>.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
WS0 = HERE.parents[1] / "ws0-harness"
ST = HERE.parent / "ws1-sentence-transformers"
SEED = 20260726
MIN_TWEETS = 5


def load_space() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    emb = np.load(ST / "intermediate" / "emb_tierA.npz", allow_pickle=True)
    X = emb["X"].astype(np.float64)
    cand = emb["candidate_id"].astype(str)
    meta = pd.read_csv(WS0 / "baselines" / "candidate_metadata.csv")
    return X - X.mean(0), cand, meta


def confound_gate(Xc, cand, meta, labels) -> tuple[np.ndarray, pd.DataFrame, list]:
    """Replicate WS1 §8 gate; returns corrected tweet vectors + report."""
    order = meta["candidate_id"].astype(str).tolist()
    cidx = {c: i for i, c in enumerate(order)}
    n = len(order)
    # candidate centroids
    C = np.zeros((n, Xc.shape[1]))
    for c, g in pd.DataFrame({"c": cand}).groupby("c").groups.items():
        C[cidx[c]] = Xc[np.asarray(g)].mean(0)
    C -= C.mean(0)
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    P = U[:, :10] * S[:10]

    # covariates
    share = meta["share_retweets"].to_numpy(float)
    vol = np.log10(meta["n_tweets"].to_numpy(float))
    ent = np.zeros(n)
    df = pd.DataFrame({"c": cand, "t": labels})
    for c, g in df.groupby("c"):
        p = g["t"].value_counts(normalize=True).to_numpy()
        ent[cidx[c]] = -(p * np.log(p + 1e-12)).sum()
    party = meta["party"].to_numpy()
    dr = np.isin(party, ["D", "R"])
    y = (party[dr] == "R").astype(float)

    rows, style_dirs = [], []
    for k in range(10):
        rs = {"pc": k + 1,
              "r_retweet_share": np.corrcoef(P[:, k], share)[0, 1],
              "r_log_volume": np.corrcoef(P[:, k], vol)[0, 1],
              "r_topic_entropy": np.corrcoef(P[:, k], ent)[0, 1],
              "r_party_pb": np.corrcoef(P[dr, k], y)[0, 1]}
        mx = max(abs(rs["r_retweet_share"]), abs(rs["r_log_volume"]),
                 abs(rs["r_topic_entropy"]))
        is_style = mx >= 0.6 and not (abs(rs["r_party_pb"]) > mx)
        rs["style_axis"] = is_style
        rows.append(rs)
        if is_style:
            style_dirs.append(Vt[k])
    rep = pd.DataFrame(rows)
    if style_dirs:
        V = np.stack(style_dirs)
        Q, _ = np.linalg.qr(V.T)
        Xcorr = Xc - (Xc @ Q) @ Q.T
    else:
        Xcorr = Xc
    return Xcorr, rep, [int(r["pc"]) for _, r in rep.iterrows() if r["style_axis"]]


def topic_distances(Xcorr, cand, meta, labels) -> dict:
    """Per-topic within-topic cosine distance matrices (qualifying candidates)."""
    order = meta["candidate_id"].astype(str).tolist()
    cidx = {c: i for i, c in enumerate(order)}
    out = {}
    df = pd.DataFrame({"c": cand, "t": labels, "i": np.arange(len(cand))})
    for t, gt in df.groupby("t"):
        sizes = gt.groupby("c").size()
        qual = sizes[sizes >= MIN_TWEETS].index.tolist()
        if len(qual) < 30:      # need enough candidates for a meaningful matrix
            continue
        cent = np.stack([Xcorr[gt.loc[gt.c == c, "i"].to_numpy()].mean(0)
                         for c in qual])
        cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
        D = 1.0 - cent @ cent.T
        np.fill_diagonal(D, 0.0)
        out[int(t)] = {"D": D, "cand_rows": np.array([cidx[c] for c in qual])}
    return out


def main(entrant: str) -> None:
    labels = np.load(HERE / "outputs" / f"assignments_{entrant}.npy")
    Xc, cand, meta = load_space()
    Xcorr, rep, style_pcs = confound_gate(Xc, cand, meta, labels)
    rep.to_csv(HERE / "outputs" / f"stagec_confound_{entrant}.csv", index=False)
    print(f"style PCs projected out: {style_pcs}")

    # overall corrected centroid distances (all topics pooled) for reference
    order = meta["candidate_id"].astype(str).tolist()
    cidx = {c: i for i, c in enumerate(order)}
    C = np.zeros((len(order), Xcorr.shape[1]))
    for c, g in pd.DataFrame({"c": cand}).groupby("c").groups.items():
        C[cidx[c]] = Xcorr[np.asarray(g)].mean(0)
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    D_all = 1.0 - Cn @ Cn.T
    np.fill_diagonal(D_all, 0.0)

    per_topic = topic_distances(Xcorr, cand, meta, labels)
    npz = {"D_overall": D_all, "style_pcs": np.array(style_pcs, dtype=int)}
    for t, v in per_topic.items():
        npz[f"D_{t}"] = v["D"]
        npz[f"rows_{t}"] = v["cand_rows"]
    np.savez_compressed(HERE / "outputs" / f"stagec_{entrant}.npz", **npz)
    print(f"topics with distance matrices: {sorted(per_topic)} "
          f"({len(per_topic)} of {len(set(labels.tolist()))})")


if __name__ == "__main__":
    main(sys.argv[1])
