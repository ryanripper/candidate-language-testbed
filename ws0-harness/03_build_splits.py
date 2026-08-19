"""
03_build_splits.py — WS0.3
--------------------------
Fixed evaluation splits, built BLIND and frozen before any workstream scores
anything. Cannot be retrofitted later without invalidating held-out claims.

(a) Tweet-level A/B split per candidate (WS3 Stage C: scores estimated on A,
    behavior measured on B). Originals and retweets are split 50/50
    SEPARATELY within each candidate, so both halves keep retweets — the
    retweet-source-choice model (C1) needs them in split B.
(b) Stratified ~150-candidate subsample for LLM-cost-bound steps (WS2 judge,
    WS3 pilot). Strata: party x chamber x within-party tercile of the BLIND
    TF-IDF partisan-axis score (frozen in ws0/baselines/axis_scores.csv).
    Within-party terciles keep moderates vs extremes represented; the proxy
    is blind by construction (identified from observable D/R separation).

Deterministic: seed 20260725 (session convention), candidates and tweets
processed in sorted order.

Outputs:
  splits.json            subsample ids + strata + A/B summary + file hashes
  tweet_split_ab.parquet tweet_id, candidate_id, split ("A"/"B")
"""

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEED = 20260725
TARGET_N = 150


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ab_split(blind: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for cid, grp in blind.sort_values("tweet_id").groupby("candidate_id", sort=True):
        for is_rt, sub in grp.groupby("is_retweet", sort=True):
            ids = sub["tweet_id"].to_numpy()
            perm = rng.permutation(len(ids))
            # Odd counts: B gets the extra RETWEET (behavior is measured on B
            # and the C1 retweet-source model needs them there); A gets the
            # extra original (the scoring side needs speech most).
            n_b = (len(ids) + 1) // 2 if is_rt else len(ids) // 2
            split = np.where(np.isin(np.arange(len(ids)), perm[:n_b]), "B", "A")
            rows.append(pd.DataFrame(
                {"tweet_id": ids, "candidate_id": cid, "split": split}))
    return pd.concat(rows, ignore_index=True)


def stratified_subsample(meta: pd.DataFrame, scores: pd.DataFrame,
                         rng: np.random.Generator) -> pd.DataFrame:
    df = meta.merge(scores, on="candidate_id", validate="1:1").copy()
    # within-party terciles of the blind ideology proxy
    df["proxy_tercile"] = (
        df.groupby("party")["tfidf_partisan_score"]
        .transform(lambda s: pd.qcut(s, 3, labels=["low", "mid", "high"]))
        .astype(str)
    )
    df["stratum"] = df["party"] + "|" + df["chamber"] + "|" + df["proxy_tercile"]

    sizes = df.groupby("stratum").size()
    # proportional allocation, min 1 per non-empty stratum, largest remainder
    raw = sizes / sizes.sum() * TARGET_N
    alloc = np.maximum(np.floor(raw).astype(int), 1)
    remainder = (raw - np.floor(raw)).sort_values(ascending=False)
    for st in remainder.index:
        if alloc.sum() >= TARGET_N:
            break
        alloc[st] += 1
    while alloc.sum() > TARGET_N:  # trim from the largest allocations
        alloc[alloc.idxmax()] -= 1

    chosen = []
    for st, k in alloc.items():
        pool = df[df["stratum"] == st].sort_values("candidate_id")
        take = min(k, len(pool))
        idx = rng.choice(len(pool), size=take, replace=False)
        chosen.append(pool.iloc[np.sort(idx)])
    return pd.concat(chosen).sort_values("candidate_id").reset_index(drop=True)


def main() -> None:
    rng = np.random.default_rng(SEED)
    blind = pd.read_parquet(HERE / "blind_corpus.parquet")
    meta = pd.read_csv(HERE / "baselines" / "candidate_metadata.csv")
    scores = pd.read_csv(HERE / "baselines" / "axis_scores.csv")

    # ---- (a) tweet-level A/B ----
    ab = ab_split(blind, rng)
    assert len(ab) == len(blind)
    assert ab["tweet_id"].is_unique
    ab_path = HERE / "tweet_split_ab.parquet"
    ab.to_parquet(ab_path, index=False, compression="zstd")

    # per-candidate balance check
    bal = (
        ab.merge(blind[["tweet_id", "is_retweet"]], on="tweet_id")
        .groupby(["candidate_id", "split"])
        .agg(n=("tweet_id", "count"), rt=("is_retweet", "sum"))
        .reset_index()
    )
    piv = bal.pivot(index="candidate_id", columns="split", values=["n", "rt"])
    n_a, n_b = piv["n"]["A"].fillna(0), piv["n"]["B"].fillna(0)
    rt_b_zero = int((piv["rt"]["B"].fillna(0) == 0).sum())

    # ---- (b) stratified subsample ----
    sub = stratified_subsample(meta, scores, rng)
    strata_counts = sub["stratum"].value_counts().sort_index().to_dict()

    splits = {
        "created": date.today().isoformat(),
        "seed": SEED,
        "tweet_ab_split": {
            "file": ab_path.name,
            "sha256": sha256(ab_path),
            "method": ("per-candidate 50/50; originals and retweets split "
                       "separately; on odd counts B gets the extra retweet, "
                       "A the extra original"),
            "n_tweets": int(len(ab)),
            "n_A": int((ab["split"] == "A").sum()),
            "n_B": int((ab["split"] == "B").sum()),
            "max_candidate_imbalance": int(np.abs(n_a - n_b).max()),
            "candidates_with_zero_retweets_in_B": rt_b_zero,
        },
        "subsample_150": {
            "n": int(len(sub)),
            "strata_definition": ("party x chamber x within-party tercile of "
                                  "blind TF-IDF partisan score "
                                  "(baselines/axis_scores.csv)"),
            "allocation": "proportional, min 1 per non-empty stratum",
            "strata_counts": strata_counts,
            "party_counts": sub["party"].value_counts().to_dict(),
            "chamber_counts": sub["chamber"].value_counts().to_dict(),
            "candidate_ids": sub["candidate_id"].tolist(),
        },
    }
    with open(HERE / "splits.json", "w") as fh:
        json.dump(splits, fh, indent=2)

    print(f"A/B split: {splits['tweet_ab_split']['n_A']:,} A / "
          f"{splits['tweet_ab_split']['n_B']:,} B; "
          f"max per-candidate imbalance {splits['tweet_ab_split']['max_candidate_imbalance']}; "
          f"{rt_b_zero} candidates with zero retweets in B")
    print(f"Subsample: {len(sub)} candidates, "
          f"party {splits['subsample_150']['party_counts']}, "
          f"chamber {splits['subsample_150']['chamber_counts']}")
    print("splits.json written.")


if __name__ == "__main__":
    main()
