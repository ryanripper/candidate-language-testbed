"""
03_llm_sample.py — WS2 entrant 5, stage (i) sample construction, BLIND
----------------------------------------------------------------------
Stratified 2,000-tweet sample, proportional by party × is_retweet,
seed 20260726 (preregistration §2). Split into two disjoint 1,000-tweet
halves (H1/H2) for the two independent taxonomy runs; the union is the
labeling sample.

Writes : outputs/llm_sample.csv  (row_idx, tweet_id, half, text)
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
WS0 = HERE.parents[1] / "ws0-harness"
SEED = 20260726
N = 2000

corpus = pd.read_parquet(WS0 / "blind_corpus.parquet").reset_index(drop=True)
corpus["row_idx"] = corpus.index
rng = np.random.default_rng(SEED)

strata = corpus.groupby(["party", "is_retweet"], sort=True)
sizes = strata.size()
alloc = (sizes / sizes.sum() * N).round().astype(int)
# fix rounding to hit N exactly
while alloc.sum() != N:
    alloc.iloc[int(rng.integers(len(alloc)))] += 1 if alloc.sum() < N else -1

parts = []
for key, grp in strata:
    n = alloc[key]
    take = rng.choice(grp.index.to_numpy(), size=n, replace=False)
    parts.append(corpus.loc[take, ["row_idx", "tweet_id", "text"]])
sample = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
sample["half"] = ["H1"] * 1000 + ["H2"] * 1000
sample.to_csv(HERE / "outputs" / "llm_sample.csv", index=False)
print(sample["half"].value_counts().to_dict())
print("sample rows:", len(sample), "| unique tweets:", sample.tweet_id.nunique())
