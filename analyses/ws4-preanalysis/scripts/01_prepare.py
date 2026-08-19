"""
01_prepare.py — WS4 preanalysis, step 1.

Standalone (NOT blind) preparation of the synthetic 2022-cycle corpus for a
static-embedding bake-off: word2vec / GloVe / fastText / doc2vec (+ TF-IDF+SVD
anchor). Ground truth stays visible throughout — this is an informal
preanalysis for the WS4 supervised extension, not a certified WS0 run.

Tokenization is IDENTICAL to 00-embeddings-pca/scripts/01_prepare_corpus.py
(2026-07-20) so candidate vectors are constructed the same way as the frozen
baselines. Retweets included as candidate speech.
"""

import gzip
import json
import re

import pandas as pd

from pathlib import Path as _Path
import os as _os
_HERE = _Path(__file__).resolve().parent.parent   # this analysis folder
_ROOT = _HERE.parents[1]                          # repository root
DATA = str(_ROOT / "data" / "synthetic-candidate-tweets" / "synthetic_candidate_tweets_2022.csv.gz")
OUT = str(_HERE / "outputs")

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
TOKEN_RE = re.compile(r"[a-z][a-z']+")


def tokenize(text: str) -> list[str]:
    t = text.lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = t.replace("#", " ")
    return TOKEN_RE.findall(t)


def main() -> None:
    df = pd.read_csv(DATA, compression="gzip")
    print(f"Loaded {len(df):,} rows, {df['candidate_id'].nunique()} candidates")

    meta = (
        df.groupby("candidate_id")
        .agg(
            candidate_name=("candidate_name", "first"),
            party=("party", "first"),
            chamber=("chamber", "first"),
            state=("state", "first"),
            incumbent=("incumbent", "first"),
            n_tweets=("tweet_id", "count"),
            share_retweets=("is_retweet", "mean"),
            true_ideology=("true_ideology", "first"),  # visible: standalone run
        )
        .reset_index()
        .sort_values("candidate_id")
        .reset_index(drop=True)
    )
    meta.to_csv(f"{OUT}/candidate_table.csv", index=False)

    df["tokens"] = df["text"].astype(str).map(tokenize)

    n_tokens = int(df["tokens"].map(len).sum())
    print(f"Tokens: {n_tokens:,}")

    with gzip.open(f"{OUT}/tokenized_corpus.jsonl.gz", "wt") as fh:
        for cid, toks in zip(df["candidate_id"], df["tokens"]):
            fh.write(json.dumps({"candidate_id": cid, "tokens": toks}) + "\n")

    # Plain-text corpus for the Stanford GloVe tool (one tweet per line).
    with open(f"{OUT}/corpus_glove.txt", "w") as fh:
        for toks in df["tokens"]:
            fh.write(" ".join(toks) + "\n")

    print("Wrote candidate_table.csv, tokenized_corpus.jsonl.gz, corpus_glove.txt")


if __name__ == "__main__":
    main()
