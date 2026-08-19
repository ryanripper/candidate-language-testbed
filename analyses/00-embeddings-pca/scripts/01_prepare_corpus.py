"""
01_prepare_corpus.py
--------------------
Step 1 of the candidate-language pipeline.

Loads the synthetic 2022-cycle candidate tweet corpus, applies the "blind"
protocol (planted ground-truth columns are set aside, not used), cleans and
tokenizes tweet text, and writes a tokenized corpus + candidate metadata
table for the downstream embedding steps.

Blind protocol: true_topic / true_framing / true_ideology are dropped from
the working data and saved separately. They are only re-joined in
05_validate.py, after the blind analysis is complete.

Retweet policy: retweets are INCLUDED as candidate speech (a retweet is
treated as language the candidate chose to amplify), consistent with the
original research design.
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
TOKEN_RE = re.compile(r"[a-z][a-z']+")  # keep simple word tokens, incl. apostrophes

def tokenize(text: str) -> list[str]:
    """Lowercase, strip URLs and @mentions, keep hashtag words, simple word tokens."""
    t = text.lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = t.replace("#", " ")  # keep hashtag content as plain words
    return TOKEN_RE.findall(t)

def main() -> None:
    df = pd.read_csv(DATA, compression="gzip")
    print(f"Loaded {len(df):,} rows, {df['candidate_id'].nunique()} candidates")

    # ---- Blind protocol: sequester ground truth ----
    truth = (
        df.groupby("candidate_id")
        .agg(true_ideology=("true_ideology", "first"))
        .reset_index()
    )
    truth.to_csv(f"{OUT}/ground_truth_SEALED.csv", index=False)
    df = df.drop(columns=["true_topic", "true_framing", "true_ideology"])

    # ---- Candidate metadata (observable fields only) ----
    meta = (
        df.groupby("candidate_id")
        .agg(
            candidate_name=("candidate_name", "first"),
            handle=("handle", "first"),
            party=("party", "first"),
            chamber=("chamber", "first"),
            state=("state", "first"),
            district=("district", "first"),
            incumbent=("incumbent", "first"),
            n_tweets=("tweet_id", "count"),
            share_retweets=("is_retweet", "mean"),
        )
        .reset_index()
    )
    meta.to_csv(f"{OUT}/candidate_metadata.csv", index=False)

    # ---- Tokenize (retweets included as candidate speech) ----
    df["tokens"] = df["text"].astype(str).map(tokenize)

    # Corpus stats
    n_tokens = int(df["tokens"].map(len).sum())
    vocab = set()
    for toks in df["tokens"]:
        vocab.update(toks)
    stats = {
        "n_tweets": int(len(df)),
        "n_candidates": int(meta.shape[0]),
        "n_tokens": n_tokens,
        "vocab_size_raw": len(vocab),
        "share_retweets": float(df["is_retweet"].mean()),
        "party_counts": meta["party"].value_counts().to_dict(),
        "chamber_counts": meta["chamber"].value_counts().to_dict(),
        "median_tweets_per_candidate": float(meta["n_tweets"].median()),
    }
    with open(f"{OUT}/corpus_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats, indent=2))

    # ---- Write tokenized corpus (one JSON line per tweet) ----
    with gzip.open(f"{OUT}/tokenized_corpus.jsonl.gz", "wt") as fh:
        for cid, toks in zip(df["candidate_id"], df["tokens"]):
            fh.write(json.dumps({"candidate_id": cid, "tokens": toks}) + "\n")
    print("Wrote tokenized corpus.")

if __name__ == "__main__":
    main()
