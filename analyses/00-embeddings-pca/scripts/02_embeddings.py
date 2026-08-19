"""
02_embeddings.py
----------------
Step 2 of the candidate-language pipeline.

Main method  : word2vec (skip-gram) trained ON the corpus itself, then each
               candidate is represented as the average of the word vectors of
               every token they tweeted (retweets included).
Baseline     : TF-IDF document-per-candidate + TruncatedSVD to the same
               dimensionality, as a classic sparse-vector comparison.

Outputs: candidate x dim matrices for both methods (.npz) with row order
matching candidate_metadata.csv.
"""

import gzip
import json

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from pathlib import Path as _Path
import os as _os
_HERE = _Path(__file__).resolve().parent.parent   # this analysis folder
_ROOT = _HERE.parents[1]                          # repository root
OUT = str(_HERE / "outputs")
DIM = 100
SEED = 20260720

def load_corpus():
    rows = []
    with gzip.open(f"{OUT}/tokenized_corpus.jsonl.gz", "rt") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows

def main() -> None:
    meta = pd.read_csv(f"{OUT}/candidate_metadata.csv")
    cand_order = meta["candidate_id"].tolist()
    rows = load_corpus()
    sentences = [r["tokens"] for r in rows]

    # ---------- Word2Vec trained on the corpus ----------
    print("Training word2vec (skip-gram)...")
    w2v = Word2Vec(
        sentences=sentences,
        vector_size=DIM,
        window=5,
        min_count=5,
        sg=1,               # skip-gram
        workers=4,
        epochs=10,
        seed=SEED,
    )
    w2v.save(f"{OUT}/word2vec.model")
    print(f"  vocab retained: {len(w2v.wv)} words")

    # Candidate vector = mean of word vectors over all their tokens
    # (token-frequency weighted by construction: every occurrence counts).
    sums = {cid: np.zeros(DIM) for cid in cand_order}
    counts = {cid: 0 for cid in cand_order}
    for r in rows:
        cid = r["candidate_id"]
        vecs = [w2v.wv[t] for t in r["tokens"] if t in w2v.wv]
        if vecs:
            sums[cid] += np.sum(vecs, axis=0)
            counts[cid] += len(vecs)
    X_w2v = np.vstack([sums[c] / max(counts[c], 1) for c in cand_order])

    # ---------- TF-IDF + SVD baseline ----------
    print("Building TF-IDF + SVD baseline...")
    docs = {cid: [] for cid in cand_order}
    for r in rows:
        docs[r["candidate_id"]].extend(r["tokens"])
    doc_strings = [" ".join(docs[c]) for c in cand_order]

    tfidf = TfidfVectorizer(min_df=5, sublinear_tf=True)
    X_sparse = tfidf.fit_transform(doc_strings)
    svd = TruncatedSVD(n_components=DIM, random_state=SEED)
    X_tfidf = svd.fit_transform(X_sparse)
    print(f"  TF-IDF vocab: {len(tfidf.vocabulary_)}; "
          f"SVD explained variance: {svd.explained_variance_ratio_.sum():.3f}")

    np.savez_compressed(
        f"{OUT}/candidate_vectors.npz",
        candidate_ids=np.array(cand_order),
        X_w2v=X_w2v,
        X_tfidf=X_tfidf,
    )
    print("Saved candidate vectors:", X_w2v.shape, X_tfidf.shape)

    # Sanity check: nearest neighbours of a few seed words in w2v space
    for seed_word in ["border", "climate", "abortion", "taxes"]:
        if seed_word in w2v.wv:
            nns = ", ".join(w for w, _ in w2v.wv.most_similar(seed_word, topn=5))
            print(f"  w2v neighbours of '{seed_word}': {nns}")

if __name__ == "__main__":
    main()
