"""
02_embed.py — WS4 preanalysis, step 2.

Trains the four classic embedding models on the corpus itself and builds one
910 x 100 candidate matrix per model, plus the TF-IDF+SVD anchor:

  w2v      word2vec skip-gram, EXACT 07-20/WS0 canonical recipe
           (seed 20260720, workers=1, crc32 hashfxn) -> anchor to frozen baseline
  glove    Stanford GloVe C tool (word+context vectors, -model 2), trained on
           the same tokenized corpus; candidate = token-average
  fasttext gensim FastText skip-gram, same hyperparams as w2v + subword ngrams
  doc2vec  gensim Doc2Vec PV-DBOW, documents = tweets TAGGED BY candidate_id,
           so the model learns one vector per candidate directly
  tfidf    TF-IDF (min_df=5, sublinear) + TruncatedSVD(100), 07-20 recipe

Candidate vector construction for word-vector models is identical to the
07-20 pipeline: frequency-weighted mean of word vectors over all tokens.

Note: fastText and doc2vec run multi-threaded (seeded but not bit-reproducible);
w2v is single-threaded to reproduce the canonical baseline exactly.
"""

import gzip
import json
import subprocess
import zlib

import numpy as np
import pandas as pd
from gensim.models import Word2Vec, FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from pathlib import Path as _Path
import os as _os
_HERE = _Path(__file__).resolve().parent.parent   # this analysis folder
_ROOT = _HERE.parents[1]                          # repository root
OUT = str(_HERE / "outputs")
GLOVE_BIN = _os.environ.get("GLOVE_BIN", str(_ROOT / "third_party" / "glove" / "build"))
DIM = 100
SEED_CANON = 20260720   # word2vec anchor: canonical recipe
SEED_NEW = 20260807     # other models


def crc32_hash(token):
    return zlib.crc32(token.encode("utf-8"))


def load_corpus():
    rows = []
    with gzip.open(f"{OUT}/tokenized_corpus.jsonl.gz", "rt") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def avg_vectors(rows, cand_order, kv):
    """Frequency-weighted token-average per candidate (07-20 construction)."""
    sums = {cid: np.zeros(DIM) for cid in cand_order}
    counts = {cid: 0 for cid in cand_order}
    for r in rows:
        cid = r["candidate_id"]
        vecs = [kv[t] for t in r["tokens"] if t in kv]
        if vecs:
            sums[cid] += np.sum(vecs, axis=0)
            counts[cid] += len(vecs)
    return np.vstack([sums[c] / max(counts[c], 1) for c in cand_order])


def main() -> None:
    meta = pd.read_csv(f"{OUT}/candidate_table.csv")
    cand_order = meta["candidate_id"].tolist()
    rows = load_corpus()
    sentences = [r["tokens"] for r in rows]
    mats = {}

    # ---------- word2vec (canonical anchor recipe) ----------
    print("Training word2vec (canonical recipe)...")
    w2v = Word2Vec(sentences=sentences, vector_size=DIM, window=5, min_count=5,
                   sg=1, workers=1, epochs=10, seed=SEED_CANON,
                   hashfxn=crc32_hash)
    print(f"  vocab: {len(w2v.wv)}")
    mats["w2v"] = avg_vectors(rows, cand_order, w2v.wv)

    # ---------- GloVe (Stanford C tool) ----------
    print("Training GloVe...")
    corpus = f"{OUT}/corpus_glove.txt"
    subprocess.run(
        f"{GLOVE_BIN}/vocab_count -min-count 5 -verbose 0 < {corpus} > {OUT}/glove_vocab.txt",
        shell=True, check=True)
    subprocess.run(
        f"{GLOVE_BIN}/cooccur -memory 4.0 -vocab-file {OUT}/glove_vocab.txt "
        f"-verbose 0 -window-size 5 < {corpus} > {OUT}/glove_cooccur.bin",
        shell=True, check=True)
    subprocess.run(
        f"{GLOVE_BIN}/shuffle -memory 4.0 -verbose 0 -seed {SEED_NEW} "
        f"< {OUT}/glove_cooccur.bin > {OUT}/glove_cooccur.shuf.bin",
        shell=True, check=True)
    subprocess.run(
        f"{GLOVE_BIN}/glove -save-file {OUT}/glove_vectors -threads 4 "
        f"-input-file {OUT}/glove_cooccur.shuf.bin -x-max 10 -iter 25 "
        f"-vector-size {DIM} -binary 0 -vocab-file {OUT}/glove_vocab.txt "
        f"-verbose 0 -model 2 -seed {SEED_NEW}",
        shell=True, check=True)
    glove_kv = {}
    with open(f"{OUT}/glove_vectors.txt") as fh:
        for line in fh:
            parts = line.rstrip().split(" ")
            if len(parts) == DIM + 1:
                glove_kv[parts[0]] = np.asarray(parts[1:], dtype=float)
    print(f"  vocab: {len(glove_kv)}")
    mats["glove"] = avg_vectors(rows, cand_order, glove_kv)

    # ---------- fastText ----------
    print("Training fastText...")
    ft = FastText(sentences=sentences, vector_size=DIM, window=5, min_count=5,
                  sg=1, workers=4, epochs=10, seed=SEED_NEW)
    print(f"  vocab: {len(ft.wv)}")
    # restrict to in-vocab tokens (same coverage rule as other models)
    ft_kv = {w: ft.wv[w] for w in ft.wv.key_to_index}
    mats["fasttext"] = avg_vectors(rows, cand_order, ft_kv)

    # ---------- doc2vec (PV-DBOW, tags = candidate_id) ----------
    print("Training doc2vec...")
    tagged = [TaggedDocument(words=r["tokens"], tags=[r["candidate_id"]])
              for r in rows]
    d2v = Doc2Vec(documents=tagged, vector_size=DIM, window=5, min_count=5,
                  dm=0, workers=4, epochs=20, seed=SEED_NEW)
    mats["doc2vec"] = np.vstack([d2v.dv[c] for c in cand_order])

    # ---------- TF-IDF + SVD anchor ----------
    print("Building TF-IDF + SVD anchor...")
    docs = {cid: [] for cid in cand_order}
    for r in rows:
        docs[r["candidate_id"]].extend(r["tokens"])
    doc_strings = [" ".join(docs[c]) for c in cand_order]
    tfidf = TfidfVectorizer(min_df=5, sublinear_tf=True)
    X_sparse = tfidf.fit_transform(doc_strings)
    svd = TruncatedSVD(n_components=DIM, random_state=SEED_CANON)
    mats["tfidf"] = svd.fit_transform(X_sparse)

    np.savez_compressed(
        f"{OUT}/candidate_vectors_all.npz",
        candidate_ids=np.array(cand_order),
        **{f"X_{k}": v for k, v in mats.items()},
    )
    for k, v in mats.items():
        print(f"  {k}: {v.shape}")
    print("Saved candidate_vectors_all.npz")


if __name__ == "__main__":
    main()
