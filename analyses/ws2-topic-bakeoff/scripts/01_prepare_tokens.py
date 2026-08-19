"""
01_prepare_tokens.py — WS2 shared lexical preprocessing (preregistration §2)
---------------------------------------------------------------------------
Reads  : ws0-harness/blind_corpus.parquet  (BLIND)
Writes : intermediate/tokens.pkl        list[list[str]], corpus row order
         intermediate/tfidf.npz         sparse TF-IDF (sublinear, min_df=5)
         intermediate/vocab.json        TF-IDF feature names
         intermediate/eval_idx.npy      10,000-row coherence eval subsample
         outputs/preprocessing_stats.json

Rules (pre-registered): strip "RT @Handle:" prefix; lowercase; tokens
[a-z]{3,} after removing apostrophes/digits; ENGLISH_STOP_WORDS ∪
{"rt","amp","icymi"}; min_df=5. Seed 20260726.
"""
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

HERE = Path(__file__).resolve().parent.parent
WS0 = HERE.parents[1] / "ws0-harness"
SEED = 20260726

RT_PREFIX = re.compile(r"^RT @\w+:\s*")
APOS_DIGIT = re.compile(r"[\'’\d]+")
TOKEN = re.compile(r"[a-z]{3,}")
STOPS = set(ENGLISH_STOP_WORDS) | {"rt", "amp", "icymi"}


def tokenize(text: str) -> list[str]:
    t = RT_PREFIX.sub("", text).lower()
    t = APOS_DIGIT.sub("", t)
    return [w for w in TOKEN.findall(t) if w not in STOPS]


def main() -> None:
    corpus = pd.read_parquet(WS0 / "blind_corpus.parquet")
    texts = corpus["text"].astype(str).tolist()
    tokens = [tokenize(t) for t in texts]

    inter = HERE / "intermediate"
    inter.mkdir(exist_ok=True)
    with open(inter / "tokens.pkl", "wb") as f:
        pickle.dump(tokens, f)

    # doc-frequency floor shared with TF-IDF: build vectorizer on the same tokens
    vec = TfidfVectorizer(analyzer=lambda toks: toks, min_df=5,
                          sublinear_tf=True, norm="l2")
    X = vec.fit_transform(tokens)
    sparse.save_npz(inter / "tfidf.npz", X)
    vocab = vec.get_feature_names_out().tolist()
    with open(inter / "vocab.json", "w") as f:
        json.dump(vocab, f)

    rng = np.random.default_rng(SEED)
    eval_idx = np.sort(rng.choice(len(texts), size=10_000, replace=False))
    np.save(inter / "eval_idx.npy", eval_idx)

    lens = np.array([len(t) for t in tokens])
    stats = {
        "n_docs": len(texts),
        "vocab_size": len(vocab),
        "mean_tokens_per_doc": float(lens.mean()),
        "empty_docs": int((lens == 0).sum()),
        "eval_subsample": 10_000,
        "seed": SEED,
    }
    (HERE / "outputs").mkdir(exist_ok=True)
    with open(HERE / "outputs" / "preprocessing_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
