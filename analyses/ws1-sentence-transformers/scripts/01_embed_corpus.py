"""
01_embed_corpus.py — WS1 E1.1: one embedding per tweet, per tier
----------------------------------------------------------------
Usage: python 01_embed_corpus.py {A|B|C}

Tiers (preregistration.md §2):
  A  minishlab/potion-base-8M          (Model2Vec static, 256-d)
  B  sentence-transformers/all-MiniLM-L6-v2  (384-d)
  C  BAAI/bge-small-en-v1.5            (384-d; ONLY if blind gate passes, see 04)

Reads  : ws0-harness/blind_corpus.parquet  (BLIND — no true_* columns exist here)
Writes : intermediate/emb_tier{X}.npz  (float32, rows = corpus row order,
         with tweet_id and candidate_id arrays for alignment)

Retweets are embedded as their text (retweets-as-speech, per plan).
Intermediate arrays are large (~100–160 MB) and are NOT committed to the
project folder; they are regenerable from this script (deterministic:
both models are frozen pretrained encoders — no training, no seed needed).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
WS0 = HERE.parents[1] / "ws0-harness"

MODELS = {
    "A": ("model2vec", "minishlab/potion-base-8M"),
    "B": ("sentence-transformers", "sentence-transformers/all-MiniLM-L6-v2"),
    "C": ("sentence-transformers", "BAAI/bge-small-en-v1.5"),
}


def main(tier: str) -> None:
    kind, name = MODELS[tier]
    corpus = pd.read_parquet(WS0 / "blind_corpus.parquet")
    texts = corpus["text"].astype(str).tolist()
    print(f"Tier {tier}: {name} on {len(texts):,} tweets", flush=True)

    t0 = time.time()
    if kind == "model2vec":
        from model2vec import StaticModel
        model = StaticModel.from_pretrained(name)
        X = model.encode(texts)
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(name)
        X = model.encode(texts, batch_size=256, show_progress_bar=False,
                         convert_to_numpy=True)
    wall = time.time() - t0
    X = np.asarray(X, dtype=np.float32)
    print(f"Tier {tier}: {X.shape} in {wall/60:.1f} min", flush=True)

    out = HERE / "intermediate" / f"emb_tier{tier}.npz"
    np.savez_compressed(
        out, X=X,
        tweet_id=corpus["tweet_id"].to_numpy(),
        candidate_id=corpus["candidate_id"].to_numpy(),
        model=np.array(name), wall_seconds=np.array(wall),
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1].upper())
