"""
06_judge_prep.py — WS2 blinded interpretability judging: packet builder, BLIND
------------------------------------------------------------------------------
Uniform rendering for every entrant (preregistration §6): top-10 c-TF-IDF
terms + 3 medoid example tweets per topic, pooled across entrants, shuffled
with seed 20260726, method identity hidden behind opaque item ids.

c-TF-IDF: per-topic summed token counts, weighted by log(1 + K / df_topic)
where df_topic = number of topics whose aggregate contains the token.
Medoids: 3 tweets with highest cosine to the topic's mean TF-IDF vector
(computed over a ≤3,000-member random subset per topic, seed 20260726).

Writes : outputs/judge_packet.json  (blinded, shuffled)
         outputs/judge_key.json     (item id -> entrant/topic; NOT shown to judge)
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent.parent
WS0 = HERE.parents[1] / "ws0-harness"
SEED = 20260726

inter = HERE / "intermediate"
out = HERE / "outputs"
X = sparse.load_npz(inter / "tfidf.npz").tocsr()
vocab = np.array(json.load(open(inter / "vocab.json")))
tokens = pickle.load(open(inter / "tokens.pkl", "rb"))
corpus = pd.read_parquet(WS0 / "blind_corpus.parquet")
texts = corpus["text"].astype(str).tolist()

ENTRANTS = ["lda", "nmf", "lsa", "bertopic", "llm"]
rng = np.random.default_rng(SEED)
MAX_JUDGED = 40  # documented deviation from prereg §6: with 763 BERTopic
# topics the full pool (875) is unjudgeable in-session; each entrant's
# judged set is a uniform random sample of <=40 of its topics (this seed).

# token counts per doc (reuse tfidf vocab index)
vindex = {w: i for i, w in enumerate(vocab)}

items = []
key = {}
all_terms = {}
for ent in ENTRANTS:
    labels = np.load(out / f"assignments_{ent}.npy")
    ks = sorted(set(labels.tolist()))
    K = len(ks)
    # aggregate token counts per topic
    counts = np.zeros((K, len(vocab)), dtype=np.float64)
    for ki, k in enumerate(ks):
        idx = np.where(labels == k)[0]
        for i in idx:
            for w in tokens[i]:
                j = vindex.get(w)
                if j is not None:
                    counts[ki, j] += 1
    tf = counts / (counts.sum(1, keepdims=True) + 1e-12)
    df_topic = (counts > 0).sum(0)
    idf = np.log(1 + K / (df_topic + 1e-12))
    ctfidf = tf * idf
    judged = set(rng.choice(len(ks), size=min(MAX_JUDGED, len(ks)),
                            replace=False).tolist())
    all_terms[ent] = {}
    for ki, k in enumerate(ks):
        top25 = vocab[np.argsort(ctfidf[ki])[::-1][:25]].tolist()
        all_terms[ent][int(k)] = top25
        if ki not in judged:
            continue
        top = top25[:10]
        idx = np.where(labels == k)[0]
        sub = rng.choice(idx, size=min(3000, len(idx)), replace=False)
        M = X[sub]
        mean = np.asarray(M.mean(0))
        sims = (M @ mean.T).ravel() / (
            np.sqrt(M.multiply(M).sum(1)).A.ravel() * np.linalg.norm(mean) + 1e-12)
        med = sub[np.argsort(sims)[::-1][:3]]
        items.append({"terms": top,
                      "examples": [texts[i][:220] for i in med],
                      "n_tweets": int(len(idx))})
        key[len(items) - 1] = {"entrant": ent, "topic": int(k)}

order = rng.permutation(len(items))
packet = [{"item_id": f"T{n:03d}", **items[i]} for n, i in enumerate(order)]
json.dump(packet, open(out / "judge_packet.json", "w"), indent=1)
json.dump({f"T{n:03d}": key[int(i)] for n, i in enumerate(order)},
          open(out / "judge_key.json", "w"), indent=1)
json.dump(all_terms, open(out / "entrant_topic_terms.json", "w"), indent=1)
print(f"packet: {len(packet)} topics across {len(ENTRANTS)} entrants")
for ent in ENTRANTS:
    n = sum(1 for v in key.values() if v["entrant"] == ent)
    print(f"  {ent}: {n} topics")
