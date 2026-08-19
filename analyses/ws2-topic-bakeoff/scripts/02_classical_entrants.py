"""
02_classical_entrants.py — WS2 Stage A entrants 1–3 (LDA / NMF / LSA), BLIND
----------------------------------------------------------------------------
Pre-registered configs (preregistration §2). Each entrant sweeps
K ∈ {5,8,10,12,15,20,25,30,40}, picks K maximizing mean NPMI coherence on
the frozen 10k eval subsample (ties → smaller K), then assigns all 104,601
tweets.

Writes : outputs/assignments_{lda,nmf,lsa}.npy   int labels, corpus order
         outputs/topics_{lda,nmf,lsa}.json       top-25 terms per topic
         outputs/ksweep_{lda,nmf,lsa}.csv        K vs coherence
         outputs/entrant_meta_classical.json     chosen K, wall-clock
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parents[1] / "ws0-harness"))
from metrics import npmi_coherence  # noqa: E402

SEED = 20260726
KGRID = [5, 8, 10, 12, 15, 20, 25, 30, 40]

inter = HERE / "intermediate"
out = HERE / "outputs"
tokens = pickle.load(open(inter / "tokens.pkl", "rb"))
X = sparse.load_npz(inter / "tfidf.npz")
vocab = np.array(json.load(open(inter / "vocab.json")))
eval_idx = np.load(inter / "eval_idx.npy")
eval_docs = [tokens[i] for i in eval_idx]

meta = {}


def pick_k(rows: list[dict]) -> int:
    best = max(rows, key=lambda r: (round(r["npmi_mean"], 6), -r["K"]))
    return best["K"]


def save(name: str, labels: np.ndarray, topics: list[list[str]],
         sweep: list[dict], k: int, wall: float) -> None:
    np.save(out / f"assignments_{name}.npy", labels.astype(np.int32))
    json.dump(topics, open(out / f"topics_{name}.json", "w"))
    import csv
    with open(out / f"ksweep_{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["K", "npmi_mean"])
        w.writeheader()
        w.writerows(sweep)
    meta[name] = {"chosen_K": k, "n_topics_used": int(len(set(labels))),
                  "wall_seconds": round(wall, 1)}
    print(f"{name}: K={k}, wall={wall:.0f}s", flush=True)


# ------------------------------------------------------------------- LDA
def run_lda() -> None:
    from gensim.corpora import Dictionary
    from gensim.models import LdaMulticore

    t0 = time.time()
    dictionary = Dictionary(tokens)
    dictionary.filter_extremes(no_below=5, no_above=1.0)
    bow = [dictionary.doc2bow(t) for t in tokens]
    sweep, models = [], {}
    for K in KGRID:
        lda = LdaMulticore(bow, num_topics=K, id2word=dictionary,
                           passes=2, chunksize=10_000, workers=3,
                           random_state=SEED)
        tops = [[w for w, _ in lda.show_topic(k, topn=25)] for k in range(K)]
        c = npmi_coherence([t[:10] for t in tops], eval_docs)["npmi_mean"]
        sweep.append({"K": K, "npmi_mean": c})
        models[K] = (lda, tops)
        print(f"  LDA K={K}: npmi={c:.4f}", flush=True)
    k = pick_k(sweep)
    lda, tops = models[k]
    # argmax assignment
    labels = np.empty(len(bow), dtype=np.int32)
    for i, doc in enumerate(bow):
        td = lda.get_document_topics(doc, minimum_probability=0.0)
        labels[i] = max(td, key=lambda x: x[1])[0]
    save("lda", labels, tops, sweep, k, time.time() - t0)


# ------------------------------------------------------------------- NMF
def run_nmf() -> None:
    from sklearn.decomposition import NMF

    t0 = time.time()
    sweep, models = [], {}
    for K in KGRID:
        nmf = NMF(n_components=K, init="nndsvda", random_state=SEED,
                  max_iter=400)
        W = nmf.fit_transform(X)
        tops = [vocab[np.argsort(h)[::-1][:25]].tolist()
                for h in nmf.components_]
        c = npmi_coherence([t[:10] for t in tops], eval_docs)["npmi_mean"]
        sweep.append({"K": K, "npmi_mean": c})
        models[K] = (W.argmax(1), tops)
        print(f"  NMF K={K}: npmi={c:.4f}", flush=True)
    k = pick_k(sweep)
    labels, tops = models[k]
    save("nmf", labels, tops, sweep, k, time.time() - t0)


# ------------------------------------------------------------------- LSA
def run_lsa() -> None:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    t0 = time.time()
    svd = TruncatedSVD(100, random_state=SEED)
    Z = normalize(svd.fit_transform(X))
    sweep, models = [], {}
    for K in KGRID:
        km = KMeans(n_clusters=K, n_init=5, random_state=SEED).fit(Z)
        labels = km.labels_
        # top terms per cluster via mean TF-IDF (c-TF-IDF style)
        tops = []
        for k in range(K):
            m = np.asarray(X[labels == k].mean(0)).ravel()
            tops.append(vocab[np.argsort(m)[::-1][:25]].tolist())
        c = npmi_coherence([t[:10] for t in tops], eval_docs)["npmi_mean"]
        sweep.append({"K": K, "npmi_mean": c})
        models[K] = (labels, tops)
        print(f"  LSA K={K}: npmi={c:.4f}", flush=True)
    k = pick_k(sweep)
    labels, tops = models[k]
    save("lsa", labels, tops, sweep, k, time.time() - t0)


if __name__ == "__main__":
    run_nmf()
    run_lsa()
    run_lda()
    json.dump(meta, open(out / "entrant_meta_classical.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))
