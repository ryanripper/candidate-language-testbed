"""
05_bertopic_entrant.py — WS2 Stage A entrant 4 (BERTopic), BLIND
----------------------------------------------------------------
Pre-registered config (preregistration §2):
  Tier B MiniLM embeddings -> UMAP(n_neighbors=15, n_components=5,
  metric=cosine, min_dist=0.0, random_state=20260726) ->
  HDBSCAN(min_cluster_size=200, euclidean, eom) -> c-TF-IDF.
Outlier ladder: report rate; >50% -> min_cluster_size=60; still >50% ->
k-means mode with entrant-3's chosen K. Remaining outliers assigned to
nearest topic embedding centroid (cosine) so all tweets are labeled.

Writes : outputs/assignments_bertopic.npy, outputs/topics_bertopic.json,
         outputs/entrant_meta_bertopic.json
"""
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
ST = HERE.parent / "ws1-sentence-transformers"
SEED = 20260726

emb = np.load(ST / "intermediate" / "emb_tierB.npz", allow_pickle=True)
X = emb["X"].astype(np.float32)

t0 = time.time()
import umap
import hdbscan

reducer = umap.UMAP(n_neighbors=15, n_components=5, metric="cosine",
                    min_dist=0.0, random_state=SEED, verbose=False)
Z = reducer.fit_transform(X)
print(f"UMAP done {time.time()-t0:.0f}s", flush=True)

ladder = []
mode = None
for mcs in (200, 60):
    cl = hdbscan.HDBSCAN(min_cluster_size=mcs, metric="euclidean",
                         cluster_selection_method="eom", core_dist_n_jobs=4)
    raw = cl.fit_predict(Z)
    out_rate = float((raw == -1).mean())
    n_topics = int(len(set(raw)) - (1 if -1 in raw else 0))
    ladder.append({"min_cluster_size": mcs, "outlier_rate": out_rate,
                   "n_topics": n_topics})
    print(f"HDBSCAN mcs={mcs}: {n_topics} topics, outliers {out_rate:.3f}",
          flush=True)
    if out_rate <= 0.5:
        mode = f"hdbscan_mcs{mcs}"
        break

if mode is None:
    from sklearn.cluster import KMeans
    k3 = json.load(open(HERE / "outputs" / "entrant_meta_classical.json"))
    K = k3["lsa"]["chosen_K"]
    raw = KMeans(n_clusters=K, n_init=5, random_state=SEED).fit_predict(Z)
    mode = f"kmeans_K{K}"
    ladder.append({"kmeans_K": K, "outlier_rate": 0.0,
                   "n_topics": int(K)})

labels = raw.copy()
if (labels == -1).any():
    ids = sorted(set(labels) - {-1})
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cent = np.stack([Xn[labels == k].mean(0) for k in ids])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
    mask = labels == -1
    labels[mask] = np.array(ids)[np.argmax(Xn[mask] @ cent.T, axis=1)]

# compact labels to 0..K-1
ids = sorted(set(labels))
remap = {k: i for i, k in enumerate(ids)}
labels = np.array([remap[k] for k in labels], dtype=np.int32)

np.save(HERE / "outputs" / "assignments_bertopic.npy", labels)
meta = {"mode": mode, "ladder": ladder, "n_topics_final": int(len(ids)),
        "wall_seconds": round(time.time() - t0, 1)}
json.dump(meta, open(HERE / "outputs" / "entrant_meta_bertopic.json", "w"),
          indent=2)
print(json.dumps(meta, indent=2))
