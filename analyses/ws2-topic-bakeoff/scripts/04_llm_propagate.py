"""
04_llm_propagate.py — WS2 entrant 5, stage (ii): corpus-wide propagation, BLIND
-------------------------------------------------------------------------------
Nearest label-centroid (cosine) in Tier B MiniLM space (preregistration §2).
Centroids = mean embedding of the 2,000 in-session-labeled sample tweets per
theme. Sample tweets keep their LLM label; all other tweets get the nearest
centroid's label.

Reads  : ws1-sentence-transformers/intermediate/emb_tierB.npz
         outputs/llm_sample_labels.csv, outputs/llm_taxonomy.json
Writes : outputs/assignments_llm.npy, outputs/entrant_meta_llm.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
ST = HERE.parent / "ws1-sentence-transformers"

emb = np.load(ST / "intermediate" / "emb_tierB.npz", allow_pickle=True)
X = emb["X"].astype(np.float32)
X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12

lab = pd.read_csv(HERE / "outputs" / "llm_sample_labels.csv")
tax = json.load(open(HERE / "outputs" / "llm_taxonomy.json"))
n_themes = len(tax["themes"])

cent = np.zeros((n_themes, X.shape[1]), dtype=np.float32)
used = []
for t in range(n_themes):
    idx = lab.loc[lab.theme_id == t, "row_idx"].to_numpy()
    if len(idx) == 0:
        continue
    c = X[idx].mean(0)
    cent[t] = c / (np.linalg.norm(c) + 1e-12)
    used.append(t)
used = np.array(used)

sims = X @ cent[used].T          # (104601, n_used)
labels = used[np.argmax(sims, axis=1)].astype(np.int32)
labels[lab.row_idx.to_numpy()] = lab.theme_id.to_numpy()  # sample keeps LLM label

np.save(HERE / "outputs" / "assignments_llm.npy", labels)
meta = {
    "themes_total": n_themes,
    "themes_with_sample_support": int(len(used)),
    "sample_size": int(len(lab)),
    "corpus_label_counts": {int(k): int(v) for k, v in
                            zip(*np.unique(labels, return_counts=True))},
}
json.dump(meta, open(HERE / "outputs" / "entrant_meta_llm.json", "w"), indent=2)
print(json.dumps({k: v for k, v in meta.items() if k != "corpus_label_counts"},
                 indent=2))
print("corpus label shares (top 10):")
u, c = np.unique(labels, return_counts=True)
order = np.argsort(c)[::-1][:10]
names = {t["id"]: t["name"] for t in tax["themes"]}
for i in order:
    print(f"  {names[int(u[i])]:20s} {c[i]:7d}  {c[i]/len(labels):.3f}")
