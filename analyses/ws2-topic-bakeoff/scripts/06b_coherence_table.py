"""
06b_coherence_table.py — uniform blind coherence/diversity table, BLIND
-----------------------------------------------------------------------
One row per entrant, computed identically for all five from the uniform
c-TF-IDF top terms (entrant_topic_terms.json): NPMI (metrics.npmi_coherence,
top-10, 10k eval subsample), gensim c_v (top-10), topic diversity (top-25).

Writes : outputs/coherence_diversity.csv
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parents[1] / "ws0-harness"))
from metrics import npmi_coherence, cv_coherence, topic_diversity  # noqa: E402

inter = HERE / "intermediate"
out = HERE / "outputs"
tokens = pickle.load(open(inter / "tokens.pkl", "rb"))
eval_idx = np.load(inter / "eval_idx.npy")
eval_docs = [tokens[i] for i in eval_idx]
terms = json.load(open(out / "entrant_topic_terms.json"))

rows = []
for ent, tt in terms.items():
    tops = [tt[k] for k in sorted(tt, key=int)]
    npmi = npmi_coherence([t[:10] for t in tops], eval_docs)["npmi_mean"]
    cv = cv_coherence([t[:10] for t in tops], eval_docs)
    div = topic_diversity(tops, topn=25)
    rows.append({"entrant": ent, "n_topics": len(tops),
                 "npmi_mean": round(npmi, 4), "c_v": round(cv, 4),
                 "diversity": round(div, 4)})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(out / "coherence_diversity.csv", index=False)
