"""
07_behavior_prediction.py — Stage C (preregistration §7, post-unseal,
instrument scores frozen blind beforehand). Reads outputs/unsealed_pack.parquet
(written by 06) — sealed_truth is NOT re-read.

C1  retweet-source choice: softmax over 20 orgs, (a,b,beta) MLE-fit on
    split-A retweets, evaluated on split-B (mean log-loss primary; top-1 acc).
C2  topic attention: per-topic linear share~score fit on split-A shares,
    predicted split-B shares, mean Jensen-Shannon divergence.
C3  framing intensity: r(score, split-B framing), overall + topics 0/1/2
    (>=5 split-B tweets in topic).

Outputs: outputs/c1_retweet_choice.csv, c2_topic_attention.csv,
         c2_topic_r.csv, c3_framing.csv
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.spatial.distance import jensenshannon

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ws3-llm-scaling" / "outputs"

pack = pd.read_parquet(OUT / "unsealed_pack.parquet")
cands = pack.candidate_id.tolist()
orgs = json.load(open(OUT / "org_ideologies.json"))
org_handles = sorted(orgs)
iota = np.array([orgs[h] for h in org_handles])
J = len(org_handles)

INSTR = {"LLM ask-and-average": "llm_score", "TF-IDF+SVD PC1": "tfidf_score",
         "WS1 Tier A Model2Vec": "ws1_score",
         "Oracle (true_ideology)": "true_ideology"}

bc = pd.read_parquet(ROOT.parent / "ws0-harness" / "blind_corpus.parquet")
ab = pd.read_parquet(ROOT.parent / "ws0-harness" / "tweet_split_ab.parquet")
df = bc.merge(ab[["tweet_id", "split"]], on="tweet_id")
df["topic"] = np.load(ROOT / "ws2-topic-bakeoff" / "outputs" /
                      "assignments_llm_refined.npy")
sub = df[df.candidate_id.isin(cands)]
rt = sub[sub.is_retweet].copy()
rt["org_idx"] = rt.retweeted_handle.map({h: i for i, h in enumerate(org_handles)})
rtA, rtB = rt[rt.split == "A"], rt[rt.split == "B"]

# ------------------------------------------------------------------ C1
def nll(params, s_cand, org_choice):
    a, b, logbeta = params
    beta = np.exp(logbeta)
    z = -beta * np.abs((a * s_cand[:, None] + b) - iota[None, :])
    z -= z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return -logp[np.arange(len(org_choice)), org_choice].mean()

cid_to_row = {c: i for i, c in enumerate(cands)}
rows = []
for name, col in INSTR.items():
    s = pack[col].values
    sA = s[[cid_to_row[c] for c in rtA.candidate_id]]
    sB = s[[cid_to_row[c] for c in rtB.candidate_id]]
    scale = 1.0 / max(np.std(s), 1e-9)
    best = None
    for a0 in [scale, 2 * scale, 0.5 * scale]:
        for lb0 in [np.log(2), np.log(5), np.log(10)]:
            res = minimize(nll, x0=[a0, 0.0, lb0], method="Nelder-Mead",
                           args=(sA, rtA.org_idx.values),
                           options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-9})
            if best is None or res.fun < best.fun:
                best = res
    a, b, logbeta = best.x
    beta = np.exp(logbeta)
    z = -beta * np.abs((a * sB[:, None] + b) - iota[None, :])
    z -= z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    ll = -logp[np.arange(len(rtB)), rtB.org_idx.values].mean()
    top1 = float((logp.argmax(axis=1) == rtB.org_idx.values).mean())
    rows.append({"instrument": name, "heldout_logloss": float(ll),
                 "top1_acc": top1, "train_logloss": float(best.fun),
                 "a": a, "b": b, "beta": beta, "n_test_retweets": len(rtB)})
# nulls
pA = np.bincount(rtA.org_idx.values, minlength=J) / len(rtA)
pA = np.clip(pA, 1e-12, None); pA /= pA.sum()
rows.append({"instrument": "NULL uniform", "heldout_logloss": float(np.log(J)),
             "top1_acc": 1.0 / J, "n_test_retweets": len(rtB)})
rows.append({"instrument": "NULL split-A org base rates",
             "heldout_logloss": float(-np.log(pA[rtB.org_idx.values]).mean()),
             "top1_acc": float((rtB.org_idx.values == pA.argmax()).mean()),
             "n_test_retweets": len(rtB)})
c1 = pd.DataFrame(rows)
c1.to_csv(OUT / "c1_retweet_choice.csv", index=False)
excluded = 150 - rtB.candidate_id.nunique()

# ------------------------------------------------------------------ C2
topics = sorted(sub.topic.unique())
def shares(frame):
    ct = frame.groupby(["candidate_id", "topic"]).size().unstack(fill_value=0)
    ct = ct.reindex(index=cands, columns=topics, fill_value=0).astype(float)
    return ct.div(ct.sum(axis=1), axis=0)
shA, shB = shares(sub[sub.split == "A"]), shares(sub[sub.split == "B"])

c2, c2r = [], []
grand = shA.mean(axis=0).values
for name, col in INSTR.items():
    s = pack[col].values
    pred = np.zeros_like(shB.values)
    for j, k in enumerate(topics):
        m, c0 = np.polyfit(s, shA.values[:, j], 1)
        pred[:, j] = m * s + c0
        c2r.append({"instrument": name, "topic": k,
                    "r_score_vs_shareB": stats.pearsonr(s, shB.values[:, j])[0]})
    pred = np.clip(pred, 0, None)
    pred /= pred.sum(axis=1, keepdims=True)
    js = [jensenshannon(pred[i], shB.values[i], base=2) ** 2 for i in range(len(cands))]
    c2.append({"instrument": name, "mean_JS_divergence": float(np.mean(js))})
gp = np.clip(grand, 0, None); gp /= gp.sum()
c2.append({"instrument": "NULL split-A grand-mean shares",
           "mean_JS_divergence": float(np.mean(
               [jensenshannon(gp, shB.values[i], base=2) ** 2 for i in range(len(cands))]))})
pd.DataFrame(c2).to_csv(OUT / "c2_topic_attention.csv", index=False)
pd.DataFrame(c2r).to_csv(OUT / "c2_topic_r.csv", index=False)

# ------------------------------------------------------------------ C3
c3 = []
for name, col in INSTR.items():
    s = pack[col].values
    ok = ~pack.framing_B.isna().values
    c3.append({"instrument": name, "scope": "overall",
               "n": int(ok.sum()),
               "pearson_r": stats.pearsonr(s[ok], pack.framing_B.values[ok])[0]})
    for k in [0, 1, 2]:
        m = (pack[f"framing_B_topic{k}_n"].values >= 5) & \
            (~pack[f"framing_B_topic{k}"].isna().values)
        if m.sum() >= 10:
            r = stats.pearsonr(s[m], pack[f"framing_B_topic{k}"].values[m])[0]
        else:
            r = np.nan
        c3.append({"instrument": name, "scope": f"topic{k}", "n": int(m.sum()),
                   "pearson_r": r})
pd.DataFrame(c3).to_csv(OUT / "c3_framing.csv", index=False)

print(c1[["instrument", "heldout_logloss", "top1_acc"]].to_string(index=False))
print(f"(C1 excluded candidates with zero split-B retweets: {excluded})")
print(); print(pd.DataFrame(c2).to_string(index=False))
print(); print(pd.DataFrame(c3).pivot(index="scope", columns="instrument",
                                      values="pearson_r").round(3).to_string())
