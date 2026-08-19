"""
03_blind_diagnostics.py — BLIND-SAFE checks, run and saved BEFORE unseal
(preregistration §6 and §8-blind).

Reads no WS3 truth. One caveat, declared in the preregistration: the
topic-entropy covariate uses ws2's assignments_llm_refined.npy, whose
L2/L3 refinement steps were WS2-post-unseal (truth-informed at the WS2
stage). No per-candidate ideology truth enters this script, but "blind" is
relative to the WS3 unseal, not truth-free end to end.

Outputs: outputs/blind_diagnostics.csv, outputs/blind_covariates.csv
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent / "ws0-harness"))
import metrics  # noqa: E402

OUT = ROOT / "ws3-llm-scaling" / "outputs"

llm = pd.read_csv(OUT / "scores_main.csv")
cands = llm.candidate_id.tolist()

meta = pd.read_csv(ROOT.parent / "ws0-harness" / "baselines" / "candidate_metadata.csv")
party = meta.set_index("candidate_id").party.reindex(cands).values

# frozen comparison instruments, restricted to the 150 (blind agreement only)
ax = pd.read_csv(ROOT.parent / "ws0-harness" / "baselines" / "axis_scores.csv").set_index("candidate_id")
ws1 = pd.read_csv(ROOT / "ws1-sentence-transformers" / "outputs" /
                  "blind_axis_scores.csv")
ws1A = (ws1[(ws1.tier == "A") & (ws1.space == "centered")]
        .set_index("candidate_id").blind_axis_score.reindex(cands).values)
tfidf = ax.tfidf_partisan_score.reindex(cands).values
# blind orientation convention (mean R > mean D), idempotent
ws1A = metrics.orient_axis(ws1A, party)
tfidf = metrics.orient_axis(tfidf, party)
s = metrics.orient_axis(llm.llm_score.values, party)  # should be a no-op; verified below
assert np.allclose(s, llm.llm_score.values), "LLM score needed a sign flip — investigate"

# observable covariates on split A
bc = pd.read_parquet(ROOT.parent / "ws0-harness" / "blind_corpus.parquet")
ab = pd.read_parquet(ROOT.parent / "ws0-harness" / "tweet_split_ab.parquet")
df = bc.merge(ab[["tweet_id", "split"]], on="tweet_id")
topics = np.load(ROOT / "ws2-topic-bakeoff" / "outputs" /
                 "assignments_llm_refined.npy")
df["topic"] = topics
A = df[(df.split == "A") & df.candidate_id.isin(cands)]
g = A.groupby("candidate_id")
cov = pd.DataFrame({
    "rt_share_A": g.is_retweet.mean(),
    "log10_vol_A": np.log10(g.size()),
    "topic_entropy_A": g.topic.apply(
        lambda t: stats.entropy(t.value_counts(normalize=True))),
}).reindex(cands)
cov.insert(0, "candidate_id", cov.index)
cov.to_csv(OUT / "blind_covariates.csv", index=False)

cue = pd.read_csv(OUT / "scores_cue.csv").set_index("candidate_id").reindex(cands)
dr = np.isin(party, ["D", "R"])
y = (party[dr] == "R").astype(float)

rows = []
def add(name, val):
    rows.append({"measure": name, "value": float(val)})

add("LLM D/R point-biserial", stats.pearsonr(s[dr], y)[0])
add("LLM vs TF-IDF frozen axis (blind r, n=150)", stats.pearsonr(s, tfidf)[0])
add("LLM vs WS1 TierA axis (blind r, n=150)", stats.pearsonr(s, ws1A)[0])
add("LLM score min", s.min()); add("LLM score max", s.max())
add("LLM score share |s|>0.5", float((np.abs(s) > 0.5).mean()))
add("median across-rep SD", llm.llm_sd.median())
for c in ["rt_share_A", "log10_vol_A", "topic_entropy_A"]:
    add(f"confound screen: LLM score vs {c} (gate |r|>=0.6 on rt/vol)",
        stats.pearsonr(s, cov[c].values)[0])
add("cue-condition D/R point-biserial",
    stats.pearsonr(cue.cue_score.values[dr], y)[0])
add("stripped(reps1-2) D/R point-biserial",
    stats.pearsonr(cue.stripped12_score.values[dr], y)[0])
shift = cue.cue_score - cue.stripped12_score
add("cue-stripped paired shift mean", shift.mean())
add("cue-stripped paired shift mean |shift|", shift.abs().mean())
add("cue shift mean (D)", shift[party == "D"].mean())
add("cue shift mean (R)", shift[party == "R"].mean())
add("cue vs stripped12 r", stats.pearsonr(cue.cue_score, cue.stripped12_score)[0])

out = pd.DataFrame(rows)
out.to_csv(OUT / "blind_diagnostics.csv", index=False)
print(out.to_string(index=False))
