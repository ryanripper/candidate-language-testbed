"""
02_collect_scores.py — parse raw agent scores, validate coverage, aggregate
(BLIND: touches no truth). Outputs:
  outputs/scores_bundles.csv   every bundle-level score (main + cue)
  outputs/scores_main.csv      per-candidate ask-and-average score, SD, n_reps
  outputs/scores_cue.csv       per-candidate cue-condition score (reps 1-2)
                               + paired stripped reps 1-2 mean
  outputs/scores_tweetlevel.csv  per-candidate tweet-level mean (30-cand subset)
  outputs/scoring_qc.json      coverage / parse QC
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ws3-llm-scaling" / "outputs"

def load_json(p: Path):
    t = p.read_text().strip()
    t = re.sub(r"^```(json)?|```$", "", t, flags=re.M).strip()
    return json.loads(t)

manifest = json.load(open(OUT / "batch_manifest.json"))
qc = {"batches": {}, "missing": [], "out_of_range": []}
records = []
for name, expected in manifest.items():
    p = OUT / "raw_scores" / f"{name}.json"
    arr = load_json(p)
    key = "item_id" if name.startswith("tweet") else "bundle_id"
    got = {}
    for o in arr:
        s = float(o["score"])
        if not (-1.0 <= s <= 1.0):
            qc["out_of_range"].append({name: o}); s = max(-1.0, min(1.0, s))
        got[o[key]] = s
    miss = [i for i in expected if i not in got]
    extra = [i for i in got if i not in expected]
    qc["batches"][name] = {"expected": len(expected), "got": len(got),
                           "missing": miss, "extra": extra}
    qc["missing"] += miss
    for i, s in got.items():
        if i in expected:
            records.append({"id": i, "score": s, "batch": name})
sc = pd.DataFrame(records)

bmap = pd.read_csv(OUT / "bundle_map.csv")
bs = bmap.merge(sc.rename(columns={"id": "bundle_id"})[["bundle_id", "score"]],
                on="bundle_id", how="left")
bs.to_csv(OUT / "scores_bundles.csv", index=False)

main = bs[bs.condition == "main"].dropna(subset=["score"])
agg = (main.groupby("candidate_id")
       .agg(llm_score=("score", "mean"), llm_sd=("score", "std"),
            n_reps=("score", "size"), small_bundle=("small_bundle", "first"))
       .reset_index())
agg.to_csv(OUT / "scores_main.csv", index=False)

cue = bs[bs.condition == "cue"].dropna(subset=["score"])
cue_agg = cue.groupby("candidate_id").score.mean().rename("cue_score")
strip12 = (main[main.rep <= 2].groupby("candidate_id").score.mean()
           .rename("stripped12_score"))
pd.concat([cue_agg, strip12], axis=1).reset_index().to_csv(
    OUT / "scores_cue.csv", index=False)

tmap = pd.read_csv(OUT / "tweetlevel_map.csv")
ts = tmap.merge(sc.rename(columns={"id": "item_id"})[["item_id", "score"]],
                on="item_id", how="left").dropna(subset=["score"])
(ts.groupby("candidate_id")
 .agg(tweetlevel_score=("score", "mean"), n_tweets=("score", "size"))
 .reset_index().to_csv(OUT / "scores_tweetlevel.csv", index=False))

qc["n_bundle_scores"] = int(len(sc) - len(ts))
qc["n_tweet_scores"] = int(len(ts))
qc["n_candidates_main"] = int(len(agg))
qc["rep_counts"] = main.groupby("candidate_id").size().value_counts().to_dict()
json.dump(qc, open(OUT / "scoring_qc.json", "w"), indent=1, default=str)
print("main candidates:", len(agg), "| mean n_reps:", agg.n_reps.mean(),
      "| missing:", len(qc["missing"]), "| out_of_range:", len(qc["out_of_range"]))
print("score range:", agg.llm_score.min().round(3), "..", agg.llm_score.max().round(3),
      "| median SD:", agg.llm_sd.median().round(3))
