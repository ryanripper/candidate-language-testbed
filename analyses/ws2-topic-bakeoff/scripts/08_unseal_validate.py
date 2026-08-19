"""
08_unseal_validate.py — WS2 SINGLE UNSEAL STEP (preregistration §7)
-------------------------------------------------------------------
Runs ONLY after: all five assignments, coherence/diversity table, judge
scores, and Stage C blind machinery are on disk.

Reads sealed_truth.parquet ONCE. Uses true_topic (tweet level) and
true_ideology (candidate level). true_framing is not read.

Writes : outputs/scoreboard.csv          (the bake-off result)
         outputs/decision.json           (winner per pre-registered rule)
         outputs/stagec_validity.csv     (per-topic distance validity)
         outputs/truth_topic_summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
WS0 = HERE.parents[1] / "ws0-harness"
sys.path.insert(0, str(WS0))
from metrics import ari_nmi, distance_validity  # noqa: E402

out = HERE / "outputs"
ENTRANTS = ["lda", "nmf", "lsa", "bertopic", "llm"]

# ---- blind-side inputs (already on disk) ----
coh = pd.read_csv(out / "coherence_diversity.csv").set_index("entrant")
judge = pd.read_csv(out / "judge_scores.csv")
key = json.load(open(out / "judge_key.json"))
jd = judge.merge(pd.DataFrame([{"item_id": k, **v} for k, v in key.items()]),
                 on="item_id")
judge_mean = jd.groupby("entrant")["score"].mean()

# ---------------- THE UNSEAL ----------------
truth = pd.read_parquet(WS0 / "sealed_truth.parquet")
corpus = pd.read_parquet(WS0 / "blind_corpus.parquet")
assert (truth["tweet_id"].to_numpy() == corpus["tweet_id"].to_numpy()).all()
true_topic = truth["true_topic"].to_numpy()
cand_truth = truth.groupby("candidate_id")["true_ideology"].first()
meta = pd.read_csv(WS0 / "baselines" / "candidate_metadata.csv")
ideo = cand_truth.reindex(meta["candidate_id"].astype(str)).to_numpy(float)
# --------------------------------------------

rows = []
for ent in ENTRANTS:
    labels = np.load(out / f"assignments_{ent}.npy")
    s = ari_nmi(labels, true_topic)
    rows.append({"entrant": ent, "ari": round(s["ari"], 4),
                 "nmi": round(s["nmi"], 4),
                 "K": int(len(set(labels.tolist()))),
                 "npmi": round(float(coh.loc[ent, "npmi_mean"]), 4),
                 "c_v": round(float(coh.loc[ent, "c_v"]), 4),
                 "diversity": round(float(coh.loc[ent, "diversity"]), 4),
                 "judge": round(float(judge_mean[ent]), 3)})
board = pd.DataFrame(rows).sort_values("ari", ascending=False)
board.to_csv(out / "scoreboard.csv", index=False)
print(board.to_string(index=False))

n_true = len(set(true_topic.tolist()))
json.dump({"n_true_topics": n_true,
           "true_topic_names": sorted(set(true_topic.tolist()))[:50]},
          open(out / "truth_topic_summary.json", "w"), indent=2, default=str)
print(f"\ntrue number of topics: {n_true}")

# winner rule (preregistration §5)
b = board.reset_index(drop=True)
winner = b.iloc[0]["entrant"]
if len(b) > 1 and b.iloc[0]["ari"] - b.iloc[1]["ari"] < 0.03:
    top2 = b.iloc[:2]
    winner = top2.sort_values("judge", ascending=False).iloc[0]["entrant"]
decision = {
    "winner": winner,
    "winner_ari": float(b.iloc[0]["ari"]),
    "tie_break_used": bool(len(b) > 1 and
                           b.iloc[0]["ari"] - b.iloc[1]["ari"] < 0.03),
    "success_bar_ari_0.60": bool((board["ari"] >= 0.60).any()),
    "rule": "highest ARI; within 0.03 -> higher judge interpretability",
}
json.dump(decision, open(out / "decision.json", "w"), indent=2)
print(json.dumps(decision, indent=2))

# ---- Stage C validity (winner's npz must exist; built blind) ----
sc_path = out / f"stagec_{winner}.npz"
if sc_path.exists():
    sc = np.load(sc_path)
    D_all = sc["D_overall"]
    dv_all = distance_validity(D_all, ideo)
    srows = [{"topic": "ALL", "n_candidates": D_all.shape[0],
              "distance_validity": round(dv_all, 4)}]
    for k in sc.files:
        if not k.startswith("D_") or k == "D_overall":
            continue
        t = k[2:]
        rowsel = sc[f"rows_{t}"]
        dv = distance_validity(sc[k], ideo[rowsel])
        srows.append({"topic": t, "n_candidates": int(len(rowsel)),
                      "distance_validity": round(dv, 4)})
    sv = pd.DataFrame(srows)
    sv.to_csv(out / "stagec_validity.csv", index=False)
    print("\nStage C distance validity (corrected Tier A space):")
    print(sv.to_string(index=False))
else:
    print(f"\nNOTE: {sc_path.name} missing — run 07_stagec.py {winner} "
          "then re-run this script (truth columns unchanged).")
