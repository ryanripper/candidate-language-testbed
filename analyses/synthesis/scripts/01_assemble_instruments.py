"""
01_assemble_instruments.py — Synthesis stage (plan §5), step 1
--------------------------------------------------------------
Assemble every instrument's candidate-level score into one master table,
on two supports:

  * n=910 (full corpus): truth, behavioral (all retweets), TF-IDF PC1,
    w2v partisan axis, WS1 Tier A (Model2Vec), WS1 Tier B (MiniLM).
  * n=150 (WS3 pilot subsample): all of the above + LLM ask-and-average
    and the split-A behavioral score used by WS3 (behav_A).

The LLM instrument exists ONLY on the 150-candidate pilot support:
the D1 full 910x5 API scale-up was DEFERRED by Ryan on 2026-07-27
(gate cleared at r=.970 but pinned/not funded). All-instrument
comparisons therefore run on n=150.

Session seed: 20260727. All truth reads are post-unseal by design —
every workstream's preregistered validation already ran; synthesis is
the §5 stage that begins "once all three workstreams have unsealed."

Inputs (paths relative to project root):
  ws0-harness/baselines/candidate_metadata.csv     — canonical 910 row order
  ws0-harness/baselines/axis_scores.csv            — frozen TF-IDF + w2v axes
  ws0-harness/sealed_truth.parquet                 — true_ideology (tweet-level)
  ws0-harness/blind_corpus.parquet                 — retweet sources (behavioral)
  ws1-sentence-transformers/outputs/blind_axis_scores.csv
  ws3-llm-scaling/outputs/unsealed_pack.parquet   — LLM pilot scores
  ws3-llm-scaling/outputs/org_ideologies.json

Outputs:
  synthesis/outputs/instruments_910.csv
  synthesis/outputs/instruments_150.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent / "ws0-harness"))
from metrics import orient_axis  # noqa: E402

OUT = ROOT / "synthesis" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ load
meta = pd.read_csv(ROOT.parent / "ws0-harness/baselines/candidate_metadata.csv")
axis = pd.read_csv(ROOT.parent / "ws0-harness/baselines/axis_scores.csv")
truth = (
    pd.read_parquet(ROOT.parent / "ws0-harness/sealed_truth.parquet")
    .groupby("candidate_id")["true_ideology"].first()
)
ws1 = pd.read_csv(
    ROOT / "ws1-sentence-transformers/outputs/blind_axis_scores.csv"
)
pack = pd.read_parquet(
    ROOT / "ws3-llm-scaling/outputs/unsealed_pack.parquet"
)
orgs = json.load(open(ROOT / "ws3-llm-scaling/outputs/org_ideologies.json"))
corpus = pd.read_parquet(
    ROOT.parent / "ws0-harness/blind_corpus.parquet",
    columns=["candidate_id", "is_retweet", "retweeted_handle"],
)

# ------------------------------------------- behavioral score (full corpus)
rts = corpus[corpus["is_retweet"]].copy()
rts["src_ideo"] = rts["retweeted_handle"].map(orgs)
assert rts["src_ideo"].notna().all(), "unmapped retweet source"
behav_full = rts.groupby("candidate_id")["src_ideo"].mean()

# ------------------------------------------------- WS1 tiers, wide format
ws1_wide = ws1.pivot_table(
    index="candidate_id", columns=["tier"], values="blind_axis_score"
)
ws1_wide.columns = [f"ws1_tier{t}" for t in ws1_wide.columns]

# ---------------------------------------------------------- master (910)
m = meta[["candidate_id", "party", "chamber", "incumbent"]].copy()
m = m.merge(axis, on="candidate_id")
m = m.merge(ws1_wide, on="candidate_id", how="left")
m["behavioral"] = m["candidate_id"].map(behav_full)  # NaN for 0-retweet cands
m["truth"] = m["candidate_id"].map(truth)
m = m.rename(columns={
    "tfidf_partisan_score": "tfidf",
    "w2v_partisan_score": "w2v",
})[["candidate_id", "party", "chamber", "incumbent",
    "tfidf", "w2v", "ws1_tierA", "ws1_tierB", "behavioral", "truth"]]

# Blind-safe orientation (mean R > mean D) for every estimated instrument,
# so signs are comparable across the agreement matrix.
party = m["party"].to_numpy()
for col in ["tfidf", "w2v", "ws1_tierA", "ws1_tierB"]:
    m[col] = orient_axis(m[col].to_numpy(), party)

assert len(m) == 910 and m["candidate_id"].is_unique
n_no_rt = int(m["behavioral"].isna().sum())
print(f"[910] behavioral undefined for {n_no_rt} zero-retweet candidates")

m.to_csv(OUT / "instruments_910.csv", index=False)

# ---------------------------------------------------------- pilot (150)
p = pack[["candidate_id", "llm_score", "llm_sd", "small_bundle",
          "behav_A"]].copy()
sub = m.merge(p, on="candidate_id", how="inner")
assert len(sub) == 150, len(sub)
sub.to_csv(OUT / "instruments_150.csv", index=False)

# ----------------------------------------------------------- sanity print
def r(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[ok], b[ok])[0, 1]

t = sub["truth"].to_numpy()
print("[150 support] r vs truth — reproduction check against ws3 "
      "validation_results.csv:")
for col in ["llm_score", "tfidf", "w2v", "ws1_tierA", "behav_A"]:
    print(f"  {col:10s} r = {r(sub[col].to_numpy(), t):+.4f}")
print(f"[910 support] tfidf r = {r(m['tfidf'].to_numpy(), m['truth'].to_numpy()):+.4f}"
      f"  (frozen: +0.9738)")
print(f"[910 support] behavioral r = "
      f"{r(m['behavioral'].to_numpy(), m['truth'].to_numpy()):+.4f}"
      f"  (plan ceiling: +0.980)")
print("assemble: OK")
