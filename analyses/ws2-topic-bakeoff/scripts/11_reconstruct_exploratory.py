"""
11_reconstruct_exploratory.py — RECONSTRUCTION of two post-unseal outputs.

PROVENANCE NOTE: outputs/exploratory_rt_routing.csv and
outputs/stagec_validity_refined.csv were originally produced by ad-hoc
in-session computations after the WS2 unseal (2026-07-26/27) and were
committed without a generating script. This script, added at the 2026-08
audit, reconstructs both from committed artifacts so the audit trail from
truth to these numbers lives in code. Both outputs are EXPLORATORY /
post-unseal by the repo's labeling convention: nothing here is blind, and
nothing here feeds a pre-registered decision.

Part 1 — exploratory_rt_routing.csv
    For every entrant: ARI of the blind assignment vs true_topic, and ARI
    after the observable retweet-routing convention (all retweets -> one
    label; ARI is label-invariant, so the specific id is irrelevant).
    Reconstruction verified against the committed CSV: all five entrants'
    ari_blind match scoreboard.csv and llm's ari_rt_routed matches
    stageb_ladder.csv L1 (0.7647).

Part 2 — stagec_validity_refined.csv
    Stage-C distance validity per refined topic, mirroring the Stage-C
    block of 08_unseal_validate.py but on the refined instrument's tensor
    outputs/stagec_llm_refined.npz (regenerate it first if absent:
    `python 07_stagec.py llm_refined` — see REGENERATE.md).

Run AFTER 08_unseal_validate.py (reads sealed truth; single-unseal
discipline already broken by design at this point in the pipeline).
"""
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
RT = 100  # convention from 09_stageb_augment.py; any fresh label works

blind = pd.read_parquet(WS0 / "blind_corpus.parquet",
                        columns=["tweet_id", "is_retweet"])
sealed = pd.read_parquet(WS0 / "sealed_truth.parquet",
                         columns=["tweet_id", "true_topic"])
assert (blind["tweet_id"].to_numpy() == sealed["tweet_id"].to_numpy()).all()
tt = sealed["true_topic"].to_numpy()
is_rt = blind["is_retweet"].to_numpy()

# ---- Part 1: retweet-routing ARI deltas -----------------------------------
rows = []
for e in ENTRANTS:
    lab = np.load(out / f"assignments_{e}.npy")
    assert len(lab) == len(tt), f"{e}: assignment length mismatch"
    routed = lab.copy()
    routed[is_rt] = max(int(lab.max()) + 1, RT)
    a0 = ari_nmi(lab, tt)["ari"]
    a1 = ari_nmi(routed, tt)["ari"]
    rows.append({"entrant": e, "ari_blind": round(a0, 4),
                 "ari_rt_routed": round(a1, 4), "delta": round(a1 - a0, 4)})
rt_tab = pd.DataFrame(rows)
rt_tab.to_csv(out / "exploratory_rt_routing.csv", index=False)
print(rt_tab.to_string(index=False))

# ---- Part 2: Stage-C validity on the refined instrument -------------------
sc_path = out / "stagec_llm_refined.npz"
if not sc_path.exists():
    print(f"\nNOTE: {sc_path.name} missing — regenerate it with "
          "`python 07_stagec.py llm_refined`, then re-run this script.")
    sys.exit(0)

ideo = (pd.read_parquet(WS0 / "sealed_truth.parquet",
                        columns=["candidate_id", "true_ideology"])
        .groupby("candidate_id")["true_ideology"].first())
meta = pd.read_csv(WS0 / "baselines" / "candidate_metadata.csv")
ideo = ideo.reindex(meta["candidate_id"]).to_numpy()

sc = np.load(sc_path)
D_all = sc["D_overall"]
srows = [{"topic": "ALL", "n_candidates": D_all.shape[0],
          "distance_validity": round(distance_validity(D_all, ideo), 4)}]
for k in sc.files:
    if not k.startswith("D_") or k == "D_overall":
        continue
    t = k[2:]
    rowsel = sc[f"rows_{t}"]
    srows.append({"topic": t, "n_candidates": int(len(rowsel)),
                  "distance_validity":
                      round(distance_validity(sc[k], ideo[rowsel]), 4)})
sv = pd.DataFrame(srows)
sv.to_csv(out / "stagec_validity_refined.csv", index=False)
print("\nStage C distance validity (refined instrument):")
print(sv.to_string(index=False))
