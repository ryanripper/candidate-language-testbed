"""
09_stageb_augment.py — WS2 Stage B: LLM augmentation of the winner (POST-UNSEAL)
--------------------------------------------------------------------------------
Everything here is explicitly post-unseal MITIGATION MEASUREMENT, not blind
discovery (preregistration §7). The winner (direct LLM theming, ARI 0.289)
missed the 0.60 bar; per prereg, a guided/refined rerun is run and the ARI
delta reported.

The refinement ladder (each step's rationale is stated with what info it uses):
  L0  winner as-is (blind)                                    [blind]
  L1  L0 + retweet routing: is_retweet==True -> single 'retweet-content'
      theme. Uses only the OBSERVABLE is_retweet column — this was available
      blind; the bake-off simply didn't think of it.            [blind-available]
  L2  L1 + dissolve the two genre themes the unseal showed to be boundary
      errors (community-visits, generic-values): their originals re-routed to
      the nearest POLICY-theme centroid in MiniLM space.       [unseal-informed]
  L3  L2 + taxonomy coarsening: merge remaining 23 themes to the granularity
      the judge/merge analysis suggests (student-debt->education,
      veterans+trade->foreign-policy, election-integrity+democracy-reform+
      gotv->democracy, rallies/endorsements/fundraising/volunteers/holidays/
      debates->campaign-process).                              [unseal-informed]
  SUP majority-vote merge of L0 themes onto true topics — the supervised
      merge ceiling for reference.                             [supervised]

Writes : outputs/stageb_ladder.csv, outputs/assignments_llm_refined.npy (L3),
         outputs/stageb_flagged_topics.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
WS0 = HERE.parents[1] / "ws0-harness"
ST = HERE.parent / "ws1-sentence-transformers"
sys.path.insert(0, str(WS0))
from metrics import ari_nmi  # noqa: E402

out = HERE / "outputs"
corpus = pd.read_parquet(WS0 / "blind_corpus.parquet")
truth = pd.read_parquet(WS0 / "sealed_truth.parquet")  # already unsealed at 08
tt = truth["true_topic"].to_numpy()
lab0 = np.load(out / "assignments_llm.npy")
tax = json.load(open(out / "llm_taxonomy.json"))
names = {t["id"]: t["name"] for t in tax["themes"]}

# judge-flagged topics (prereg: interpretability <= 2.5)
judge = pd.read_csv(out / "judge_scores.csv")
key = json.load(open(out / "judge_key.json"))
kdf = pd.DataFrame([{"item_id": k, **v} for k, v in key.items()])
j = judge.merge(kdf, on="item_id")
flagged = j[(j.entrant == "llm") & (j.score <= 2.5)]
json.dump({"flagged": [{"topic": int(r.topic), "name": names[int(r.topic)],
                        "score": float(r.score)} for r in flagged.itertuples()]},
          open(out / "stageb_flagged_topics.json", "w"), indent=2)
print(f"judge-flagged LLM topics (<=2.5): {len(flagged)}")

rows = [{"level": "L0_blind_winner", **ari_nmi(lab0, tt)}]

# L1: observable retweet routing
RT = 100
lab1 = lab0.copy()
lab1[corpus["is_retweet"].to_numpy()] = RT
rows.append({"level": "L1_retweet_routing", **ari_nmi(lab1, tt)})

# L2: dissolve genre themes -> nearest policy centroid (MiniLM space)
POLICY = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
GENRE_DISSOLVE = [15, 24]  # community-visits, generic-values
emb = np.load(ST / "intermediate" / "emb_tierB.npz", allow_pickle=True)
X = emb["X"].astype(np.float32)
X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
sample = pd.read_csv(out / "llm_sample_labels.csv")
cent, ids = [], []
for t in POLICY:
    idx = sample.loc[sample.theme_id == t, "row_idx"].to_numpy()
    if len(idx) == 0:
        continue
    c = X[idx].mean(0)
    cent.append(c / (np.linalg.norm(c) + 1e-12))
    ids.append(t)
cent = np.stack(cent)
lab2 = lab1.copy()
mask = np.isin(lab2, GENRE_DISSOLVE)
lab2[mask] = np.array(ids)[np.argmax(X[mask] @ cent.T, axis=1)]
rows.append({"level": "L2_dissolve_genres", **ari_nmi(lab2, tt)})

# L3: taxonomy coarsening
MERGE = {10: 9,              # student-debt -> education-schools
         11: 12, 7: 12,      # veterans, trade -> foreign-policy
         13: 14, 16: 14,     # election-integrity, gotv -> democracy(14)
         17: 20, 18: 20, 19: 20, 21: 20, 22: 20}  # process -> campaign(20)
lab3 = lab2.copy()
for a, b in MERGE.items():
    lab3[lab3 == a] = b
# Guard: MERGE deliberately omits theme 23 (horse-race-news) because in the
# committed run every L0 tweet labeled 23 is a retweet, rerouted to RT at
# L1. If that ever stops holding, an unmapped 14th topic would silently
# survive, breaking the K=13 design and the REFINED_NAMES dicts downstream
# (10b_fig5_refined.py, synthesis/03) — fail loudly instead.
EXPECTED_L3 = set(POLICY) | {20, RT}
leftover = set(np.unique(lab3)) - EXPECTED_L3
assert not leftover, (
    f"L3 coarsening left unmapped theme ids {sorted(leftover)} — extend "
    "MERGE/GENRE_DISSOLVE (theme 23 horse-race-news is the known gap) "
    "before writing assignments_llm_refined.npy")
rows.append({"level": "L3_coarsened", **ari_nmi(lab3, tt)})
np.save(out / "assignments_llm_refined.npy", lab3)

# SUP: majority-merge ceiling of the BLIND winner
mapping = pd.DataFrame({"l": lab0, "t": tt}).groupby("l")["t"] \
            .agg(lambda s: s.value_counts().index[0])
sup = pd.Series(lab0).map(mapping).to_numpy()
rows.append({"level": "SUP_majority_ceiling_of_L0", **ari_nmi(sup, tt)})

lad = pd.DataFrame(rows)
lad["ari"] = lad["ari"].round(4)
lad["nmi"] = lad["nmi"].round(4)
lad.to_csv(out / "stageb_ladder.csv", index=False)
print(lad.to_string(index=False))
print(f"\nK at L3: {len(set(lab3.tolist()))}")
