"""
06_unseal_validate.py — THE single unseal step (preregistration §7).
`ws0-harness/sealed_truth.parquet` is read exactly once, here, after all LLM scores
(scripts 01–02) and blind diagnostics (script 03) are on disk. Numbering
gap 04–05 is deliberate: the preregistration names this script 06.

Reads truth: true_ideology (candidate-level), true_framing (split-B rows).
Also parses the 20 planted org ideologies from the generator (unseal-scoped).

Outputs:
  outputs/validation_results.csv   Stage B table
  outputs/miss_anatomy.csv         error diagnostics + post-unseal confound screen
  outputs/unsealed_pack.parquet    candidate-level truth-derived quantities for 07
  outputs/org_ideologies.json
  outputs/decision.json            pilot gate (r >= 0.90) & D1 recommendation
"""
from __future__ import annotations
import ast, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent / "ws0-harness"))
import metrics  # noqa: E402

OUT = ROOT / "ws3-llm-scaling" / "outputs"
for req in ["scores_main.csv", "scores_cue.csv", "scores_tweetlevel.csv",
            "blind_diagnostics.csv", "blind_covariates.csv"]:
    assert (OUT / req).exists(), f"blind artifact missing before unseal: {req}"

llm = pd.read_csv(OUT / "scores_main.csv")
cands = llm.candidate_id.tolist()
meta = pd.read_csv(ROOT.parent / "ws0-harness" / "baselines" / "candidate_metadata.csv")
party = meta.set_index("candidate_id").party.reindex(cands).values

# ---------------------------------------------------------------- UNSEAL (once)
truth = pd.read_parquet(ROOT.parent / "ws0-harness" / "sealed_truth.parquet")
cand_truth = (truth.groupby("candidate_id").true_ideology.first()
              .reindex(cands))
t = cand_truth.values

# org ideologies: planted generator truth
gen = (ROOT.parent / "data" / "synthetic-candidate-tweets" /
       "generate_synthetic_candidates.py").read_text()
pairs = re.findall(r'\("(@\w+)",\s*(-?\d+\.?\d*)\)', gen)
orgs = {h: float(v) for h, v in pairs}
assert len(orgs) == 20, f"expected 20 orgs, parsed {len(orgs)}"
json.dump(orgs, open(OUT / "org_ideologies.json", "w"), indent=1)

# split-B framing targets + split-A behavioral instrument
bc = pd.read_parquet(ROOT.parent / "ws0-harness" / "blind_corpus.parquet")
ab = pd.read_parquet(ROOT.parent / "ws0-harness" / "tweet_split_ab.parquet")
df = bc.merge(ab[["tweet_id", "split"]], on="tweet_id")
df["topic"] = np.load(ROOT / "ws2-topic-bakeoff" / "outputs" /
                      "assignments_llm_refined.npy")
df = df.merge(truth[["tweet_id", "true_framing"]], on="tweet_id")
fr_labels = sorted(str(x) for x in df.true_framing.unique())
def fmap(x):
    if pd.isna(x):
        return np.nan  # unframed rows are excluded from framing means
    x = str(x).lower()
    return -1.0 if x.startswith("lib") else (1.0 if x.startswith("con") else 0.0)
df["framing_val"] = df.true_framing.map(fmap)

sub = df[df.candidate_id.isin(cands)]
B = sub[sub.split == "B"]
A = sub[sub.split == "A"]
framing_B = B.groupby("candidate_id").framing_val.mean().reindex(cands)
fr_topic = {}
for k in [0, 1, 2]:
    g = B[B.topic == k].groupby("candidate_id").framing_val
    fr_topic[k] = pd.concat([g.mean().rename("val"), g.size().rename("n")], axis=1)

Art = A[A.is_retweet]
behav_A = (Art.retweeted_handle.map(orgs).groupby(Art.candidate_id)
           .mean().reindex(cands))

# frozen instruments on the same 150 (blind-oriented as in script 03)
ax = pd.read_csv(ROOT.parent / "ws0-harness" / "baselines" / "axis_scores.csv").set_index("candidate_id")
ws1 = pd.read_csv(ROOT / "ws1-sentence-transformers" / "outputs" /
                  "blind_axis_scores.csv")
instruments = {
    "LLM ask-and-average (pilot, stripped, n=25 x m=5)": llm.llm_score.values,
    "LLM cue-condition (reps 1-2)":
        pd.read_csv(OUT / "scores_cue.csv").set_index("candidate_id")
        .cue_score.reindex(cands).values,
    "LLM stripped reps 1-2 (paired comparator)":
        pd.read_csv(OUT / "scores_cue.csv").set_index("candidate_id")
        .stripped12_score.reindex(cands).values,
    "TF-IDF+SVD PC1 (frozen)":
        metrics.orient_axis(ax.tfidf_partisan_score.reindex(cands).values, party),
    "word2vec partisan axis (frozen)":
        metrics.orient_axis(ax.w2v_partisan_score.reindex(cands).values, party),
    "WS1 Tier A Model2Vec axis":
        metrics.orient_axis(
            ws1[(ws1.tier == "A") & (ws1.space == "centered")]
            .set_index("candidate_id").blind_axis_score.reindex(cands).values, party),
    "Behavioral: mean retweet-source ideology (split A)": behav_A.values,
}

rows = []
for name, s in instruments.items():
    ok = ~pd.isna(s)
    r = metrics.axis_recovery(np.asarray(s)[ok], t[ok])
    rows.append({"instrument": name, "n": int(ok.sum()),
                 "pearson_r": r["pearson_r"], "spearman_rho": r["spearman_rho"]})

# tweet-level ablation on its 30-candidate subset (vs bundle rep-1 on same 30)
tw = pd.read_csv(OUT / "scores_tweetlevel.csv").set_index("candidate_id")
sub30 = tw.index.tolist()
t30 = cand_truth.reindex(sub30).values
bund = pd.read_csv(OUT / "scores_bundles.csv")
rep1 = (bund[(bund.condition == "main") & (bund.rep == 1)]
        .set_index("candidate_id").score.reindex(sub30).values)
r_tw = metrics.axis_recovery(tw.tweetlevel_score.values, t30)
r_b1 = metrics.axis_recovery(rep1, t30)
rows.append({"instrument": "ABLATION tweet-level mean (30-cand subset)",
             "n": 30, "pearson_r": r_tw["pearson_r"],
             "spearman_rho": r_tw["spearman_rho"]})
rows.append({"instrument": "ABLATION bundle rep-1 (same 30)", "n": 30,
             "pearson_r": r_b1["pearson_r"], "spearman_rho": r_b1["spearman_rho"]})
val = pd.DataFrame(rows)
val.to_csv(OUT / "validation_results.csv", index=False)

# ------------------------------------------------ miss anatomy & error screen
s = llm.llm_score.values
a1, b1 = np.polyfit(s, t, 1)          # affine calibration for error space
err = (a1 * s + b1) - t
cov = pd.read_csv(OUT / "blind_covariates.csv").set_index("candidate_id").reindex(cands)
ma = []
ns = ~llm.small_bundle.values
ma.append({"measure": "stability: across-rep SD vs |error| (non-small-bundle)",
           "value": stats.pearsonr(llm.llm_sd.values[ns], np.abs(err)[ns])[0]})
ma.append({"measure": "|error| vs |true_ideology| (moderates harder if negative)",
           "value": stats.pearsonr(np.abs(err), np.abs(t))[0]})
for c in ["log10_vol_A", "rt_share_A", "topic_entropy_A"]:
    ma.append({"measure": f"|error| vs {c}",
               "value": stats.pearsonr(np.abs(err), cov[c].values)[0]})
    ma.append({"measure": f"error vs {c} (post-unseal confound gate |r|>=0.6)",
               "value": stats.pearsonr(err, cov[c].values)[0]})
ma.append({"measure": "RMSE of affine-calibrated LLM score",
           "value": float(np.sqrt(np.mean(err ** 2)))})
ma.append({"measure": "cue-bias delta r (cue - stripped12)",
           "value": val.loc[val.instrument.str.startswith("LLM cue"), "pearson_r"].iloc[0]
                    - val.loc[val.instrument.str.startswith("LLM stripped"), "pearson_r"].iloc[0]})
pd.DataFrame(ma).to_csv(OUT / "miss_anatomy.csv", index=False)

# ------------------------------------------------ pack for Stage C (07)
pack = pd.DataFrame({
    "candidate_id": cands, "party": party, "true_ideology": t,
    "llm_score": s, "llm_sd": llm.llm_sd.values,
    "small_bundle": llm.small_bundle.values,
    "tfidf_score": instruments["TF-IDF+SVD PC1 (frozen)"],
    "ws1_score": instruments["WS1 Tier A Model2Vec axis"],
    "behav_A": behav_A.values, "framing_B": framing_B.values,
})
for k in [0, 1, 2]:
    pack[f"framing_B_topic{k}"] = fr_topic[k].val.reindex(cands).values
    pack[f"framing_B_topic{k}_n"] = fr_topic[k].n.reindex(cands).fillna(0).values
pack.to_parquet(OUT / "unsealed_pack.parquet", index=False)

r_main = float(val.iloc[0].pearson_r)
decision = {
    "preregistered_bar": 0.90,
    "llm_pilot_r": r_main,
    "bar_cleared": bool(r_main >= 0.90),
    "d1_recommendation": ("scale to full 910 x 5 via Anthropic API (Ryan's key)"
                          if r_main >= 0.90 else
                          "do not spend API budget; pilot is the result"),
    "true_framing_labels_found": fr_labels,
}
json.dump(decision, open(OUT / "decision.json", "w"), indent=1)
print(val.to_string(index=False))
print(); print(pd.DataFrame(ma).to_string(index=False))
print(); print(json.dumps(decision, indent=1))
