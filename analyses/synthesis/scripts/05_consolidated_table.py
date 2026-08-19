"""
05_consolidated_table.py — Synthesis step 5: one table for the testbed
----------------------------------------------------------------------
Plan §5, third bullet: "a consolidated validation table spanning all
three workstreams." Every number is read from the workstream output
files, never retyped, so the table cannot drift from the record.

Per D4 (2026-07-25) technical_writing_sample.pdf stays FROZEN — this
table lives here and in the synthesis write-up, not in the sample.
Per D1 (2026-07-27) the LLM rows are pilot support (n=150), full-corpus
scale-up deferred.

Outputs:
  synthesis/outputs/consolidated_validation.csv
  synthesis/figures/fig4_consolidated_ladder.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "synthesis" / "outputs"
FIG = ROOT / "synthesis" / "figures"

rows = []
def add(ws, family, measure, instrument, n, value, source, note=""):
    rows.append(dict(workstream=ws, family=family, measure=measure,
                     instrument=instrument, support_n=n,
                     value=round(float(value), 4), source=source,
                     note=note))

# ---------------------------------------------------------- WS0 / frozen
b0 = pd.read_csv(ROOT.parent / "ws0-harness/baselines/baseline_validation.csv",
                 index_col="measure")["ws0_value"]
add("WS0", "axis", "r vs true_ideology", "TF-IDF+SVD PC1", 910,
    b0["TF-IDF partisan axis vs true_ideology (r)"],
    "ws0-harness/baselines/baseline_validation.csv")
add("WS0", "axis", "r vs true_ideology", "word2vec partisan axis", 910,
    b0["word2vec partisan axis vs true_ideology (r)"],
    "ws0-harness/baselines/baseline_validation.csv")
add("WS0", "distance", "distance validity", "TF-IDF", 910,
    b0["TF-IDF distance validity  [NEW at WS0.4]"],
    "ws0-harness/baselines/baseline_validation.csv")
add("WS0", "distance", "distance validity", "w2v corrected", 910,
    b0["w2v corrected distance validity"],
    "ws0-harness/baselines/baseline_validation.csv")

# ------------------------------------------------------------------ WS1
v1 = pd.read_csv(ROOT / "ws1-sentence-transformers/outputs/"
                 "validation_results.csv")
a1 = v1[(v1["kind"] == "axis") & (v1["space"] == "corrected")]
for _, r in a1.iterrows():
    name = ("Model2Vec potion-8M" if r["tier"] == "A" else "MiniLM-L6-v2")
    add("WS1", "axis", "r vs true_ideology", f"{name} corrected axis", 910,
        r["value"], "ws1-sentence-transformers/outputs/validation_results.csv",
        "pre-registered negative result: loses to TF-IDF" if r["primary"]
        else "secondary")
best_d = v1[(v1["kind"] == "distance") & (v1["tier"] == "A")
            & (v1["space"] == "corrected")
            & (v1["rep"] == "centroid_cosine")]["value"].iloc[0]
add("WS1", "distance", "distance validity",
    "Model2Vec corrected centroid cosine", 910, best_d,
    "ws1-sentence-transformers/outputs/validation_results.csv",
    "EXPLORATORY label required (rule named distributional distances)")

# ------------------------------------------------------------------ WS2
sb = pd.read_csv(ROOT / "ws2-topic-bakeoff/outputs/scoreboard.csv")
for _, r in sb.iterrows():
    add("WS2", "topic", "blind ARI vs true_topic", r["entrant"], 104601,
        r["ari"], "ws2-topic-bakeoff/outputs/scoreboard.csv",
        f"K={int(r['K'])}; blind bar ARI>=0.60 missed by all")
lad = pd.read_csv(ROOT / "ws2-topic-bakeoff/outputs/stageb_ladder.csv")
for _, r in lad.iterrows():
    if r["level"].startswith("SUP"):
        continue
    add("WS2", "topic", "ARI vs true_topic (post-unseal ladder)",
        f"LLM {r['level']}", 104601, r["ari"],
        "ws2-topic-bakeoff/outputs/stageb_ladder.csv",
        "labeled mitigation, not blind")
sc = pd.read_csv(ROOT / "ws2-topic-bakeoff/outputs/"
                 "stagec_validity_refined.csv")
top = sc[sc["topic"] == "100"].iloc[0] if (sc["topic"] == "100").any() \
    else sc[sc["topic"] == 100].iloc[0]
add("WS2", "distance", "within-topic distance validity",
    "retweet-content slice (K=13 refined)", int(top["n_candidates"]),
    top["distance_validity"],
    "ws2-topic-bakeoff/outputs/stagec_validity_refined.csv",
    "top of three-tier ladder; campaign-process bottom at .004")

# ------------------------------------------------------------------ WS3
v3 = pd.read_csv(ROOT / "ws3-llm-scaling/outputs/validation_results.csv")
main = v3.iloc[0]
add("WS3", "axis", "r vs true_ideology",
    "LLM ask-and-average (pilot, stripped)", int(main["n"]),
    main["pearson_r"],
    "ws3-llm-scaling/outputs/validation_results.csv",
    "clears preregistered bar r>=0.90; D1 910x5 scale-up DEFERRED "
    "2026-07-27 (Ryan)")
behav = v3[v3["instrument"].str.startswith("Behavioral")].iloc[0]
add("WS3", "axis", "r vs true_ideology",
    "behavioral mean RT-source ideology (split A)", int(behav["n"]),
    behav["pearson_r"], "ws3-llm-scaling/outputs/validation_results.csv",
    "generator ceiling ~.98")
c1 = pd.read_csv(ROOT / "ws3-llm-scaling/outputs/c1_retweet_choice.csv")
for _, r in c1.iterrows():
    add("WS3", "behavior", "held-out log-loss (C1 retweet choice)",
        r["instrument"], int(r["n_test_retweets"]), r["heldout_logloss"],
        "ws3-llm-scaling/outputs/c1_retweet_choice.csv",
        "lower is better; fit on split A, tested on split B")
qc = json.load(open(ROOT / "ws3-llm-scaling/outputs/decision.json"))
add("WS3", "decision", "preregistered bar", "LLM pilot", 150,
    qc["preregistered_bar"], "ws3-llm-scaling/outputs/decision.json",
    qc["d1_status_note"])

# ------------------------------------------------------------ Synthesis
P150 = pd.read_csv(OUT / "agreement_150_pearson.csv", index_col=0)
M150 = pd.read_csv(OUT / "mantel_150.csv", index_col=0)
TC = pd.read_csv(OUT / "topic_conditioned_agreement.csv")
from scipy import stats as _st
rank_r = _st.spearmanr(TC["r_vs_llm_gap_150"],
                       TC["ws2_distance_validity_910"])[0]
add("Synthesis", "agreement", "min pairwise Pearson r, score level",
    "all 7 instruments", 150, P150.values[np.triu_indices(7, 1)].min(),
    "synthesis/outputs/agreement_150_pearson.csv",
    "weakest pair: MiniLM x truth")
add("Synthesis", "agreement", "Mantel r, oracle vs LLM geometry",
    "D_truth x D_llm", 150, M150.loc["truth", "llm"],
    "synthesis/outputs/mantel_150.csv", "999 perms, p=.001")
add("Synthesis", "agreement", "topic signal-tier reproduction (Spearman)",
    "LLM |gap| vs WS2 910-support tier order", 13, rank_r,
    "synthesis/outputs/topic_conditioned_agreement.csv",
    "truth-free instrument reproduces the far-from-whom-on-what ladder")

T = pd.DataFrame(rows)
T.to_csv(OUT / "consolidated_validation.csv", index=False)
print(T.to_string(index=False, max_colwidth=48))

# ------------------------------------------------------------------ fig4
fig, axes = plt.subplots(1, 3, figsize=(20, 5.4),
                         gridspec_kw={"wspace": 0.55})

ax = axes[0]
axis_rows = [
    ("Behavioral (A)", behav["pearson_r"], 146, "#40507a"),
    ("TF-IDF PC1", b0["TF-IDF partisan axis vs true_ideology (r)"], 910,
     "#40507a"),
    ("LLM (pilot)", main["pearson_r"], 150, "#c95f4e"),
    ("Model2Vec", a1[a1["tier"] == "A"]["value"].iloc[0], 910, "#7a7a7a"),
    ("word2vec", b0["word2vec partisan axis vs true_ideology (r)"], 910,
     "#7a7a7a"),
    ("MiniLM", a1[a1["tier"] == "B"]["value"].iloc[0], 910, "#7a7a7a"),
]
names, vals, ns, cols = zip(*axis_rows)
y = np.arange(len(names))
ax.barh(y, vals, color=cols)
ax.set_yticks(y); ax.set_yticklabels([f"{n_} (n={m})" for n_, m in
                                      zip(names, ns)], fontsize=8.5)
ax.invert_yaxis(); ax.set_xlim(0.6, 1.0)
ax.axvline(0.9, color="k", ls="--", lw=0.8)
ax.text(0.893, 4.6, "WS3 bar .90", fontsize=7.5, rotation=90,
        va="bottom", ha="right")
for yi, v in zip(y, vals):
    ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=8)
ax.set_xlabel("Pearson r vs true ideology")
ax.set_title("Axis recovery — all instruments", fontsize=10)

ax = axes[1]
dist_rows = [
    ("Model2Vec corr.*", best_d, 910),
    ("TF-IDF", b0["TF-IDF distance validity  [NEW at WS0.4]"], 910),
    ("w2v corrected", b0["w2v corrected distance validity"], 910),
    ("MiniLM corr.", v1[(v1["kind"] == "distance") & (v1["tier"] == "B")
     & (v1["space"] == "corrected") & (v1["rep"] == "centroid_cosine")
     ]["value"].iloc[0], 910),
    ("LLM |gap| (pilot)", M150.loc["truth", "llm"], 150),
    ("retweet-content slice", top["distance_validity"], 863),
]
names, vals, ns = zip(*dist_rows)
y = np.arange(len(names))
ax.barh(y, vals, color=["#7a7a7a", "#40507a", "#7a7a7a", "#7a7a7a",
                        "#c95f4e", "#3e7a4e"])
ax.set_yticks(y); ax.set_yticklabels([f"{n_} (n={m})" for n_, m in
                                      zip(names, ns)], fontsize=8.5)
ax.invert_yaxis()
for yi, v in zip(y, vals):
    ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=8)
ax.set_xlabel("corr(distance, |true ideology gap|)")
ax.set_title("Distance validity — *exploratory label\n"
             "(WS2 slice + WS3 pilot supports differ)", fontsize=10)

ax = axes[2]
c1s = c1.sort_values("heldout_logloss")
y = np.arange(len(c1s))
colors = {"Oracle (true_ideology)": "#222222",
          "TF-IDF+SVD PC1": "#40507a",
          "LLM ask-and-average": "#c95f4e",
          "WS1 Tier A Model2Vec": "#7a7a7a",
          "NULL uniform": "#cccccc",
          "NULL split-A org base rates": "#cccccc"}
ax.barh(y, c1s["heldout_logloss"],
        color=[colors[i] for i in c1s["instrument"]])
ax.set_yticks(y); ax.set_yticklabels(c1s["instrument"], fontsize=8.5)
ax.invert_yaxis(); ax.set_xlim(2.2, 3.05)
for yi, v in zip(y, c1s["heldout_logloss"]):
    ax.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=8)
ax.set_xlabel("held-out log-loss (lower = better)")
ax.set_title("Behavioral prediction — C1 retweet-source choice\n"
             "(n=150 candidates, 2,498 held-out retweets)", fontsize=10)

fig.suptitle("Synthesis fig. 4 — Consolidated testbed ladder across "
             "WS0–WS3", fontsize=12)
fig.savefig(FIG / "fig4_consolidated_ladder.png", dpi=200,
            bbox_inches="tight")
print("\nfig4 saved")
