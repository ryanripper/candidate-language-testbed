"""
08_figures.py — WS3 figures (post-unseal; reads outputs/ only).
Okabe-Ito colorblind-safe palette; one color per instrument/party throughout.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ws3-llm-scaling" / "outputs"
FIG = ROOT / "ws3-llm-scaling" / "figures"
FIG.mkdir(exist_ok=True)

C = {"llm": "#009E73", "tfidf": "#E69F00", "ws1": "#CC79A7", "w2v": "#56B4E9",
     "behav": "#0072B2", "oracle": "#333333", "null": "#999999",
     "D": "#0072B2", "R": "#D55E00", "I": "#999999"}
plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "font.size": 9.5})

val = pd.read_csv(OUT / "validation_results.csv")
pack = pd.read_parquet(OUT / "unsealed_pack.parquet")
ma = pd.read_csv(OUT / "miss_anatomy.csv").set_index("measure").value
bd = pd.read_csv(OUT / "blind_diagnostics.csv").set_index("measure").value

# ---------------------------------------------------------------- fig 1
order = [
    ("Behavioral: mean retweet-source ideology (split A)", C["behav"]),
    ("TF-IDF+SVD PC1 (frozen)", C["tfidf"]),
    ("LLM ask-and-average (pilot, stripped, n=25 x m=5)", C["llm"]),
    ("WS1 Tier A Model2Vec axis", C["ws1"]),
    ("word2vec partisan axis (frozen)", C["w2v"]),
]
labels = ["Behavioral (retweet sources, split A)", "TF-IDF+SVD PC1 (frozen)",
          "LLM ask-and-average (pilot)", "WS1 Model2Vec axis", "word2vec axis"]
vals = [val.set_index("instrument").pearson_r[k] for k, _ in order]
fig, axx = plt.subplots(figsize=(7.2, 3.4))
y = np.arange(len(order))[::-1]
axx.barh(y, vals, height=0.55, color=[c for _, c in order])
for yi, v in zip(y, vals):
    axx.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=9)
axx.axvline(0.90, color="#333333", lw=1, ls="--")
axx.text(0.899, 0.55, "pre-registered bar\nr = 0.90 ", fontsize=8,
         color="#333333", ha="right", va="center")
axx.set_yticks(y); axx.set_yticklabels(labels)
axx.set_xlim(0.85, 1.0); axx.set_xlabel("Pearson r vs true_ideology (n = 150)")
axx.set_title("Stage B — instrument comparison (frozen 150-candidate subsample)",
              fontsize=10.5)
fig.tight_layout(); fig.savefig(FIG / "fig1_instrument_comparison.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 2
fig, axx = plt.subplots(figsize=(5.4, 4.6))
for p in ["D", "I", "R"]:
    m = pack.party == p
    axx.errorbar(pack.true_ideology[m], pack.llm_score[m],
                 yerr=pack.llm_sd[m], fmt="o", ms=4.5, lw=0, elinewidth=0.8,
                 ecolor=C[p], color=C[p], alpha=0.85, label=f"{p} (n={m.sum()})")
r = val.iloc[0].pearson_r
axx.set_xlabel("true_ideology (planted)"); axx.set_ylabel("LLM ask-and-average score")
axx.set_title(f"LLM pilot vs planted ideology — r = {r:.3f}\n"
              "(error bars = across-repetition SD, m = 5)")
axx.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "fig2_llm_vs_truth.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 3
cue = pd.read_csv(OUT / "scores_cue.csv").merge(
    pack[["candidate_id", "party"]], on="candidate_id")
vi = val.set_index("instrument").pearson_r
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
for p in ["D", "I", "R"]:
    m = cue.party == p
    a1.scatter(cue.stripped12_score[m], cue.cue_score[m] - cue.stripped12_score[m],
               s=16, color=C[p], alpha=0.8, label=p)
a1.axhline(0, color="#333333", lw=0.8)
a1.set_xlabel("stripped score (reps 1–2)")
a1.set_ylabel("cue − stripped (paired shift)")
a1.set_title("Cue-bias ablation: org handles left in\n"
             f"mean |shift| = {bd['cue-stripped paired shift mean |shift|']:.3f}")
a1.legend(frameon=False)
names = ["stripped\n(reps 1–2)", "cues intact\n(reps 1–2)",
         "full main\n(m = 5)", "bundle rep-1\n(30 subset)", "tweet-level\n(30 subset)"]
vv = [vi["LLM stripped reps 1-2 (paired comparator)"],
      vi["LLM cue-condition (reps 1-2)"], vi.iloc[0],
      vi["ABLATION bundle rep-1 (same 30)"],
      vi["ABLATION tweet-level mean (30-cand subset)"]]
cols = [C["llm"], "#66c2a5", C["llm"], "#b8b8b8", "#8d8d8d"]
x = np.arange(5)
a2.bar(x, vv, width=0.6, color=cols)
for xi, v in zip(x, vv):
    a2.text(xi, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)
a2.axhline(0.90, color="#333333", lw=1, ls="--")
a2.set_xticks(x); a2.set_xticklabels(names, fontsize=8)
a2.set_ylim(0.85, 1.0); a2.set_ylabel("Pearson r vs true_ideology")
a2.set_title("Design ablations (all clear the 0.90 bar)")
fig.tight_layout(); fig.savefig(FIG / "fig3_ablations.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 4
c1 = pd.read_csv(OUT / "c1_retweet_choice.csv")
c2 = pd.read_csv(OUT / "c2_topic_attention.csv")
c3 = pd.read_csv(OUT / "c3_framing.csv")
imap = {"LLM ask-and-average": ("LLM", C["llm"]), "TF-IDF+SVD PC1": ("TF-IDF", C["tfidf"]),
        "WS1 Tier A Model2Vec": ("WS1", C["ws1"]),
        "Oracle (true_ideology)": ("Oracle", C["oracle"]),
        "NULL uniform": ("Null:\nuniform", C["null"]),
        "NULL split-A org base rates": ("Null:\nbase rates", C["null"]),
        "NULL split-A grand-mean shares": ("Null:\nmean shares", C["null"])}
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11.6, 3.9))
d = c1.sort_values("heldout_logloss")
x = np.arange(len(d))
a1.bar(x, d.heldout_logloss, width=0.62, color=[imap[i][1] for i in d.instrument])
for xi, v, t1 in zip(x, d.heldout_logloss, d.top1_acc):
    a1.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=7.5)
a1.set_xticks(x); a1.set_xticklabels([imap[i][0] for i in d.instrument], fontsize=8)
a1.set_ylim(2.0, 3.15); a1.set_ylabel("held-out log-loss (nats, lower = better)")
a1.set_title("C1 — split-B retweet-source choice\n(softmax fit on split A)")
d = c2.sort_values("mean_JS_divergence")
x = np.arange(len(d))
a2.bar(x, d.mean_JS_divergence, width=0.62, color=[imap[i][1] for i in d.instrument])
for xi, v in zip(x, d.mean_JS_divergence):
    a2.text(xi, v + 0.0008, f"{v:.4f}", ha="center", fontsize=7.5)
a2.set_xticks(x); a2.set_xticklabels([imap[i][0] for i in d.instrument], fontsize=8)
a2.set_ylim(0.14, 0.162); a2.set_ylabel("mean JS divergence (lower = better)")
a2.set_title("C2 — split-B topic attention\n(K = 13 refined topics)")
scopes = ["overall", "topic0", "topic1", "topic2"]
w = 0.19
for k, instr in enumerate(["LLM ask-and-average", "TF-IDF+SVD PC1",
                           "WS1 Tier A Model2Vec", "Oracle (true_ideology)"]):
    dd = c3[c3.instrument == instr].set_index("scope").reindex(scopes)
    a3.bar(np.arange(4) + (k - 1.5) * w, dd.pearson_r, width=w,
           color=imap[instr][1], label=imap[instr][0])
ns = c3[c3.instrument == "LLM ask-and-average"].set_index("scope").reindex(scopes).n
a3.set_xticks(np.arange(4))
a3.set_xticklabels([f"{s}\n(n={int(n)})" for s, n in zip(
    ["overall", "healthcare", "abortion", "guns"], ns)], fontsize=7.5)
a3.set_ylim(0.7, 1.0); a3.set_ylabel("r (score vs split-B framing)")
a3.set_title("C3 — split-B framing intensity")
a3.legend(frameon=False, fontsize=7.5, ncol=2)
fig.suptitle("Stage C — held-out behavioral prediction (estimate on split A, evaluate on split B)",
             y=1.02, fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "fig4_behavior_prediction.png",
                                bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- fig 5
a, b = np.polyfit(pack.llm_score, pack.true_ideology, 1)
err = a * pack.llm_score + b - pack.true_ideology
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.8))
ns = ~pack.small_bundle
a1.scatter(pack.llm_sd[ns], err.abs()[ns], s=16, color=C["llm"], alpha=0.8,
           label="regular bundles")
a1.scatter(pack.llm_sd[~ns], err.abs()[~ns], s=16, color="#b8b8b8", alpha=0.9,
           marker="^", label="small bundles (<25 split-A tweets)")
a1.set_xlabel("across-repetition SD (m = 5)")
a1.set_ylabel("|calibrated error| vs truth")
a1.set_title("Stability vs error\n"
             f"r = {ma['stability: across-rep SD vs |error| (non-small-bundle)']:.3f} "
             "(non-small bundles)")
a1.legend(frameon=False, fontsize=8)
a2.scatter(pack.true_ideology.abs(), err.abs(), s=16, color=C["llm"], alpha=0.8)
a2.set_xlabel("|true_ideology| (0 = moderate)")
a2.set_ylabel("|calibrated error|")
a2.set_title("Miss anatomy: are moderates harder?\n"
             f"r = {ma['|error| vs |true_ideology| (moderates harder if negative)']:.3f}")
fig.tight_layout(); fig.savefig(FIG / "fig5_stability_misses.png"); plt.close(fig)
print("figures written:", sorted(p.name for p in FIG.glob("*.png")))
