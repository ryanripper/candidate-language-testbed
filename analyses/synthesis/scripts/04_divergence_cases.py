"""
04_divergence_cases.py — Synthesis step 4: where instruments disagree
---------------------------------------------------------------------
Plan §5, second bullet: "the 5–10 candidates where instruments disagree
most, read qualitatively — numbers flag WHERE to look; reading the
tweets says WHY." This is the quant-qual bridge from NOTES.md
(07/20/26 second entry).

Method (on the n=150 pilot support, the all-instrument support):
  1. z-score each estimated instrument across candidates (all already
     oriented mean-R > mean-D): behav_A, llm, tfidf, w2v, ws1_tierA,
     ws1_tierB. z-score truth on the same support as reference.
  2. Disagreement index = SD of the six instrument z-scores per
     candidate (high = the instruments tell different stories).
  3. Top 8 candidates by disagreement -> case table with each
     instrument's z, its signed deviation from truth-z, and candidate
     covariates that WS1–WS3 flagged as failure modes (retweet share,
     tweet count, small_bundle, LLM rep SD, topic mix).
  4. Dump each case's split-A tweets (the text the LLM instrument saw
     sampled from) to a markdown packet for the qualitative read.

Outputs:
  synthesis/outputs/divergence_index.csv        (all 150, ranked)
  synthesis/outputs/divergence_cases.csv        (top 8, wide)
  synthesis/outputs/divergence_case_tweets.md   (qualitative packet)
  synthesis/figures/fig3_divergence.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "synthesis" / "outputs"
FIG = ROOT / "synthesis" / "figures"

I150 = pd.read_csv(OUT / "instruments_150.csv")
meta = pd.read_csv(ROOT.parent / "ws0-harness/baselines/candidate_metadata.csv")
corpus = pd.read_parquet(ROOT.parent / "ws0-harness/blind_corpus.parquet")
ab = pd.read_parquet(ROOT.parent / "ws0-harness/tweet_split_ab.parquet")
assign = np.load(ROOT / "ws2-topic-bakeoff/outputs/"
                 "assignments_llm_refined.npy")
TOPIC_NAMES = {0: "healthcare", 1: "abortion", 2: "guns",
               3: "crime-policing", 4: "immigration-border",
               5: "taxes-spending", 6: "workers-wages",
               8: "energy-climate", 9: "education-schools",
               12: "foreign-policy", 14: "democracy-reform",
               20: "campaign-process", 100: "retweet-content"}
corpus = corpus.merge(ab[["tweet_id", "split"]], on="tweet_id")
corpus["topic_llm"] = [TOPIC_NAMES.get(t, str(t)) for t in assign]

INSTR = ["behav_A", "llm_score", "tfidf", "w2v", "ws1_tierA", "ws1_tierB"]
SHORT = {"behav_A": "behav", "llm_score": "llm", "tfidf": "tfidf",
         "w2v": "w2v", "ws1_tierA": "m2v", "ws1_tierB": "minilm"}

Z = pd.DataFrame({"candidate_id": I150["candidate_id"]})
for c in INSTR + ["truth"]:
    x = I150[c].to_numpy(dtype=float)
    Z[f"z_{c}"] = (x - np.nanmean(x)) / np.nanstd(x)
zcols = [f"z_{c}" for c in INSTR]
Z["disagreement"] = Z[zcols].std(axis=1, ddof=1)
Z["max_span"] = Z[zcols].max(axis=1) - Z[zcols].min(axis=1)

# candidate covariates (blind-safe ones + WS3 flags)
cov = (corpus.groupby("candidate_id")
       .agg(n_tweets=("tweet_id", "size"),
            retweet_share=("is_retweet", "mean")))
Z = Z.merge(cov, on="candidate_id")
Z = Z.merge(I150[["candidate_id", "party", "truth", "llm_sd",
                  "small_bundle"]], on="candidate_id")
Z = Z.sort_values("disagreement", ascending=False).reset_index(drop=True)
Z.to_csv(OUT / "divergence_index.csv", index=False)

cases = Z.head(8).copy()
cases = cases.merge(meta[["candidate_id", "candidate_name", "chamber",
                          "state", "incumbent"]], on="candidate_id")
cases.to_csv(OUT / "divergence_cases.csv", index=False)

# --------------------------------------------------- qualitative packet
rng = np.random.default_rng(20260727)
lines = ["# Divergence case packet — synthesis stage",
         "",
         "Top-8 instrument-disagreement candidates (n=150 pilot support).",
         "Split-A tweets shown (the split WS3 scores were estimated from);",
         "topic labels are the WS2 refined instrument (no truth used).",
         ""]
for r in cases.itertuples():
    zs = " | ".join(f"{SHORT[c]} {getattr(r, f'z_{c}'):+.2f}"
                    for c in INSTR)
    lines += [f"## {r.candidate_id} — {r.candidate_name} "
              f"({r.party}, {r.chamber}, {r.state})",
              f"- truth z {r.z_truth:+.2f} (raw {r.truth:+.3f}) | {zs}",
              f"- disagreement SD {r.disagreement:.2f}, span "
              f"{r.max_span:.2f} | retweet share {r.retweet_share:.2f} | "
              f"{r.n_tweets} tweets | LLM rep SD {r.llm_sd:.3f}"
              f"{' | SMALL BUNDLE' if r.small_bundle else ''}", ""]
    ct = corpus[(corpus["candidate_id"] == r.candidate_id)
                & (corpus["split"] == "A")]
    mix = ct["topic_llm"].value_counts(normalize=True).head(4)
    lines.append("- split-A topic mix: "
                 + ", ".join(f"{k} {v:.0%}" for k, v in mix.items()))
    take = ct.sample(min(12, len(ct)), random_state=rng.integers(1 << 31))
    take = take.sort_values("is_retweet")
    for t in take.itertuples():
        tag = "RT" if t.is_retweet else "orig"
        lines.append(f"  - [{tag}|{t.topic_llm}] {t.text}")
    lines.append("")
Path(OUT / "divergence_case_tweets.md").write_text("\n".join(lines))

# ------------------------------------------------------------------ fig
fig, axes = plt.subplots(1, 2, figsize=(15, 5.8),
                         gridspec_kw={"width_ratios": [1, 1.4]})
ax = axes[0]
sc = ax.scatter(Z["z_truth"], Z["disagreement"], c=Z["retweet_share"],
                cmap="magma", s=26, edgecolor="k", linewidth=0.3)
for r in cases.itertuples():
    ax.annotate(r.candidate_id, (r.z_truth, r.disagreement),
                fontsize=7, xytext=(4, 3), textcoords="offset points")
fig.colorbar(sc, ax=ax, label="retweet share")
ax.set_xlabel("true ideology (z)")
ax.set_ylabel("instrument disagreement (SD of 6 instrument z-scores)")
ax.set_title("Where the instruments fall out with each other", fontsize=10)

ax = axes[1]
x = np.arange(len(INSTR))
for k, r in enumerate(cases.itertuples()):
    ax.plot(x, [getattr(r, f"z_{c}") for c in INSTR], "-o", ms=4,
            label=f"{r.candidate_id} ({r.party}, truth {r.z_truth:+.1f})")
    ax.scatter([len(INSTR) - 0.6 + 0.08 * k], [r.z_truth], marker="*",
               s=60, color=ax.lines[-1].get_color())
ax.set_xticks(x)
ax.set_xticklabels([SHORT[c] for c in INSTR])
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("score (z)")
ax.set_title("Top-8 divergence cases: instrument profiles "
             "(★ = true ideology)", fontsize=10)
ax.legend(fontsize=7, ncol=2, loc="best")
fig.suptitle("Synthesis fig. 3 — Divergence cases: numbers flag where "
             "to look", fontsize=12)
fig.savefig(FIG / "fig3_divergence.png", dpi=200, bbox_inches="tight")

print(cases[["candidate_id", "party", "truth", "disagreement", "max_span",
             "retweet_share", "n_tweets", "llm_sd", "small_bundle"]]
      .round(3).to_string(index=False))
print("\nDisagreement correlates (n=150):")
for c in ["retweet_share", "n_tweets", "llm_sd"]:
    ok = np.isfinite(Z[c])
    print(f"  {c:14s} r = {np.corrcoef(Z['disagreement'][ok], Z[c][ok])[0,1]:+.3f}")
print(f"  |truth z|      r = "
      f"{np.corrcoef(Z['disagreement'], Z['z_truth'].abs())[0,1]:+.3f}")
print("packet + fig3 saved")
