"""
03_distance_agreement.py — Synthesis step 3: Mantel / Procrustes
----------------------------------------------------------------
Distance-matrix agreement across instruments (plan §5, first bullet).

Instruments as DISTANCE matrices on the n=150 pilot support:
  D_truth   |true ideology gap|            (oracle geometry)
  D_behav   |split-A behavioral gap|       (n=146 complete)
  D_llm     |LLM score gap|                (pilot; D1 scale-up deferred)
  D_tfidf   frozen WS0 baseline            (lexical)
  D_w2v_c   corrected word2vec             (07-20 instrument)
  D_wsA     WS1 Tier A corrected centroid  (exploratory best, dv=.640)
  D_wsB     WS1 Tier B corrected centroid

Analyses:
  1. Pairwise Mantel tests (999 permutations, seed 20260727) on 150.
  2. Pairwise Mantel r on the full 910 support for the five non-LLM
     matrices (r only; every p at this n is <<.001 and the permutation
     cost buys nothing).
  3. Procrustes similarity of 2-D classical-MDS configurations —
     "do the instruments draw the same MAP of the space, not just the
     same pairwise orderings."
  4. Topic-conditioned tie-in (WS2 x WS3): correlate WS2's refined
     per-topic distance matrices with |LLM gap| vs |truth gap| on the
     candidates common to (topic support ∩ pilot 150). Question: does
     the LLM instrument reproduce WS2's three-tier signal ladder
     (retweet-content >> policy >> campaign-process) without truth?

Outputs:
  synthesis/outputs/mantel_150.csv, mantel_150_pvalues.csv
  synthesis/outputs/mantel_910.csv
  synthesis/outputs/procrustes_150.csv
  synthesis/outputs/topic_conditioned_agreement.csv
  synthesis/figures/fig2_distance_agreement.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent / "ws0-harness"))
from metrics import mantel_test, procrustes_similarity, upper_tri  # noqa: E402

OUT = ROOT / "synthesis" / "outputs"
FIG = ROOT / "synthesis" / "figures"
SEED = 20260727

meta = pd.read_csv(ROOT.parent / "ws0-harness/baselines/candidate_metadata.csv")
I150 = pd.read_csv(OUT / "instruments_150.csv")
row_of = {c: i for i, c in enumerate(meta["candidate_id"])}
idx150 = np.array([row_of[c] for c in I150["candidate_id"]])

def gap_matrix(x: np.ndarray) -> np.ndarray:
    return np.abs(x[:, None] - x[None, :])

D910 = {
    "tfidf": np.load(ROOT.parent / "ws0-harness/baselines/D_tfidf.npy"),
    "w2v_corr": np.load(ROOT.parent / "ws0-harness/baselines/D_w2v_corrected.npy"),
    "ws1_A": np.load(ROOT / "ws1-sentence-transformers/outputs/"
                     "D_tierA_corrected_centroid_cosine.npy"),
    "ws1_B": np.load(ROOT / "ws1-sentence-transformers/outputs/"
                     "D_tierB_corrected_centroid_cosine.npy"),
}
I910 = pd.read_csv(OUT / "instruments_910.csv")
D910["truth"] = gap_matrix(I910["truth"].to_numpy())

D150 = {k: D[np.ix_(idx150, idx150)] for k, D in D910.items()}
D150["llm"] = gap_matrix(I150["llm_score"].to_numpy())
D150["behav"] = gap_matrix(I150["behav_A"].to_numpy())  # NaN rows handled below
ORDER = ["truth", "behav", "llm", "tfidf", "w2v_corr", "ws1_A", "ws1_B"]
LABELS = {"truth": "oracle |gap|", "behav": "behavioral |gap|",
          "llm": "LLM |gap|", "tfidf": "TF-IDF", "w2v_corr": "w2v corr.",
          "ws1_A": "Model2Vec corr.", "ws1_B": "MiniLM corr."}

# ------------------------------------------------------- 1. Mantel on 150
ok_behav = np.isfinite(I150["behav_A"].to_numpy())
n = len(ORDER)
MR = np.eye(n)
MP = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        a, b = ORDER[i], ORDER[j]
        if "behav" in (a, b):
            sel = np.where(ok_behav)[0]
            Da, Db = D150[a][np.ix_(sel, sel)], D150[b][np.ix_(sel, sel)]
        else:
            Da, Db = D150[a], D150[b]
        res = mantel_test(Da, Db, permutations=999, seed=SEED)
        MR[i, j] = MR[j, i] = res["mantel_r"]
        MP[i, j] = MP[j, i] = res["p_value"]
MRdf = pd.DataFrame(MR, index=ORDER, columns=ORDER)
MRdf.to_csv(OUT / "mantel_150.csv")
pd.DataFrame(MP, index=ORDER, columns=ORDER).to_csv(
    OUT / "mantel_150_pvalues.csv")

# --------------------------------------------------- 2. Mantel r on 910
O910 = ["truth", "tfidf", "w2v_corr", "ws1_A", "ws1_B"]
M9 = np.eye(len(O910))
tri = {k: upper_tri(D910[k]) for k in O910}
for i in range(len(O910)):
    for j in range(i + 1, len(O910)):
        r_ = stats.pearsonr(tri[O910[i]], tri[O910[j]])[0]
        M9[i, j] = M9[j, i] = r_
M9df = pd.DataFrame(M9, index=O910, columns=O910)
M9df.to_csv(OUT / "mantel_910.csv")

# ------------------------------------------- 3. Procrustes of MDS maps
def cmds_2d(D: np.ndarray) -> np.ndarray:
    """Classical (Torgerson) MDS, top-2 dims, deterministic."""
    D = np.asarray(D, float)
    n_ = D.shape[0]
    J = np.eye(n_) - np.ones((n_, n_)) / n_
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    o = np.argsort(w)[::-1][:2]
    return V[:, o] * np.sqrt(np.maximum(w[o], 0))

sel = np.where(ok_behav)[0]
configs = {k: cmds_2d(D150[k][np.ix_(sel, sel)]) for k in ORDER}
PR = np.eye(n)
for i in range(n):
    for j in range(i + 1, n):
        s = procrustes_similarity(configs[ORDER[i]], configs[ORDER[j]])
        PR[i, j] = PR[j, i] = s["similarity"]
PRdf = pd.DataFrame(PR, index=ORDER, columns=ORDER)
PRdf.to_csv(OUT / "procrustes_150.csv")

# --------------------------- 4. topic-conditioned agreement (WS2 x WS3)
TOPIC_NAMES = {0: "healthcare", 1: "abortion", 2: "guns",
               3: "crime-policing", 4: "immigration-border",
               5: "taxes-spending", 6: "workers-wages",
               8: "energy-climate", 9: "education-schools",
               12: "foreign-policy", 14: "democracy-reform",
               20: "campaign-process", 100: "retweet-content"}
z = np.load(ROOT / "ws2-topic-bakeoff/outputs/stagec_llm_refined.npz")
ws2_ref = pd.read_csv(ROOT / "ws2-topic-bakeoff/outputs/"
                      "stagec_validity_refined.csv")
llm_by_row = dict(zip(idx150, I150["llm_score"]))
truth_by_row = dict(zip(idx150, I150["truth"]))
rows150 = set(idx150.tolist())
rec = []
for t, name in TOPIC_NAMES.items():
    Dt = z[f"D_{t}"]
    rows_t = z[f"rows_{t}"]
    common = [k for k, r_ in enumerate(rows_t) if r_ in rows150]
    if len(common) < 25:
        continue
    sub_rows = rows_t[common]
    Dsub = Dt[np.ix_(common, common)].astype(float)
    llm_gap = gap_matrix(np.array([llm_by_row[r_] for r_ in sub_rows]))
    tru_gap = gap_matrix(np.array([truth_by_row[r_] for r_ in sub_rows]))
    r_llm = stats.pearsonr(upper_tri(Dsub), upper_tri(llm_gap))[0]
    r_tru = stats.pearsonr(upper_tri(Dsub), upper_tri(tru_gap))[0]
    ws2_910 = ws2_ref.loc[ws2_ref["topic"].astype(str) == str(t),
                          "distance_validity"]
    rec.append({"topic": t, "name": name, "n_candidates": len(common),
                "r_vs_llm_gap_150": r_llm, "r_vs_truth_gap_150": r_tru,
                "ws2_distance_validity_910": float(ws2_910.iloc[0])})
TC = pd.DataFrame(rec).sort_values("ws2_distance_validity_910",
                                   ascending=False)
TC.to_csv(OUT / "topic_conditioned_agreement.csv", index=False)
rank_r = stats.spearmanr(TC["r_vs_llm_gap_150"],
                         TC["ws2_distance_validity_910"])[0]

# ------------------------------------------------------------------ fig
fig = plt.figure(figsize=(16.5, 6.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35], wspace=0.42)

def heat(ax, M, order, title):
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels([LABELS[o] for o in order], rotation=40,
                       ha="right", fontsize=8)
    ax.set_yticklabels([LABELS[o] for o in order], fontsize=8)
    for i in range(len(order)):
        for j in range(len(order)):
            ax.text(j, i, f"{M[i, j]:.2f}".lstrip("0") or "1.0",
                    ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] < 0.72 else "black")
    ax.set_title(title, fontsize=9.5)
    return im

ax1 = fig.add_subplot(gs[0])
heat(ax1, MRdf.values, ORDER,
     "Mantel r between distance matrices\n(n=150 pilot support; all p ≤ .001)")
ax2 = fig.add_subplot(gs[1])
im = heat(ax2, PRdf.values, ORDER,
          "Procrustes similarity of 2-D MDS maps\n(1 − disparity, n=146)")
fig.colorbar(im, ax=[ax1, ax2], shrink=0.62, label="agreement")

ax3 = fig.add_subplot(gs[2])
y = np.arange(len(TC))
ax3.barh(y + 0.2, TC["r_vs_truth_gap_150"], height=0.38,
         label="vs oracle |gap| (n=150 support)", color="#40507a")
ax3.barh(y - 0.2, TC["r_vs_llm_gap_150"], height=0.38,
         label="vs LLM |gap| (no truth used)", color="#c95f4e")
ax3.set_yticks(y)
ax3.set_yticklabels([f"{r.name} ({r.ws2_distance_validity_910:.2f})"
                     for r in TC.itertuples()], fontsize=8)
ax3.invert_yaxis()
ax3.set_xlabel("corr(within-topic distance, |score gap|)", fontsize=9)
ax3.set_title("WS2 topic-conditioned distances vs WS3 instrument\n"
              f"(labels show WS2 910-support validity; "
              f"tier-order Spearman ρ = {rank_r:.2f})", fontsize=9.5)
ax3.axvline(0, color="k", lw=0.6)
ax3.legend(fontsize=8, loc="lower right")
fig.suptitle("Synthesis fig. 2 — Do the instruments draw the same geometry?",
             fontsize=12, y=1.00)
fig.savefig(FIG / "fig2_distance_agreement.png", dpi=200,
            bbox_inches="tight")

print("Mantel r (150):"); print(MRdf.round(3).to_string())
print("\nMantel r (910):"); print(M9df.round(3).to_string())
print("\nProcrustes similarity (146):"); print(PRdf.round(3).to_string())
print("\nTopic-conditioned:"); print(TC.round(3).to_string(index=False))
print(f"\ntier-order Spearman (LLM vs WS2 910 validity): {rank_r:.3f}")
print("fig2 saved")
