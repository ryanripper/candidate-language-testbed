"""
10_figures.py — WS2 figures (post-unseal; reads outputs only)
-------------------------------------------------------------
fig1  bake-off scoreboard (ARI + NMI)
fig2  blind coherence vs unsealed ARI (the interpretability question)
fig3  Stage B mitigation ladder
fig4  Stage C per-topic distance validity (policy vs process)
fig5  "far from whom, on what" pair × topic heatmap
Palette: dataviz reference categorical slots; sequential = single blue ramp.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
WS0 = HERE.parents[1] / "ws0-harness"
out = HERE / "outputs"
fig_dir = HERE / "figures"
fig_dir.mkdir(exist_ok=True)
SEED = 20260726

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY = "#9a9891"
plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e8e7e3",
                     "grid.linewidth": 0.6, "axes.axisbelow": True})

board = pd.read_csv(out / "scoreboard.csv")
tax = json.load(open(out / "llm_taxonomy.json"))
names = {t["id"]: t["name"] for t in tax["themes"]}

# ---------------------------------------------------------------- fig1
fig, ax = plt.subplots(figsize=(7, 3.2))
y = np.arange(len(board))[::-1]
ax.barh(y + 0.2, board["ari"], height=0.36, color=BLUE, label="ARI (primary)")
ax.barh(y - 0.2, board["nmi"], height=0.36, color=AQUA, label="NMI")
for yi, (a, n) in zip(y, zip(board["ari"], board["nmi"])):
    ax.text(a + 0.01, yi + 0.2, f"{a:.3f}", va="center", fontsize=8)
    ax.text(n + 0.01, yi - 0.2, f"{n:.3f}", va="center", fontsize=8,
            color="#52514e")
ax.axvline(0.60, color=ORANGE, lw=1.2, ls="--")
ax.text(0.605, y.max() + 0.55, "pre-registered bar (ARI ≥ 0.60)",
        color=ORANGE, fontsize=8)
ax.set_yticks(y)
ax.set_yticklabels([f"{e}  (K={k})" for e, k in
                    zip(board["entrant"], board["K"])])
ax.set_xlim(0, 1.0)
ax.set_xlabel("agreement with planted true_topic (104,601 tweets)")
ax.set_title("WS2 Stage A — blind topic-recovery bake-off")
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(fig_dir / "fig1_scoreboard.png")
plt.close(fig)

# ---------------------------------------------------------------- fig2
fig, ax = plt.subplots(figsize=(5.4, 3.6))
ax.scatter(board["npmi"], board["ari"], s=60, color=BLUE, zorder=3)
for _, r in board.iterrows():
    ax.annotate(r["entrant"], (r["npmi"], r["ari"]),
                xytext=(6, 4), textcoords="offset points", fontsize=8.5)
ax.set_xlabel("blind NPMI coherence (uniform c-TF-IDF top-10)")
ax.set_ylabel("ARI vs true_topic (unsealed)")
ax.set_title("Blind coherence did not rank recovery")
fig.tight_layout()
fig.savefig(fig_dir / "fig2_coherence_vs_ari.png")
plt.close(fig)

# ---------------------------------------------------------------- fig3
lad = pd.read_csv(out / "stageb_ladder.csv")
steps = lad[lad.level != "SUP_majority_ceiling_of_L0"]
sup = float(lad.loc[lad.level == "SUP_majority_ceiling_of_L0", "ari"].iloc[0])
labels_map = {
    "L0_blind_winner": "L0\nblind winner",
    "L1_retweet_routing": "L1\n+ retweets → own topic\n(observable rule)",
    "L2_dissolve_genres": "L2\n+ dissolve genre themes",
    "L3_coarsened": "L3\n+ coarsen to K=13",
}
fig, ax = plt.subplots(figsize=(6.4, 3.4))
x = np.arange(len(steps))
colors = [BLUE, AQUA, AQUA, AQUA]
ax.bar(x, steps["ari"], width=0.6, color=colors)
for xi, a in zip(x, steps["ari"]):
    ax.text(xi, a + 0.015, f"{a:.3f}", ha="center", fontsize=8.5)
ax.axhline(0.60, color=ORANGE, lw=1.2, ls="--")
ax.text(-0.38, 0.615, "0.60 bar", color=ORANGE, fontsize=8)
ax.axhline(sup, color=GRAY, lw=1.2, ls=":")
ax.text(len(steps) - 0.45, sup + 0.012,
        f"majority-merge ceiling of L0 ({sup:.3f})", color="#52514e",
        fontsize=7.5, ha="right")
ax.set_xticks(x)
ax.set_xticklabels([labels_map[s] for s in steps["level"]], fontsize=7.5)
ax.set_ylabel("ARI vs true_topic")
ax.set_ylim(0, 1.0)
ax.set_title("Stage B — post-unseal mitigation ladder (LLM winner)")
fig.tight_layout()
fig.savefig(fig_dir / "fig3_stageb_ladder.png")
plt.close(fig)

# ---------------------------------------------------------------- fig4
sv = pd.read_csv(out / "stagec_validity.csv")
overall = float(sv.loc[sv.topic == "ALL", "distance_validity"].iloc[0])
sv = sv[sv.topic != "ALL"].copy()
sv["tid"] = sv["topic"].astype(int)
POLICY = set(range(15))
sv["kind"] = np.where(sv.tid.isin(POLICY), "policy", "process")
sv["name"] = sv.tid.map(names)
sv = sv.sort_values("distance_validity")
fig, ax = plt.subplots(figsize=(6.8, 5.4))
cols = {"policy": BLUE, "process": YELLOW}
ax.barh(np.arange(len(sv)), sv["distance_validity"],
        color=[cols[k] for k in sv["kind"]], height=0.7)
ax.set_yticks(np.arange(len(sv)))
ax.set_yticklabels([f"{n}  (n={c})" for n, c in
                    zip(sv["name"], sv["n_candidates"])], fontsize=7.5)
ax.axvline(overall, color=ORANGE, lw=1.2, ls="--")
ax.text(overall + 0.005, len(sv) - 0.8, f"overall corrected space ({overall:.3f})",
        color=ORANGE, fontsize=7.5, rotation=90, va="top")
ax.axvline(0.6238, color=GRAY, lw=1.0, ls=":")
ax.text(0.6238 - 0.012, 0.2, "TF-IDF frozen bar (0.624)", color="#52514e",
        fontsize=7.5, rotation=90, va="bottom")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="policy topic"),
                   Patch(color=YELLOW, label="process/genre topic")],
          loc="lower right", frameon=False)
ax.set_xlabel("within-topic distance validity  corr(distance, |Δ true_ideology|)")
ax.set_title("Stage C — where the partisan signal lives (winner's topics,\n"
             "corrected Model2Vec space)")
fig.tight_layout()
fig.savefig(fig_dir / "fig4_topic_validity.png")
plt.close(fig)

# ---------------------------------------------------------------- fig5
sc = np.load(out / "stagec_llm.npz")
meta = pd.read_csv(WS0 / "baselines" / "candidate_metadata.csv")
party = meta["party"].to_numpy()
cid = meta["candidate_id"].astype(str).to_numpy()
D = sc["D_overall"]
iu = np.triu_indices(D.shape[0], k=1)
flat = D[iu]
cross = party[iu[0]] != party[iu[1]]
dr = np.isin(party[iu[0]], ["D", "R"]) & np.isin(party[iu[1]], ["D", "R"])
rng = np.random.default_rng(SEED)

def pick(mask, n, top=True):
    idx = np.where(mask)[0]
    if top:
        sel = idx[np.argsort(flat[idx])[::-1][:n]]
    else:
        sel = rng.choice(idx, size=n, replace=False)
    return [(iu[0][s], iu[1][s]) for s in sel]

pairs = (pick(cross & dr, 5) + pick(~cross & dr, 5) + pick(dr, 5, top=False))
pair_kind = ["cross-party\nmost distant"] * 5 + ["same-party\nmost distant"] * 5 \
            + ["random"] * 5

topic_ids = sorted([int(k[2:]) for k in sc.files
                    if k.startswith("D_") and k != "D_overall"])
policy_ids = [t for t in topic_ids if t < 15]
M = np.full((len(pairs), len(policy_ids)), np.nan)
for j, t in enumerate(policy_ids):
    rows = sc[f"rows_{t}"]
    pos = {r: i for i, r in enumerate(rows)}
    Dt = sc[f"D_{t}"]
    for i, (a, b) in enumerate(pairs):
        if a in pos and b in pos:
            M[i, j] = Dt[pos[a], pos[b]]

fig, ax = plt.subplots(figsize=(8.2, 5.2))
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("blues", ["#f3f7fc", "#2a78d6", "#123a6b"])
cmap.set_bad("#e8e7e3")
im = ax.imshow(M, aspect="auto", cmap=cmap)
ax.set_xticks(np.arange(len(policy_ids)))
ax.set_xticklabels([names[t] for t in policy_ids], rotation=45, ha="right",
                   fontsize=7.5)
ax.set_yticks(np.arange(len(pairs)))
ax.set_yticklabels([f"{cid[a]} ({party[a]}) × {cid[b]} ({party[b]})"
                    for a, b in pairs], fontsize=7.5)
for i in (4.5, 9.5):
    ax.axhline(i, color="white", lw=2)
ax.text(-0.32, 2, "cross-party\nmost distant", transform=ax.get_yaxis_transform(),
        fontsize=7.5, va="center", ha="right", color="#52514e", style="italic")
ax.text(-0.32, 7, "same-party\nmost distant", transform=ax.get_yaxis_transform(),
        fontsize=7.5, va="center", ha="right", color="#52514e", style="italic")
ax.text(-0.32, 12, "random", transform=ax.get_yaxis_transform(),
        fontsize=7.5, va="center", ha="right", color="#52514e", style="italic")
cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("within-topic cosine distance (corrected Model2Vec)", fontsize=8)
ax.set_title('"Far from whom, on what" — candidate-pair × topic distances\n'
             "(gray = pair not comparable in topic: <5 tweets for a member)")
ax.grid(False)
fig.tight_layout()
fig.savefig(fig_dir / "fig5_pair_topic_heatmap.png")
plt.close(fig)

print("figures written:", sorted(p.name for p in fig_dir.glob("*.png")))
