"""
10b_fig5_refined.py — rebuild fig5 on the Stage B refined instrument (K=13).
Documented deviation from prereg fig spec: the blind winner's 25 fine themes
leave most pair × topic cells empty (<5 tweets); the refined instrument is
the one carried forward, and its coarser topics make the heatmap readable.
The blind-winner version is kept as fig5b for the record.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent.parent
WS0 = HERE.parents[1] / "ws0-harness"
out = HERE / "outputs"
fig_dir = HERE / "figures"
SEED = 20260726

REFINED_NAMES = {0: "healthcare", 1: "abortion", 2: "guns",
                 3: "crime-policing", 4: "immigration-border",
                 5: "taxes-spending", 6: "workers-wages",
                 8: "energy-climate", 9: "education (merged)",
                 12: "foreign-policy (merged)", 14: "democracy (merged)",
                 20: "campaign-process", 100: "retweet-content"}

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": False})

sc = np.load(out / "stagec_llm_refined.npz")
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
    sel = idx[np.argsort(flat[idx])[::-1][:n]] if top else \
        rng.choice(idx, size=n, replace=False)
    return [(iu[0][s], iu[1][s]) for s in sel]


pairs = pick(cross & dr, 5) + pick(~cross & dr, 5) + pick(dr, 5, top=False)
topic_ids = sorted(int(k[2:]) for k in sc.files
                   if k.startswith("D_") and k != "D_overall")
M = np.full((len(pairs), len(topic_ids)), np.nan)
for j, t in enumerate(topic_ids):
    rows = sc[f"rows_{t}"]
    pos = {r: i for i, r in enumerate(rows)}
    Dt = sc[f"D_{t}"]
    for i, (a, b) in enumerate(pairs):
        if a in pos and b in pos:
            M[i, j] = Dt[pos[a], pos[b]]

fig, ax = plt.subplots(figsize=(8.4, 5.2))
cmap = LinearSegmentedColormap.from_list("blues",
                                         ["#f3f7fc", "#2a78d6", "#123a6b"])
cmap.set_bad("#e8e7e3")
im = ax.imshow(M, aspect="auto", cmap=cmap)
ax.set_xticks(np.arange(len(topic_ids)))
ax.set_xticklabels([REFINED_NAMES[t] for t in topic_ids], rotation=45,
                   ha="right", fontsize=7.5)
ax.set_yticks(np.arange(len(pairs)))
ax.set_yticklabels([f"{cid[a]} ({party[a]}) × {cid[b]} ({party[b]})"
                    for a, b in pairs], fontsize=7.5)
for i in (4.5, 9.5):
    ax.axhline(i, color="white", lw=2)
for ytext, lab in ((2, "cross-party\nmost distant"),
                   (7, "same-party\nmost distant"), (12, "random")):
    ax.text(-0.34, ytext, lab, transform=ax.get_yaxis_transform(),
            fontsize=7.5, va="center", ha="right", color="#52514e",
            style="italic")
cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("within-topic cosine distance (corrected Model2Vec)", fontsize=8)
ax.set_title('"Far from whom, on what" — refined instrument (K=13)\n'
             "(gray = pair not comparable: a member has <5 tweets in topic)")
fig.tight_layout()
# Preserve the blind-winner fig5 as fig5b before overwriting — the docstring
# promises it is "kept for the record", so keep it in code, not by hand.
fig5 = fig_dir / "fig5_pair_topic_heatmap.png"
fig5b = fig_dir / "fig5b_pair_topic_heatmap_blindwinner.png"
if fig5.exists() and not fig5b.exists():
    import shutil
    shutil.copy2(fig5, fig5b)
    print("blind-winner fig5 preserved as fig5b")
fig.savefig(fig5)
print("fig5 rebuilt on refined instrument")
