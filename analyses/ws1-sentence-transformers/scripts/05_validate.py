"""
05_validate.py — WS1 E1.5: THE UNSEAL STEP
------------------------------------------
Reads ws0-harness/sealed_truth.parquet exactly once, per preregistration §7
(only per-candidate true_ideology is used; true_topic / true_framing are
not read). Everything scored here was fixed by scripts 01–04 before this
script ran.

  * axis recovery (metrics.axis_recovery) for the pre-declared primary
    instrument (Tier B corrected-space partisan axis) and all secondary
    axes (Tier A / Tier C, centered + corrected)
  * distance validity (metrics.distance_validity) + blind ratios for every
    pre-registered distance matrix
  * comparison against ws0-harness/baselines/baseline_validation.csv frozen values
  * applies the preregistration §5 decision rule
  * figures 3 (validity by tier/variant/rep) and 4 (axis scatter)

Writes: outputs/validation_results.csv, outputs/decision.json,
        figures/fig3_distance_validity.png, figures/fig4_axis_scatter.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
WS0 = HERE.parents[1] / "ws0-harness"
sys.path.insert(0, str(WS0))
from metrics import axis_recovery, distance_validity, orient_axis  # noqa: E402

NAMES = {"A": "Model2Vec potion-8M", "B": "MiniLM-L6-v2",
         "C": "bge-small-en-v1.5"}


def main() -> None:
    meta = pd.read_csv(WS0 / "baselines" / "candidate_metadata.csv")
    cand_order = meta["candidate_id"].tolist()
    party = meta["party"].to_numpy()

    # ---------------- THE SINGLE UNSEAL ----------------
    truth_df = pd.read_parquet(WS0 / "sealed_truth.parquet",
                               columns=["candidate_id", "true_ideology"])
    truth = (truth_df.groupby("candidate_id")["true_ideology"].first()
             .loc[cand_order].to_numpy())
    print(f"UNSEALED: true_ideology for {len(truth)} candidates "
          f"(read once, this script only)")
    # ---------------------------------------------------

    tiers = sorted(p.stem[-1] for p in (HERE / "intermediate").glob("emb_tier?.npz"))
    rows = []

    # ---- axes ----
    axis_results = {}
    for tier in tiers:
        pz = np.load(HERE / "outputs" / f"pca_tier{tier}.npz")
        k = int(pz["partisan_axis"])
        s_cen = orient_axis(pz["P"][:, k], party)
        cz = np.load(HERE / "outputs" / f"corrected_tier{tier}.npz")
        s_cor = np.asarray(cz["blind_axis_score"])
        for space, s, kk in [("centered", s_cen, k),
                             ("corrected", s_cor, int(cz["partisan_axis"]))]:
            r = axis_recovery(s, truth)
            primary = (tier == "B" and space == "corrected")
            rows.append({
                "measure": f"Tier {tier} ({NAMES[tier]}) {space} partisan "
                           f"axis PC{kk+1} vs true_ideology (r)",
                "kind": "axis", "tier": tier, "space": space,
                "value": r["pearson_r"], "spearman": r["spearman_rho"],
                "primary": primary})
            axis_results[(tier, space)] = (s, r["pearson_r"])
            tag = "  << PRIMARY" if primary else ""
            print(f"tier {tier} {space}: r={r['pearson_r']:+.4f} "
                  f"rho={r['spearman_rho']:+.4f}{tag}")

    # ---- distances ----
    diag = pd.read_csv(HERE / "outputs" / "blind_diagnostics.csv")
    for _, d in diag.iterrows():
        D = np.load(HERE / "outputs" /
                    f"D_tier{d['tier']}_{d['variant']}_{d['rep']}.npy")
        dv = distance_validity(D, truth)
        rows.append({
            "measure": f"Tier {d['tier']} {d['variant']} {d['rep']} "
                       f"distance validity",
            "kind": "distance", "tier": d["tier"], "space": d["variant"],
            "rep": d["rep"], "value": dv, "ratio": d["ratio"],
            "primary": False})
        print(f"tier {d['tier']} {d['variant']}/{d['rep']}: dv={dv:+.4f} "
              f"ratio={d['ratio']:.3f}")

    res = pd.DataFrame(rows)

    # ---- frozen references ----
    bv = pd.read_csv(WS0 / "baselines" / "baseline_validation.csv")
    frozen = {r["measure"]: r["ws0_value"] for _, r in bv.iterrows()}
    TFIDF_AXIS = frozen["TF-IDF partisan axis vs true_ideology (r)"]
    TFIDF_DV = frozen["TF-IDF distance validity  [NEW at WS0.4]"]
    W2V_DV = frozen["w2v corrected distance validity"]

    # ---- preregistration §5 decision rule ----
    primary_r = float(res.loc[res["primary"], "value"].iloc[0])
    dist_rows = res[res["kind"] == "distance"]
    distr = dist_rows[dist_rows["rep"].isin(["energy", "mmd"])]
    centro = dist_rows[dist_rows["rep"] == "centroid_cosine"]
    best_distr = distr.loc[distr["value"].idxmax()]
    best_centro = centro.loc[centro["value"].idxmax()]
    axis_wins = primary_r > TFIDF_AXIS
    dist_wins = (not axis_wins and
                 best_distr["value"] > max(best_centro["value"], TFIDF_DV))
    if axis_wins:
        outcome = ("ST axis beats TF-IDF: contextual embeddings become the "
                   "primary instrument for the real-data pipeline.")
    elif dist_wins:
        outcome = ("Axis does not beat TF-IDF but distributional distances "
                   "win: adopt ST as distance instrument, keep TF-IDF as "
                   "axis instrument.")
    else:
        outcome = ("Negative result: on short, topically-planted political "
                   "text, lexical baselines remain sufficient; Model2Vec "
                   "becomes the real-data default on cost grounds.")
    decision = {
        "primary_axis_r": primary_r, "tfidf_axis_r": TFIDF_AXIS,
        "axis_beats_tfidf": bool(axis_wins),
        "best_distributional": {"which": f"tier {best_distr['tier']} "
                                f"{best_distr['space']}/{best_distr['rep']}",
                                "dv": float(best_distr["value"])},
        "best_centroid": {"which": f"tier {best_centro['tier']} "
                          f"{best_centro['space']}/{best_centro['rep']}",
                          "dv": float(best_centro["value"])},
        "tfidf_distance_validity": TFIDF_DV,
        "w2v_corrected_distance_validity": W2V_DV,
        "distributional_wins_distances": bool(dist_wins),
        "outcome": outcome,
    }
    with open(HERE / "outputs" / "decision.json", "w") as fh:
        json.dump(decision, fh, indent=2)
    print("\nDECISION:", json.dumps(decision, indent=2))

    # append frozen references for the combined table
    for m, v in [("TF-IDF PC1 (frozen baseline) vs true_ideology (r)", TFIDF_AXIS),
                 ("word2vec partisan axis (frozen baseline) (r)",
                  frozen["word2vec partisan axis vs true_ideology (r)"]),
                 ("TF-IDF distance validity (frozen baseline)", TFIDF_DV),
                 ("w2v corrected distance validity (frozen baseline)", W2V_DV)]:
        rows.append({"measure": m, "kind": "frozen", "value": v,
                     "primary": False})
    pd.DataFrame(rows).to_csv(HERE / "outputs" / "validation_results.csv",
                              index=False)

    # ---- figure 3: distance validity panorama ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"A": "#4269d0", "B": "#efb118", "C": "#ff725c"}
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=150)
    dd = dist_rows.copy()
    dd["label"] = ("T" + dd["tier"] + " " + dd["space"] + "\n" +
                   dd["rep"].str.replace("centroid_cosine", "centroid"))
    dd = dd.sort_values(["tier", "rep", "space"]).reset_index(drop=True)
    ax.bar(range(len(dd)), dd["value"],
           color=[colors[t] for t in dd["tier"]], width=0.7)
    ax.axhline(TFIDF_DV, color="#1c2733", ls="--", lw=1.2,
               label=f"TF-IDF baseline ({TFIDF_DV:.3f})")
    ax.axhline(W2V_DV, color="#9498a0", ls=":", lw=1.2,
               label=f"corrected w2v ({W2V_DV:.3f})")
    ax.set_xticks(range(len(dd)), dd["label"], fontsize=7, rotation=45,
                  ha="right")
    ax.set_ylabel("corr(distance, |true ideology gap|)")
    ax.set_title("WS1 — distance validity by tier / variant / representation")
    ax.set_ylim(0, 0.72)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "fig3_distance_validity.png",
                bbox_inches="tight")

    # ---- figure 4: primary axis scatter ----
    s, r = axis_results[("B", "corrected")]
    fig, axes = plt.subplots(1, len(tiers), figsize=(4.4 * len(tiers), 4.2),
                             dpi=150, squeeze=False)
    pcol = {"D": "#4269d0", "R": "#ff725c", "I": "#3ca951"}
    for axx, tier in zip(axes[0], tiers):
        sc, rr = axis_results[(tier, "corrected")]
        z = (sc - sc.mean()) / sc.std()
        for p in ["D", "R", "I"]:
            m = party == p
            axx.scatter(truth[m], z[m], s=9, alpha=0.55, c=pcol[p], label=p)
        axx.set(xlabel="true_ideology", ylabel="blind axis score (z)",
                title=f"Tier {tier} corrected axis  r={rr:+.3f}")
        axx.spines[["top", "right"]].set_visible(False)
        axx.grid(alpha=0.25, lw=0.5)
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"WS1 axis recovery at unseal (TF-IDF bar: r={TFIDF_AXIS:.3f})",
                 y=1.03)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "fig4_axis_scatter.png",
                bbox_inches="tight")
    print("wrote outputs/validation_results.csv, decision.json, figs 3–4")


if __name__ == "__main__":
    main()
