"""
03_confound_gate.py — WS1 E1.4: mandatory confound diagnosis + correction
-------------------------------------------------------------------------
BLIND. Preregistration §8:

  * covariates per candidate: share_retweets, log10(n_tweets), topic-mix
    entropy from a blind proxy (tweet-level k-means, k=15, seed 20260725,
    on TF-IDF+SVD tweet vectors built with the WS0 recipe/tokenizer)
  * per tier: Pearson r of each of the top-10 centroid PCs against each
    covariate; a PC with max |r| >= 0.6 is a style axis and is projected
    out of the CENTERED tweet vectors -> `corrected` space — unless it is
    the blind partisan PC and its |D/R corr| exceeds its max covariate |r|
  * recomputes corrected centroids + PCA + blind axis; writes the
    regression table and figure

Writes: outputs/topic_entropy_proxy.csv, outputs/confound_regressions.csv,
        outputs/corrected_tier{X}.npz (centroids, PCA, axis, style PCs),
        intermediate/emb_tier{X}_corrected.npz (tweet-level, for 04),
        figures/fig2_confound_gate.png
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
WS0 = HERE.parents[1] / "ws0-harness"
sys.path.insert(0, str(WS0))
from metrics import identify_partisan_axis, orient_axis, project_out  # noqa: E402

SEED = 20260725
STYLE_R = 0.6

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
TOKEN_RE = re.compile(r"[a-z][a-z']+")


def tokenize(text: str) -> list[str]:
    t = text.lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = t.replace("#", " ")
    return TOKEN_RE.findall(t)


def topic_entropy_proxy(blind: pd.DataFrame, cand_order: list[str]) -> pd.Series:
    """Blind per-candidate topic-mix entropy (preregistration §8)."""
    out = HERE / "outputs" / "topic_entropy_proxy.csv"
    if out.exists():
        df = pd.read_csv(out).set_index("candidate_id")
        return df.loc[cand_order, "topic_entropy"]
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = [" ".join(tokenize(t)) for t in blind["text"].astype(str)]
    Xs = TfidfVectorizer(min_df=5, sublinear_tf=True).fit_transform(docs)
    Xr = TruncatedSVD(n_components=100, random_state=SEED).fit_transform(Xs)
    labels = KMeans(n_clusters=15, random_state=SEED, n_init=3).fit_predict(Xr)
    ent = {}
    for cid, grp in pd.DataFrame(
            {"candidate_id": blind["candidate_id"], "lab": labels}
    ).groupby("candidate_id"):
        p = grp["lab"].value_counts(normalize=True).to_numpy()
        ent[cid] = float(-(p * np.log(p)).sum())
    s = pd.Series(ent).loc[cand_order]
    s.rename("topic_entropy").rename_axis("candidate_id").reset_index().to_csv(
        out, index=False)
    return s


def main() -> None:
    meta = pd.read_csv(WS0 / "baselines" / "candidate_metadata.csv")
    cand_order = meta["candidate_id"].tolist()
    party = meta["party"].to_numpy()
    blind = pd.read_parquet(WS0 / "blind_corpus.parquet")
    cid_index = {c: i for i, c in enumerate(cand_order)}

    cov = pd.DataFrame({
        "share_retweets": meta["share_retweets"].to_numpy(),
        "log10_n_tweets": np.log10(meta["n_tweets"].to_numpy()),
        "topic_entropy": topic_entropy_proxy(blind, cand_order).to_numpy(),
    })
    dr = np.isin(party, ["D", "R"])
    y_dr = (party[dr] == "R").astype(float)

    tiers = sorted(p.stem[-1] for p in (HERE / "intermediate").glob("emb_tier?.npz"))
    rows, figdata = [], {}
    for tier in tiers:
        pz = np.load(HERE / "outputs" / f"pca_tier{tier}.npz")
        P, comps, k = pz["P"], pz["components"], int(pz["partisan_axis"])
        R = np.zeros((10, len(cov.columns)))
        for j in range(10):
            for m, c in enumerate(cov.columns):
                R[j, m] = np.corrcoef(P[:, j], cov[c])[0, 1]
            rows.append({"tier": tier, "pc": j + 1,
                         **{c: R[j, m] for m, c in enumerate(cov.columns)},
                         "dr_abscorr": float(pz["pc_dr_abscorr"][j])})
        maxabs = np.abs(R).max(axis=1)
        style = [j for j in range(10) if maxabs[j] >= STYLE_R
                 and not (j == k and pz["pc_dr_abscorr"][j] > maxabs[j])]
        print(f"tier {tier}: style PCs {[j+1 for j in style]} "
              f"(max|r| {maxabs.round(2)}), partisan PC{k+1} exempt-check ok")
        figdata[tier] = (R, style, k, np.asarray(pz["pc_dr_abscorr"]))

        # ---- corrected space ----
        z = np.load(HERE / "intermediate" / f"emb_tier{tier}.npz",
                    allow_pickle=True)
        X = z["X"].astype(np.float32)
        codes = np.array([cid_index[c] for c in z["candidate_id"]])
        mu = X.mean(axis=0, keepdims=True)
        Xc = X - mu
        if style:
            Xcorr = project_out(Xc, comps[style]).astype(np.float32)
        else:
            Xcorr = Xc
        np.savez_compressed(
            HERE / "intermediate" / f"emb_tier{tier}_corrected.npz",
            X=Xcorr, candidate_id=z["candidate_id"],
            style_pcs=np.array([j + 1 for j in style]))

        S = np.zeros((len(cand_order), Xcorr.shape[1]))
        np.add.at(S, codes, Xcorr)
        Ccorr = S / np.bincount(codes, minlength=len(cand_order))[:, None]

        from sklearn.decomposition import PCA
        Pc = PCA(n_components=10, random_state=SEED).fit(Ccorr)
        sc = (Ccorr - Ccorr.mean(axis=0)) @ Pc.components_.T
        kc = identify_partisan_axis(sc, party)
        axis_c = orient_axis(sc[:, kc], party)
        dr_corr = abs(np.corrcoef(sc[dr, kc], y_dr)[0, 1])
        print(f"tier {tier}: corrected partisan axis PC{kc+1}, "
              f"blind |D/R corr| {dr_corr:.4f}")
        np.savez_compressed(
            HERE / "outputs" / f"corrected_tier{tier}.npz",
            candidate_ids=np.array(cand_order),
            C_corrected=Ccorr.astype(np.float32),
            P=sc.astype(np.float32), partisan_axis=kc,
            blind_axis_score=axis_c.astype(np.float32),
            dr_abscorr=dr_corr, style_pcs=np.array([j + 1 for j in style]))

    pd.DataFrame(rows).to_csv(HERE / "outputs" / "confound_regressions.csv",
                              index=False)

    # ---- figure 2: confound heatmaps ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {"A": "Model2Vec potion-8M", "B": "MiniLM-L6-v2",
             "C": "bge-small-en-v1.5"}
    fig, axes = plt.subplots(1, len(tiers), figsize=(4.6 * len(tiers), 4.4),
                             dpi=150, squeeze=False)
    labels = ["retweet share", "log10 volume", "topic entropy", "D/R label"]
    for ax, tier in zip(axes[0], tiers):
        R, style, k, pc_dr = figdata[tier]
        M = np.column_stack([R, pc_dr])
        im = ax.imshow(np.abs(M), vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(4), labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(10), [f"PC{j+1}" for j in range(10)], fontsize=8)
        for j in range(10):
            for m in range(4):
                ax.text(m, j, f"{abs(M[j, m]):.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(M[j, m]) > 0.6 else "#1c2733")
        for j in style:
            ax.add_patch(plt.Rectangle((-0.5, j - 0.5), 4, 1, fill=False,
                                       edgecolor="#ff725c", lw=2))
        ax.add_patch(plt.Rectangle((-0.5, k - 0.5), 4, 1, fill=False,
                                   edgecolor="#3ca951", lw=2))
        ax.set_title(f"Tier {tier} — {names[tier]}\n(red = style PC removed, "
                     f"green = partisan PC)", fontsize=9)
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="|Pearson r|")
    fig.suptitle("WS1 confound gate — top-10 PCs vs behavioral covariates",
                 y=1.02)
    fig.savefig(HERE / "figures" / "fig2_confound_gate.png",
                bbox_inches="tight")
    print("wrote outputs/confound_regressions.csv + figures/fig2_confound_gate.png")


if __name__ == "__main__":
    main()
