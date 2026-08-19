"""
02_freeze_baselines.py — WS0.4 (blind part)
-------------------------------------------
Regenerates the 2026-07-20 baseline instruments into ws0/baselines/ so all
three workstreams compare against identical arrays.

NOTE ON PROVENANCE: the plan's step 0.4 says "copy, no recomputation", but
the 07-20 session committed only summary CSVs — the .npz score arrays and
distance matrices were never written to the project folder. This script
therefore REGENERATES them with the original recipe and seed (20260720),
byte-for-byte identical tokenization/params from
embeddings-pca-analysis/scripts/01_prepare_corpus.py + 02_embeddings.py.
Two deliberate changes for reproducibility, recorded in the manifest:
  * word2vec workers=1 (original used 4; multithreaded training is not
    deterministic, and the original arrays are lost — the arrays produced
    here become the CANONICAL frozen baselines going forward)
  * explicit deterministic hashfxn (crc32) so results don't depend on
    PYTHONHASHSEED
04_verify_harness.py checks the regenerated instruments against the frozen
07-20 validation numbers (validation_results.csv) before anything is frozen
for downstream use.

Everything here is BLIND in the sense that no design decision touches truth:
instruments are built from blind_corpus.parquet only, and partisan axes are
identified from the observable party label (D/R separation), exactly as in
the 07-20 step 3, and oriented so that R > D on the axis score. One
non-blind file IS read: the pilot's published unseal output
(analyses/00-embeddings-pca/outputs/validation_results.csv, five aggregate
r values) is copied verbatim into baselines/frozen_validation_20260720.csv
as the frozen comparison table. No per-candidate truth enters this script.

Outputs (ws0/baselines/):
  candidate_vectors.npz   candidate_ids, X_w2v, X_tfidf  (910 x 100 each)
  pca_scores.npz          P_w2v, P_tfidf (top 10 PCs), partisan axis indices
  axis_scores.csv         per-candidate blind axis scores (both instruments)
  D_w2v_raw.npy, D_w2v_corrected.npy, D_tfidf.npy   (910x910, float32)
  frozen_validation_20260720.csv   copy of the 07-20 unseal results
  candidate_metadata.csv  observables per candidate (from blind corpus)
  manifest.json           params, hashes, provenance notes
"""

import hashlib
import json
import re
import zlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

HERE = Path(__file__).resolve().parent
BASE = HERE / "baselines"
BASE.mkdir(exist_ok=True)

DIM = 100
SEED = 20260720  # original 07-20 seed — this is a reconstruction, not a new run
FROZEN_VALIDATION = (
    HERE.parent
    / "analyses"
    / "00-embeddings-pca"
    / "outputs"
    / "validation_results.csv"
)

# --- tokenization identical to 07-20 01_prepare_corpus.py ---
URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
TOKEN_RE = re.compile(r"[a-z][a-z']+")


def tokenize(text: str) -> list[str]:
    t = text.lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = t.replace("#", " ")
    return TOKEN_RE.findall(t)


def det_hash(s: str) -> int:
    """Deterministic gensim hashfxn (independent of PYTHONHASHSEED)."""
    return zlib.crc32(s.encode("utf-8"))


def identify_partisan_axis(P: np.ndarray, party: np.ndarray) -> int:
    """Blind axis identification: PC most separating D vs R (observable)."""
    dr = np.isin(party, ["D", "R"])
    y = (party[dr] == "R").astype(float)
    corrs = [abs(np.corrcoef(P[dr, k], y)[0, 1]) for k in range(P.shape[1])]
    return int(np.argmax(corrs))


def orient(score: np.ndarray, party: np.ndarray) -> np.ndarray:
    """Sign convention: R mean > D mean."""
    if score[party == "R"].mean() < score[party == "D"].mean():
        return -score
    return score


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    blind = pd.read_parquet(HERE / "blind_corpus.parquet")
    print(f"Blind corpus: {len(blind):,} rows")

    # ---- candidate metadata (observables only) ----
    meta = (
        blind.groupby("candidate_id")
        .agg(
            candidate_name=("candidate_name", "first"),
            handle=("handle", "first"),
            party=("party", "first"),
            chamber=("chamber", "first"),
            state=("state", "first"),
            district=("district", "first"),
            incumbent=("incumbent", "first"),
            n_tweets=("tweet_id", "count"),
            share_retweets=("is_retweet", "mean"),
        )
        .reset_index()
        .sort_values("candidate_id")
        .reset_index(drop=True)
    )
    cand_order = meta["candidate_id"].tolist()
    party = meta["party"].to_numpy()
    meta.to_csv(BASE / "candidate_metadata.csv", index=False)

    # ---- tokenize (retweets included as candidate speech) ----
    tokens = blind["text"].astype(str).map(tokenize)
    sentences = tokens.tolist()

    # ---- word2vec, 07-20 recipe ----
    print("Training word2vec (skip-gram, workers=1 for determinism)…")
    w2v = Word2Vec(
        sentences=sentences, vector_size=DIM, window=5, min_count=5,
        sg=1, workers=1, epochs=10, seed=SEED, hashfxn=det_hash,
    )
    print(f"  vocab retained: {len(w2v.wv)}")

    cid_arr = blind["candidate_id"].to_numpy()
    sums = {c: np.zeros(DIM) for c in cand_order}
    counts = {c: 0 for c in cand_order}
    for cid, toks in zip(cid_arr, sentences):
        vecs = [w2v.wv[t] for t in toks if t in w2v.wv]
        if vecs:
            sums[cid] += np.sum(vecs, axis=0)
            counts[cid] += len(vecs)
    X_w2v = np.vstack([sums[c] / max(counts[c], 1) for c in cand_order])

    # ---- TF-IDF + SVD, 07-20 recipe ----
    print("Building TF-IDF + SVD…")
    docs = {c: [] for c in cand_order}
    for cid, toks in zip(cid_arr, sentences):
        docs[cid].extend(toks)
    doc_strings = [" ".join(docs[c]) for c in cand_order]
    tfidf = TfidfVectorizer(min_df=5, sublinear_tf=True)
    X_sparse = tfidf.fit_transform(doc_strings)
    svd = TruncatedSVD(n_components=DIM, random_state=SEED)
    X_tfidf = svd.fit_transform(X_sparse)

    # ---- PCA + blind partisan-axis identification (07-20 step 3) ----
    P_w2v = PCA(n_components=10, random_state=SEED).fit_transform(
        X_w2v - X_w2v.mean(axis=0))
    P_tfidf = PCA(n_components=10, random_state=SEED).fit_transform(
        X_tfidf - X_tfidf.mean(axis=0))
    k_w2v = identify_partisan_axis(P_w2v, party)
    k_tfidf = identify_partisan_axis(P_tfidf, party)
    print(f"  partisan axis: w2v PC{k_w2v+1}, tfidf PC{k_tfidf+1}")

    score_w2v = orient(P_w2v[:, k_w2v], party)
    score_tfidf = orient(P_tfidf[:, k_tfidf], party)

    # ---- distances: raw, style-corrected (project out w2v PC1), tfidf ----
    D_w2v_raw = cosine_distances(X_w2v)
    Xc = X_w2v - X_w2v.mean(axis=0, keepdims=True)
    v1 = PCA(n_components=1, random_state=SEED).fit(Xc).components_[0]
    X_corr = Xc - np.outer(Xc @ v1, v1)
    D_w2v_corr = cosine_distances(X_corr)
    D_tfidf = cosine_distances(X_tfidf)

    # ---- persist ----
    np.savez_compressed(BASE / "candidate_vectors.npz",
                        candidate_ids=np.array(cand_order),
                        X_w2v=X_w2v, X_tfidf=X_tfidf)
    np.savez_compressed(BASE / "pca_scores.npz",
                        candidate_ids=np.array(cand_order),
                        P_w2v=P_w2v, P_tfidf=P_tfidf,
                        partisan_axis_w2v=k_w2v, partisan_axis_tfidf=k_tfidf)
    pd.DataFrame({
        "candidate_id": cand_order,
        "tfidf_partisan_score": score_tfidf,
        "w2v_partisan_score": score_w2v,
        "w2v_pc1_style_score": P_w2v[:, 0],
    }).to_csv(BASE / "axis_scores.csv", index=False)
    np.save(BASE / "D_w2v_raw.npy", D_w2v_raw.astype(np.float32))
    np.save(BASE / "D_w2v_corrected.npy", D_w2v_corr.astype(np.float32))
    np.save(BASE / "D_tfidf.npy", D_tfidf.astype(np.float32))
    pd.read_csv(FROZEN_VALIDATION).to_csv(
        BASE / "frozen_validation_20260720.csv", index=False)

    files = ["candidate_vectors.npz", "pca_scores.npz", "axis_scores.csv",
             "D_w2v_raw.npy", "D_w2v_corrected.npy", "D_tfidf.npy",
             "frozen_validation_20260720.csv", "candidate_metadata.csv"]
    manifest = {
        "frozen_on": date.today().isoformat(),
        "provenance": (
            "Regenerated from blind_corpus.parquet with the 2026-07-20 recipe "
            "(seed 20260720, identical tokenization and hyperparameters). "
            "Original arrays were never committed; these arrays are now the "
            "canonical frozen baselines. Changes vs 07-20 run: workers=1 and "
            "deterministic hashfxn (crc32) for exact reproducibility. "
            "Verified against frozen 07-20 validation numbers in "
            "04_verify_harness.py (see baselines/baseline_validation.csv)."
        ),
        "w2v_params": {"vector_size": DIM, "window": 5, "min_count": 5,
                       "sg": 1, "workers": 1, "epochs": 10, "seed": SEED,
                       "hashfxn": "zlib.crc32"},
        "tfidf_params": {"min_df": 5, "sublinear_tf": True,
                         "svd_components": DIM, "svd_seed": SEED},
        "partisan_axis": {"w2v": f"PC{k_w2v+1}", "tfidf": f"PC{k_tfidf+1}",
                          "identified_by": "max |corr| with observable D/R "
                                           "label; oriented so R > D"},
        "sha256": {f: sha256(BASE / f) for f in files},
    }
    with open(BASE / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("Baselines frozen to ws0/baselines/ with manifest.")


if __name__ == "__main__":
    main()
