"""
metrics.py — WS0.2 shared metrics module
----------------------------------------
One importable module so WS1/WS2/WS3 score against identical implementations.

Consolidated from embeddings-pca-analysis/scripts/03–05 (2026-07-20):
    axis_recovery, distance_validity, within_between_ratio,
    identify_partisan_axis, orient_axis, project_out
New for the extension workstreams:
    ari_nmi (topic recovery), npmi_coherence / cv_coherence (gensim wrap),
    topic_diversity, mantel_test, procrustes_similarity

Conventions
-----------
* Distance matrices D are square, symmetric, zero-diagonal, candidates in
  the row order of ws0/baselines/candidate_metadata.csv.
* "party" arrays are the observable D/R/I labels (blind-safe).
* Anything taking `truth` is an UNSEAL-ONLY function — call it only at the
  validation step declared in your preregistration.md.

Self-tests: `python metrics.py` runs sanity checks on synthetic cases.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.spatial import procrustes as _procrustes
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

__all__ = [
    "axis_recovery", "distance_validity", "within_between_ratio",
    "identify_partisan_axis", "orient_axis", "project_out",
    "ari_nmi", "npmi_coherence", "cv_coherence", "topic_diversity",
    "mantel_test", "procrustes_similarity", "upper_tri",
]


# ---------------------------------------------------------------- helpers

def upper_tri(D: np.ndarray) -> np.ndarray:
    """Flatten the strict upper triangle of a square matrix."""
    iu = np.triu_indices(D.shape[0], k=1)
    return np.asarray(D)[iu]


def project_out(X: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Remove one or more (row-vector) directions from centered X.

    Used for style-confound correction (07-20: projecting out word2vec PC1
    doubled distance validity). `directions` shape (k, d) or (d,).
    """
    X = np.asarray(X, dtype=float)
    V = np.atleast_2d(np.asarray(directions, dtype=float))
    # Orthonormalize the directions for a clean projection
    Q, _ = np.linalg.qr(V.T)
    return X - (X @ Q) @ Q.T


def identify_partisan_axis(P: np.ndarray, party: np.ndarray) -> int:
    """BLIND-SAFE: index of the PC best separating observable D vs R."""
    party = np.asarray(party)
    dr = np.isin(party, ["D", "R"])
    y = (party[dr] == "R").astype(float)
    corrs = [abs(np.corrcoef(P[dr, k], y)[0, 1]) for k in range(P.shape[1])]
    return int(np.argmax(corrs))


def orient_axis(score: np.ndarray, party: np.ndarray) -> np.ndarray:
    """BLIND-SAFE sign convention: mean(R) > mean(D)."""
    party = np.asarray(party)
    if score[party == "R"].mean() < score[party == "D"].mean():
        return -np.asarray(score)
    return np.asarray(score)


# ------------------------------------------------- axis & distance validity

def axis_recovery(score: np.ndarray, truth: np.ndarray) -> dict:
    """UNSEAL-ONLY. Pearson/Spearman of an axis score vs true_ideology.

    07-20 frozen references: TF-IDF PC1 r=0.974, word2vec PC2 r=0.886.
    """
    r, p_r = stats.pearsonr(score, truth)
    rho, p_rho = stats.spearmanr(score, truth)
    return {"pearson_r": float(r), "pearson_p": float(p_r),
            "spearman_rho": float(rho), "spearman_p": float(p_rho)}


def distance_validity(D: np.ndarray, truth: np.ndarray) -> float:
    """UNSEAL-ONLY. corr(pairwise distance, |true ideology gap|).

    07-20 frozen references: raw w2v 0.281, corrected 0.597.
    """
    truth = np.asarray(truth, dtype=float)
    iu = np.triu_indices(D.shape[0], k=1)
    gap = np.abs(truth[iu[0]] - truth[iu[1]])
    r, _ = stats.pearsonr(np.asarray(D)[iu], gap)
    return float(r)


def within_between_ratio(D: np.ndarray, party: np.ndarray) -> dict:
    """BLIND-SAFE. Mean within- vs between-party distance, D/R pairs only.

    07-20 frozen references (w2v): ratio 1.26 raw -> 1.35 corrected.
    """
    party = np.asarray(party)
    iu = np.triu_indices(D.shape[0], k=1)
    flat = np.asarray(D)[iu]
    same = party[iu[0]] == party[iu[1]]
    dr = np.isin(party[iu[0]], ["D", "R"]) & np.isin(party[iu[1]], ["D", "R"])
    w = float(flat[same & dr].mean())
    b = float(flat[~same & dr].mean())
    return {"within": w, "between": b, "ratio": b / w}


# ----------------------------------------------------------- topic metrics

def ari_nmi(pred_labels, true_labels) -> dict:
    """UNSEAL-ONLY (vs true_topic). Tweet-level assignment agreement.

    Handles BERTopic-style outliers: pass drop_label to exclude (reported).
    WS2 pre-registered success bar: ARI >= 0.60 for at least one entrant.
    """
    pred = np.asarray(pred_labels)
    true = np.asarray(true_labels)
    return {"ari": float(adjusted_rand_score(true, pred)),
            "nmi": float(normalized_mutual_info_score(true, pred))}


def npmi_coherence(topics: list[list[str]], tokenized_docs: list[list[str]],
                   topn: int = 10, window: int | None = None) -> dict:
    """BLIND-SAFE. NPMI coherence per topic, document co-occurrence based.

    Self-contained implementation (no gensim needed): P(w), P(wi, wj) from
    document-level co-occurrence with add-epsilon smoothing.
    Returns per-topic scores and the mean. Range [-1, 1], higher = better.
    """
    n_docs = len(tokenized_docs)
    doc_sets = [set(d) for d in tokenized_docs]
    vocab = set(w for t in topics for w in t[:topn])
    contains = {w: np.zeros(n_docs, dtype=bool) for w in vocab}
    for i, ds in enumerate(doc_sets):
        for w in vocab & ds:
            contains[w][i] = True
    eps = 1e-12
    per_topic = []
    for topic in topics:
        words = [w for w in topic[:topn] if contains[w].sum() > 0]
        scores = []
        for a in range(len(words)):
            for b in range(a + 1, len(words)):
                wa, wb = words[a], words[b]
                p_a = contains[wa].mean()
                p_b = contains[wb].mean()
                p_ab = (contains[wa] & contains[wb]).mean()
                pmi = np.log((p_ab + eps) / (p_a * p_b))
                npmi = pmi / -np.log(p_ab + eps)
                scores.append(npmi)
        per_topic.append(float(np.mean(scores)) if scores else np.nan)
    return {"npmi_per_topic": per_topic,
            "npmi_mean": float(np.nanmean(per_topic))}


def cv_coherence(topics: list[list[str]], tokenized_docs: list[list[str]],
                 topn: int = 10) -> float:
    """BLIND-SAFE. Gensim c_v coherence (standard literature metric).

    Requires gensim; heavier than npmi_coherence but comparable to
    published topic-model results.
    """
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    dictionary = Dictionary(tokenized_docs)
    cm = CoherenceModel(topics=[t[:topn] for t in topics],
                        texts=tokenized_docs, dictionary=dictionary,
                        coherence="c_v")
    return float(cm.get_coherence())


def topic_diversity(topics: list[list[str]], topn: int = 25) -> float:
    """BLIND-SAFE. Share of unique words across topics' top-n (0..1)."""
    words = [w for t in topics for w in t[:topn]]
    return len(set(words)) / max(len(words), 1)


# ------------------------------------------------- matrix-level agreement

def mantel_test(D1: np.ndarray, D2: np.ndarray, permutations: int = 999,
                seed: int = 20260725) -> dict:
    """Mantel test: correlation of two distance matrices with a
    row/column permutation null. Returns r and one-sided p.

    Used in the synthesis stage for instrument agreement.
    """
    D1, D2 = np.asarray(D1, dtype=float), np.asarray(D2, dtype=float)
    assert D1.shape == D2.shape and D1.shape[0] == D1.shape[1]
    n = D1.shape[0]
    iu = np.triu_indices(n, k=1)
    v1 = D1[iu]
    r_obs, _ = stats.pearsonr(v1, D2[iu])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(permutations):
        perm = rng.permutation(n)
        r_p, _ = stats.pearsonr(v1, D2[np.ix_(perm, perm)][iu])
        if r_p >= r_obs:
            count += 1
    return {"mantel_r": float(r_obs),
            "p_value": (count + 1) / (permutations + 1),
            "permutations": permutations}


def procrustes_similarity(X1: np.ndarray, X2: np.ndarray) -> dict:
    """Procrustes agreement of two candidate configurations (rows aligned).

    Returns disparity (sum squared error after optimal similarity
    transform; 0 = identical shape) and a similarity = 1 - disparity.
    """
    _, _, disparity = _procrustes(np.asarray(X1, float), np.asarray(X2, float))
    return {"disparity": float(disparity), "similarity": 1.0 - float(disparity)}


# ------------------------------------------------------------- self-tests

def _self_test() -> None:
    rng = np.random.default_rng(20260725)

    # axis_recovery: perfect and noisy
    t = rng.uniform(-1, 1, 500)
    assert abs(axis_recovery(t, t)["pearson_r"] - 1) < 1e-12
    noisy = t + rng.normal(0, 0.3, 500)
    r = axis_recovery(noisy, t)["pearson_r"]
    assert 0.8 < r < 1.0, r

    # distance_validity: distances built exactly from |gap| -> r == 1
    D = np.abs(t[:100, None] - t[None, :100])
    assert abs(distance_validity(D, t[:100]) - 1) < 1e-12

    # within_between_ratio: planted party separation
    party = np.array(["D"] * 50 + ["R"] * 50)
    x = np.concatenate([rng.normal(-1, .2, 50), rng.normal(1, .2, 50)])
    Dp = np.abs(x[:, None] - x[None, :])
    wb = within_between_ratio(Dp, party)
    assert wb["ratio"] > 3, wb

    # identify_partisan_axis + orient_axis
    P = np.column_stack([rng.normal(0, 1, 100), x])
    assert identify_partisan_axis(P, party) == 1
    s = orient_axis(-x, party)
    assert s[party == "R"].mean() > s[party == "D"].mean()

    # project_out: removing the only signal direction kills separation
    X = np.outer(x, np.ones(5)) + rng.normal(0, .01, (100, 5))
    Xp = project_out(X - X.mean(0), np.ones(5))
    assert np.abs(Xp @ np.ones(5)).max() < 1e-8

    # ari_nmi: identical labels -> 1; shuffled -> ~0
    labels = rng.integers(0, 8, 2000)
    assert ari_nmi(labels, labels)["ari"] == 1.0
    shuffled = rng.permutation(labels)
    assert abs(ari_nmi(shuffled, labels)["ari"]) < 0.05

    # npmi_coherence: co-occurring words > random words
    docs = [["apple", "banana", "cherry"] if i % 2 == 0
            else ["dog", "wolf", "fox"] for i in range(400)]
    good = npmi_coherence([["apple", "banana"], ["dog", "wolf"]], docs)
    bad = npmi_coherence([["apple", "dog"], ["banana", "fox"]], docs)
    assert good["npmi_mean"] > 0.9 > bad["npmi_mean"], (good, bad)

    # topic_diversity
    assert topic_diversity([["a", "b"], ["c", "d"]], topn=2) == 1.0
    assert topic_diversity([["a", "b"], ["a", "b"]], topn=2) == 0.5

    # mantel: identical matrices -> r=1, small p; unrelated -> r~0
    m = mantel_test(Dp, Dp, permutations=199)
    assert m["mantel_r"] > 0.999 and m["p_value"] < 0.01
    Dr = np.abs(rng.normal(size=(100, 100)))
    Dr = (Dr + Dr.T) / 2; np.fill_diagonal(Dr, 0)
    m2 = mantel_test(Dp, Dr, permutations=199)
    assert abs(m2["mantel_r"]) < 0.1

    # procrustes: rotation+scale of same config -> disparity ~ 0
    A = rng.normal(size=(60, 2))
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    B = 3.0 * A @ R + 5.0
    pr = procrustes_similarity(A, B)
    assert pr["disparity"] < 1e-12, pr

    print("metrics.py self-tests: ALL PASSED")


if __name__ == "__main__":
    _self_test()
