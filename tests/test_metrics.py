"""Unit tests for ws0-harness/metrics.py.

These formalize (and extend) the module's inline self-tests as individual
pytest cases, so CI exercises every metric the workstreams score against.
"""

import numpy as np
import pytest

from metrics import (
    ari_nmi,
    axis_recovery,
    distance_validity,
    identify_partisan_axis,
    mantel_test,
    npmi_coherence,
    orient_axis,
    procrustes_similarity,
    project_out,
    topic_diversity,
    upper_tri,
    within_between_ratio,
)

RNG = np.random.default_rng(20260725)


@pytest.fixture()
def planted_party():
    """100 candidates, 50 D / 50 R, with a 1-D score cleanly separated by
    party, and the pairwise distance matrix built from it."""
    party = np.array(["D"] * 50 + ["R"] * 50)
    x = np.concatenate([RNG.normal(-1, 0.2, 50), RNG.normal(1, 0.2, 50)])
    D = np.abs(x[:, None] - x[None, :])
    return party, x, D


class TestHelpers:
    def test_upper_tri_shape_and_values(self):
        D = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
        np.testing.assert_array_equal(upper_tri(D), [1, 2, 3])

    def test_project_out_removes_direction(self):
        x = RNG.normal(size=100)
        X = np.outer(x, np.ones(5)) + RNG.normal(0, 0.01, (100, 5))
        Xp = project_out(X - X.mean(0), np.ones(5))
        assert np.abs(Xp @ np.ones(5)).max() < 1e-8

    def test_project_out_is_idempotent(self):
        X = RNG.normal(size=(50, 4))
        v = RNG.normal(size=4)
        once = project_out(X, v)
        twice = project_out(once, v)
        np.testing.assert_allclose(once, twice, atol=1e-12)

    def test_identify_partisan_axis_finds_planted_axis(self, planted_party):
        party, x, _ = planted_party
        P = np.column_stack([RNG.normal(0, 1, 100), x])
        assert identify_partisan_axis(P, party) == 1

    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    def test_identify_partisan_axis_rejects_degenerate_pcs(self, planted_party):
        party, _, _ = planted_party
        P = np.zeros((100, 3))  # all PCs degenerate -> NaN correlations
        with pytest.raises(ValueError, match="no PC correlates"):
            identify_partisan_axis(P, party)

    def test_orient_axis_enforces_r_above_d(self, planted_party):
        party, x, _ = planted_party
        s = orient_axis(-x, party)
        assert s[party == "R"].mean() > s[party == "D"].mean()
        # Already-oriented input is untouched.
        np.testing.assert_array_equal(orient_axis(x, party), x)


class TestAxisAndDistance:
    def test_axis_recovery_perfect(self):
        t = RNG.uniform(-1, 1, 500)
        out = axis_recovery(t, t)
        assert out["pearson_r"] == pytest.approx(1.0)
        assert out["spearman_rho"] == pytest.approx(1.0)

    def test_axis_recovery_anticorrelated(self):
        t = RNG.uniform(-1, 1, 500)
        assert axis_recovery(-t, t)["pearson_r"] == pytest.approx(-1.0)

    def test_distance_validity_perfect_when_planted(self):
        t = RNG.uniform(-1, 1, 100)
        D = np.abs(t[:, None] - t[None, :])
        assert distance_validity(D, t) == pytest.approx(1.0)

    def test_within_between_ratio_separates_planted_parties(self, planted_party):
        party, _, D = planted_party
        wb = within_between_ratio(D, party)
        assert wb["ratio"] > 3
        assert wb["between"] > wb["within"]

    def test_within_between_ratio_ignores_independents(self, planted_party):
        party, x, _ = planted_party
        # Add independents at an extreme position: if they leaked into the
        # D/R computation the ratio would move.
        party_i = np.concatenate([party, ["I"] * 10])
        x_i = np.concatenate([x, np.full(10, 25.0)])
        D_i = np.abs(x_i[:, None] - x_i[None, :])
        base = within_between_ratio(np.abs(x[:, None] - x[None, :]), party)
        with_i = within_between_ratio(D_i, party_i)
        assert with_i["ratio"] == pytest.approx(base["ratio"])


class TestTopicMetrics:
    def test_ari_nmi_identical_labels(self):
        labels = RNG.integers(0, 8, 2000)
        out = ari_nmi(labels, labels)
        assert out["ari"] == 1.0
        assert out["nmi"] == pytest.approx(1.0)

    def test_ari_nmi_shuffled_labels_near_zero(self):
        labels = RNG.integers(0, 8, 2000)
        assert abs(ari_nmi(RNG.permutation(labels), labels)["ari"]) < 0.05

    def test_npmi_coherence_orders_good_above_bad(self):
        docs = [
            ["apple", "banana", "cherry"] if i % 2 == 0 else ["dog", "wolf", "fox"]
            for i in range(400)
        ]
        good = npmi_coherence([["apple", "banana"], ["dog", "wolf"]], docs)
        bad = npmi_coherence([["apple", "dog"], ["banana", "fox"]], docs)
        assert good["npmi_mean"] > 0.9 > bad["npmi_mean"]

    def test_npmi_coherence_rejects_window_variant(self):
        with pytest.raises(NotImplementedError):
            npmi_coherence([["a", "b"]], [["a", "b"]], window=10)

    def test_topic_diversity_bounds(self):
        assert topic_diversity([["a", "b"], ["c", "d"]], topn=2) == 1.0
        assert topic_diversity([["a", "b"], ["a", "b"]], topn=2) == 0.5


class TestMatrixAgreement:
    def test_mantel_identical_matrices(self, planted_party):
        _, _, D = planted_party
        out = mantel_test(D, D, permutations=199)
        assert out["mantel_r"] > 0.999
        assert out["p_value"] < 0.01

    def test_mantel_unrelated_matrices(self, planted_party):
        _, _, D = planted_party
        Dr = np.abs(RNG.normal(size=(100, 100)))
        Dr = (Dr + Dr.T) / 2
        np.fill_diagonal(Dr, 0)
        assert abs(mantel_test(D, Dr, permutations=199)["mantel_r"]) < 0.1

    def test_mantel_deterministic_under_seed(self, planted_party):
        _, _, D = planted_party
        Dr = np.abs(RNG.normal(size=(100, 100)))
        Dr = (Dr + Dr.T) / 2
        np.fill_diagonal(Dr, 0)
        a = mantel_test(D, Dr, permutations=99, seed=7)
        b = mantel_test(D, Dr, permutations=99, seed=7)
        assert a == b

    def test_procrustes_invariant_to_rotation_and_scale(self):
        A = RNG.normal(size=(60, 2))
        theta = 0.7
        R = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        out = procrustes_similarity(A, 3.0 * A @ R + 5.0)
        assert out["disparity"] < 1e-12
        assert out["similarity"] == pytest.approx(1.0)
