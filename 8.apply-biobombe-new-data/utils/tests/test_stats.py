"""
Tests for utils/stats.py. Plain assert-based (not pytest - not a declared
dependency in this project's environment.yml), runnable directly:
    python3 utils/tests/test_stats.py
or individually picked up by pytest if it happens to be installed.
"""

import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from stats import (
    benjamini_hochberg,
    collect_replicate_groups,
    derive_comparisons,
    differential_test,
    empty_differential_result,
    fit_shared_prior,
    moderated_t_test,
    pivot_score_matrix,
    pooled_variance,
)


def test_benjamini_hochberg_hand_computed():
    p = np.array([0.01, 0.02, 0.03, 0.5])
    q = benjamini_hochberg(p)
    expected = np.array([0.04, 0.04, 0.04, 0.5])
    assert np.allclose(q, expected), f"expected {expected}, got {q}"


def test_moderated_t_test_matches_unshrunk_at_large_n():
    # With n1=n2=30 (df_i=58) and a small prior d0=4, the shrunk result
    # should be close to an ordinary pooled-variance t-test - shrinkage
    # should barely move the answer when there's plenty of empirical data.
    rng = np.random.RandomState(0)
    group1 = rng.normal(loc=5.0, scale=2.0, size=30)
    group2 = rng.normal(loc=4.0, scale=2.0, size=30)

    t_shrunk, df_shrunk, p_shrunk = moderated_t_test(group1, group2, prior_d0=4, prior_s0_sq=4.0)
    t_plain, p_plain = stats.ttest_ind(group1, group2)

    assert abs(t_shrunk - t_plain) < 0.1, f"moderated t ({t_shrunk}) too far from unshrunk t ({t_plain})"
    assert abs(p_shrunk - p_plain) < 0.05, f"moderated p ({p_shrunk}) too far from unshrunk p ({p_plain})"


def test_moderated_t_test_handles_n1_vs_n1():
    # Both groups n=1: df_i=0, zero empirical variance information. Must
    # rely entirely on the prior and return a finite, sensible result -
    # not NaN, not a crash.
    group1 = np.array([5.0])
    group2 = np.array([3.0])
    t, df, p = moderated_t_test(group1, group2, prior_d0=4, prior_s0_sq=1.0)

    assert np.isfinite(t) and np.isfinite(p), f"expected finite t/p, got t={t}, p={p}"
    assert df == 4, f"expected df to equal prior_d0 (4) when df_i=0, got {df}"
    # s_shrunk_sq should equal prior_s0_sq exactly when df_i=0
    expected_se = np.sqrt(1.0 * (1 / 1 + 1 / 1))
    expected_t = (5.0 - 3.0) / expected_se
    assert np.isclose(t, expected_t), f"expected t={expected_t}, got {t}"
    assert 0 < p < 1


def test_moderated_t_test_vectorized_matches_scalar():
    # Running on a 2D (features x samples) input should give the same
    # per-feature result as calling the scalar/1D version in a loop.
    rng = np.random.RandomState(1)
    group1 = rng.normal(size=(5, 3))
    group2 = rng.normal(size=(5, 2))

    t_vec, df_vec, p_vec = moderated_t_test(group1, group2, prior_d0=4, prior_s0_sq=1.0)
    for i in range(5):
        t_i, df_i, p_i = moderated_t_test(group1[i], group2[i], prior_d0=4, prior_s0_sq=1.0)
        assert np.isclose(t_vec[i], t_i)
        assert np.isclose(p_vec[i], p_i)


def test_moderated_t_test_zero_variance_gives_zero_not_nan():
    # Both groups identical (zero pooled variance) AND a zero shared prior
    # (s0_sq=0): with no variance evidence anywhere, t must be 0 (no
    # evidence of a difference), not NaN from a literal 0/0 division.
    group1 = np.array([2.0, 2.0])
    group2 = np.array([2.0, 2.0])
    t, df, p = moderated_t_test(group1, group2, prior_d0=4, prior_s0_sq=0.0)
    assert t == 0.0, f"expected t=0.0 for zero variance and zero mean_diff, got {t}"
    assert np.isfinite(p), f"expected a finite p-value, got {p}"


def test_moderated_t_test_zero_variance_does_not_corrupt_other_features_bh():
    # A batch with one zero-variance/zero-diff feature (feature 0) and two
    # well-separated, clearly-significant features (1, 2). Before the fix,
    # feature 0's NaN t/p-value would propagate through
    # benjamini_hochberg's sort-based reverse-cummin into every other
    # feature's q-value, not just its own.
    # Features 1/2 need real (nonzero) within-group variance of their own,
    # not just a big mean_diff - otherwise their own pooled variance is
    # also exactly 0 and they'd trivially get t=0 too, defeating the point.
    group1 = np.array([[2.0, 2.0], [100.0, 105.0], [100.0, 105.0]])
    group2 = np.array([[2.0, 2.0], [0.0, 5.0], [0.0, 5.0]])
    t, df, p = moderated_t_test(group1, group2, prior_d0=4, prior_s0_sq=0.0)
    assert np.all(np.isfinite(p)), f"expected every p-value finite, got {p}"
    q = benjamini_hochberg(p)
    assert np.all(np.isfinite(q)), f"a NaN p-value must not propagate into other features' q-values, got {q}"
    assert q[1] < 0.05 and q[2] < 0.05, f"the two clearly-significant features should still show up as such, got {q}"


def test_fit_shared_prior_pools_median():
    # Both arrays at df=1, so the bias correction factor is identical for
    # both and the relative pooling behaves like a plain median of the
    # corrected values.
    variance_arrays = [np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0, 100.0])]
    d0, s0_sq = fit_shared_prior(variance_arrays, dfs=[1, 1], d0=4)
    correction = 1 * stats.chi2.ppf(0.5, 1)
    expected_median = np.median([1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 100.0]) / correction
    assert d0 == 4
    assert np.isclose(s0_sq, expected_median), f"expected {expected_median}, got {s0_sq}"


def test_fit_shared_prior_corrects_low_df_bias():
    # A df=1 variance array systematically underestimates the true
    # variance at its raw median (chi2_1's median is ~0.455, not 1, times
    # the true variance) - fit_shared_prior must correct for this, not
    # just pass the raw median through.
    rng = np.random.RandomState(2)
    true_variance = 10.0
    # Many independent df=1 (chi2_1-scaled) variance draws at the same true variance.
    draws = true_variance * rng.chisquare(df=1, size=5000)
    _, s0_sq = fit_shared_prior([draws], dfs=[1], d0=4)
    assert abs(s0_sq - true_variance) / true_variance < 0.1, (
        f"expected corrected s0_sq near the true variance ({true_variance}), got {s0_sq}"
    )
    raw_median = np.median(draws)
    assert s0_sq > raw_median, "corrected estimate should be larger than the raw (biased-low) median"


def test_fit_shared_prior_corrects_low_df_bias_at_df_2():
    # Regression test for a real bug: the correction factor needs
    # median(S^2) * df / chi2.ppf(0.5, df), not just / chi2.ppf(0.5, df).
    # At df=1 that missing "* df" factor is 1 and invisible (the test
    # above can't catch it) - at df=2 it isn't, and the uncorrected
    # version would silently estimate half the true variance.
    rng = np.random.RandomState(3)
    true_variance = 10.0
    df = 2
    # A pooled sample variance S^2 with df degrees of freedom is
    # distributed as (true_variance / df) * chi2(df), not true_variance *
    # chi2(df) - the division by df here matters for the simulation to be
    # realistic (confirmed empirically before relying on it: the buggy,
    # pre-fix formula recovers ~5.0, almost exactly half of 10.0, on this
    # same simulation).
    draws = (true_variance / df) * rng.chisquare(df=df, size=5000)
    _, s0_sq = fit_shared_prior([draws], dfs=[df], d0=4)
    assert abs(s0_sq - true_variance) / true_variance < 0.1, (
        f"expected corrected s0_sq near the true variance ({true_variance}), got {s0_sq} "
        "(got ~half the true variance? the '* df' factor is probably missing again)"
    )


def test_pooled_variance_zero_for_single_observation_groups():
    s_sq, df = pooled_variance(np.array([5.0]), np.array([3.0]))
    assert df == 0
    assert np.allclose(s_sq, 0.0)


def test_derive_comparisons_on_real_metadata():
    metadata_df = pd.DataFrame({
        "ModelID": ["S462", "ST88-14", "STS-26T", "iHSC1L_C3", "iHSC1L_F12", "iPN1L", "iPN2J", "iPN3D"],
        "SUZ12": ["neg", "neg", "pos", "pos", "neg", "pos", "pos", "neg"],
        "NF1": ["neg", "neg", "pos", "neg", "neg", "pos", "neg", "neg"],
        "type": [
            "tumor_derived", "tumor_derived", "tumor_derived",
            "immortalized_human_schwann_cell", "immortalized_human_schwann_cell",
            "immortalized_peripheral_nerve", "immortalized_peripheral_nerve", "immortalized_peripheral_nerve",
        ],
    })
    comparisons = derive_comparisons(metadata_df)
    assert len(comparisons) == 4, f"expected 4 testable comparisons, got {len(comparisons)}: {comparisons}"

    by_stratum = {(c["stratum"], c["marker"]): c for c in comparisons}

    tumor = by_stratum[("tumor_derived", "SUZ12/NF1")]
    assert tumor["pos_ids"] == ["STS-26T"]
    assert tumor["neg_ids"] == ["S462", "ST88-14"]

    schwann = by_stratum[("immortalized_human_schwann_cell", "SUZ12")]
    assert schwann["pos_ids"] == ["iHSC1L_C3"]
    assert schwann["neg_ids"] == ["iHSC1L_F12"]
    assert ("immortalized_human_schwann_cell", "NF1") not in by_stratum, "schwann x NF1 has no variation and must be skipped"

    pn_suz12 = by_stratum[("immortalized_peripheral_nerve", "SUZ12")]
    assert pn_suz12["pos_ids"] == ["iPN1L", "iPN2J"]
    assert pn_suz12["neg_ids"] == ["iPN3D"]

    pn_nf1 = by_stratum[("immortalized_peripheral_nerve", "NF1")]
    assert pn_nf1["pos_ids"] == ["iPN1L"]
    assert pn_nf1["neg_ids"] == ["iPN2J", "iPN3D"]


def test_pivot_score_matrix_fixed_sample_order():
    df = pd.DataFrame({
        "ModelID": ["A", "B", "A", "B"],
        "feature": ["f1", "f1", "f2", "f2"],
        "score": [1.0, 2.0, 3.0, 4.0],
    })
    matrix = pivot_score_matrix(df, "feature", "score", ["B", "A"])
    assert list(matrix.columns) == ["B", "A"]
    assert matrix.loc["f1", "A"] == 1.0
    assert matrix.loc["f1", "B"] == 2.0


def test_pivot_score_matrix_drops_incomplete_features():
    df = pd.DataFrame({
        "ModelID": ["A", "B", "A"],
        "feature": ["f1", "f1", "f2"],  # f2 missing for B
        "score": [1.0, 2.0, 3.0],
    })
    matrix = pivot_score_matrix(df, "feature", "score", ["A", "B"])
    assert list(matrix.index) == ["f1"], f"expected only f1 (complete for both samples), got {list(matrix.index)}"


def test_collect_replicate_groups_dedups_shared_group():
    comparisons = [
        {"stratum": "s1", "marker": "SUZ12/NF1", "pos_ids": ["X"], "neg_ids": ["Y", "Z"]},
        {"stratum": "s2", "marker": "SUZ12", "pos_ids": ["Y", "Z"], "neg_ids": ["W"]},  # same neg group reused
        {"stratum": "s3", "marker": "NF1", "pos_ids": ["P", "Q"], "neg_ids": ["R"]},
    ]
    groups = collect_replicate_groups(comparisons)
    assert len(groups) == 2, f"expected 2 distinct n>=2 groups (Y/Z deduped), got {len(groups)}: {groups}"
    group_sets = [set(g) for g in groups]
    assert {"Y", "Z"} in group_sets
    assert {"P", "Q"} in group_sets


def test_differential_test_matches_moderated_t_test_directly():
    # A tiny synthetic family of 2 features, 1 comparison (n=1 vs n=2),
    # 1 replicate group of n=2 to fit the prior from. differential_test's
    # output should match calling fit_shared_prior/moderated_t_test by hand.
    scores_df = pd.DataFrame({
        "ModelID": ["X", "Y", "Z", "X", "Y", "Z"],
        "feature": ["f1", "f1", "f1", "f2", "f2", "f2"],
        "score": [5.0, 1.0, 2.0, 3.0, 4.0, 4.5],
    })
    comparisons = [{"stratum": "s1", "marker": "SUZ12", "pos_ids": ["X"], "neg_ids": ["Y", "Z"]}]
    replicate_groups = [["Y", "Z"]]

    result_df = differential_test(scores_df, "feature", "score", comparisons, replicate_groups, d0_values=(4,))
    assert list(result_df["feature"]) == ["f1", "f2"]
    assert list(result_df["n_pos"]) == [1, 1]
    assert list(result_df["n_neg"]) == [2, 2]

    pos_values = np.array([[5.0], [3.0]])
    neg_values = np.array([[1.0, 2.0], [4.0, 4.5]])
    variance_arrays = [np.var(np.array([[1.0, 2.0], [4.0, 4.5]]), axis=1, ddof=1)]
    prior_d0, prior_s0_sq = fit_shared_prior(variance_arrays, dfs=[1], d0=4)
    expected_t, expected_df, expected_p = moderated_t_test(pos_values, neg_values, prior_d0, prior_s0_sq)

    assert np.allclose(result_df["t_d0_4"], expected_t)
    assert np.allclose(result_df["p_d0_4"], expected_p)
    assert np.allclose(result_df["df_d0_4"], expected_df)
    assert np.allclose(result_df["mean_diff"], pos_values.mean(axis=1) - neg_values.mean(axis=1))


def test_empty_differential_result_matches_differential_test_columns():
    scores_df = pd.DataFrame({
        "ModelID": ["a", "b", "c", "d"],
        "name": ["drug1"] * 4,
        "pathway_score": [1.0, 2.0, 3.0, 4.0],
    })
    comparisons = [{"stratum": "s", "marker": "m", "pos_ids": ["a", "b"], "neg_ids": ["c", "d"]}]
    replicate_groups = [["a", "b"], ["c", "d"]]
    real_result = differential_test(scores_df, "name", "pathway_score", comparisons, replicate_groups)

    empty_result = empty_differential_result()
    assert list(empty_result.columns) == list(real_result.columns)
    assert len(empty_result) == 0


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failures = []
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as e:
            failures.append(test.__name__)
            print(f"FAIL {test.__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    run_all()
