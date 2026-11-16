"""
Small, hand-rolled statistics helpers for module 8's own analyses - kept
here rather than in the shared top-level utils/ since they're specific to
this module's one-off Largaespada analyses, not generic infrastructure.

moderated_t_test() and fit_shared_prior() implement a simplified,
limma/Smyth(2004)-style empirical-Bayes variance shrinkage: with sample
sizes as small as n=1 per group, an ordinary t-test's per-feature variance
is undefined or wildly unstable, so each feature's variance is shrunk
toward a prior estimated from many other features tested at the same time.
Unlike limma's own eBayes fit (which estimates the prior's precision, d0,
via a trigamma-based method-of-moments fit), this uses a small FIXED d0:
with only a handful of independent replicate groups available in this
8-sample dataset, a precisely-fit d0 isn't meaningfully more justified than
a sensible fixed one, and it avoids hand-rolling a numerical trigamma
inversion with no prior art in this repo and no local reference
implementation to validate it against.

fit_shared_prior() DOES correct for one other bias: the per-feature
variance estimates it pools are each built from only 1-2 degrees of
freedom, and a low-df chi-square-scaled variance estimate has a median
well below its mean - pooling raw medians from many such estimates
silently underestimates the shared variance and inflates every resulting
t-statistic. See fit_shared_prior()'s own docstring for the (closed-form,
not fitted) correction.
"""

import numpy as np
import pandas as pd
from scipy import stats


def pivot_score_matrix(df, id_col, score_col, sample_ids):
    """
    Pivot a long per-sample scores table (columns including ModelID,
    id_col, score_col) to a (feature x sample) matrix, with samples in the
    given fixed order. Drops any feature not present for every one of
    sample_ids - a no-op if df was already filtered to complete features,
    but cheap insurance against a caller passing an incomplete subset.
    """
    matrix = df[df["ModelID"].isin(sample_ids)].pivot(index=id_col, columns="ModelID", values=score_col)
    return matrix.reindex(columns=sample_ids).dropna()


def collect_replicate_groups(comparisons):
    """
    From the comparisons derive_comparisons() returns, collects the
    distinct (n>=2) sample groups - the "replicate" groups with enough
    members to estimate a per-feature empirical variance from directly -
    deduplicating a group that's reused as the n>=2 side of more than one
    comparison (counting it once, not once per comparison it appears in).
    """
    seen = set()
    groups = []
    for comparison in comparisons:
        for group in (comparison["pos_ids"], comparison["neg_ids"]):
            if len(group) >= 2:
                key = frozenset(group)
                if key not in seen:
                    seen.add(key)
                    groups.append(group)
    return groups


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg FDR-adjusted q-values for an array of p-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(q_sorted, 0, 1)
    return q


def pooled_variance(group1, group2):
    """
    Pooled within-group variance and degrees of freedom for two groups,
    one value per feature (group1/group2 are 2D arrays, features x samples,
    or 1D for a single feature). A group of size 1 contributes 0 to both
    the sum of squares and the degrees of freedom (consistent with "no
    within-group information from a single observation"), rather than the
    NaN pandas/numpy .var() would otherwise produce.
    """
    group1 = np.atleast_2d(group1)
    group2 = np.atleast_2d(group2)
    n1, n2 = group1.shape[-1], group2.shape[-1]

    def sum_of_squares(group, n):
        if n < 2:
            return np.zeros(group.shape[0])
        return np.nan_to_num(np.var(group, axis=-1, ddof=1)) * (n - 1)

    df = (n1 - 1) + (n2 - 1)
    ss = sum_of_squares(group1, n1) + sum_of_squares(group2, n2)
    s_sq = ss / df if df > 0 else np.zeros(group1.shape[0])
    return s_sq, df


def fit_shared_prior(variance_arrays, dfs, d0=4):
    """
    Fit a shared empirical-Bayes prior (d0, s0_sq) for the moderated t-test,
    from a list of per-feature variance arrays (one array per independent
    n>=2 replicate group, with dfs[i] the degrees of freedom - group size
    minus 1 - that array's variances were each computed with) and each
    array's matching degrees of freedom in dfs.

    s0_sq is the median of all pooled variances across every array and
    feature, after a bias correction: a ddof=1 variance estimate with df
    degrees of freedom is distributed as (true_variance / df) * chi2(df),
    and chi2 is right-skewed, so its MEDIAN sits below its MEAN (e.g.
    ~0.455x the true variance at df=1, via chi2.ppf(0.5, 1)) - taking a
    raw median of many such low-df estimates therefore systematically
    underestimates the shared variance, which silently inflates every
    downstream moderated t-statistic (the shrunk variance in its
    denominator is too small). Recovering true_variance from median(S^2)
    needs median(S^2) * df / chi2.ppf(0.5, df) - at df=1 that's a no-op
    beyond the divide (df=1), which is why the divide-only version of this
    passed its own df=1-only tests but was wrong for any df>1: it was
    silently estimating half the true variance at df=2, for example. A
    closed-form correction, not a distributional fit, so it doesn't
    require the trigamma-based machinery this module's docstring explains
    was deliberately avoided for d0.
    """
    corrected = [np.ravel(v) * df / stats.chi2.ppf(0.5, df) for v, df in zip(variance_arrays, dfs, strict=True)]
    all_variances = np.concatenate(corrected)
    s0_sq = np.median(all_variances)
    return d0, s0_sq


def moderated_t_test(group1, group2, prior_d0, prior_s0_sq):
    """
    Two-sample moderated t-test for one or more features at once.
    group1, group2: features x samples (2D), or samples only (1D, single feature).
    prior_d0, prior_s0_sq: the shared empirical-Bayes prior from fit_shared_prior().

    Returns (t_statistic, df, p_value), each a scalar if the inputs were 1D,
    else an array of the same length as the number of features. When a
    feature's own pooled df is 0 (both groups have n=1), the shrunken
    variance collapses entirely to the prior s0_sq, rather than producing
    NaN - this is the point of the shrinkage, not an edge case to special-case
    away.

    A feature with truly zero shrunk variance (prior_s0_sq itself is 0
    AND this feature's own pooled variance is 0) gets t=0, not NaN/inf:
    with literally no variance evidence anywhere to stand on, there's
    nothing to divide mean_diff by, and reporting it as "no evidence of a
    difference" is the conservative, defensible reading - not "infinitely
    significant" (if mean_diff != 0) or "undefined" (if mean_diff == 0,
    which a plain 0/0 division would otherwise silently turn into a NaN
    that corrupts OTHER features' q-values too, since
    benjamini_hochberg's sort-based implementation doesn't isolate NaN to
    the row that produced it).
    """
    single_feature = np.ndim(group1) == 1
    group1 = np.atleast_2d(group1)
    group2 = np.atleast_2d(group2)
    n1, n2 = group1.shape[-1], group2.shape[-1]

    s_sq, df_i = pooled_variance(group1, group2)
    s_shrunk_sq = (prior_d0 * prior_s0_sq + df_i * s_sq) / (prior_d0 + df_i)

    mean_diff = np.mean(group1, axis=-1) - np.mean(group2, axis=-1)
    standard_error = np.sqrt(s_shrunk_sq * (1 / n1 + 1 / n2))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_statistic = np.where(standard_error > 0, mean_diff / standard_error, 0.0)
    moderated_df = prior_d0 + df_i
    p_value = 2 * stats.t.sf(np.abs(t_statistic), df=moderated_df)

    if single_feature:
        return t_statistic[0], moderated_df, p_value[0]
    return t_statistic, np.full_like(t_statistic, moderated_df, dtype=float), p_value


def derive_comparisons(metadata_df):
    """
    Given a metadata dataframe with ModelID, SUZ12, NF1, type ("pos"/"neg"
    mutation status, cell-type stratum), returns the list of testable
    mutation-status comparisons: one dict per (stratum, marker) pair that
    has at least one sample on each side, as
    {"stratum": ..., "marker": ..., "pos_ids": [...], "neg_ids": [...]}.

    A (stratum, marker) pair is skipped (not returned) if the stratum has
    no variation in that marker (e.g. all samples neg). Within a stratum,
    if two markers produce the identical pos/neg split, only the first one
    (by column order) is returned, labeled with both marker names joined
    by "/" - testing the same split twice under two different names would
    double-count one piece of evidence as two.
    """
    markers = ["SUZ12", "NF1"]
    comparisons = []
    for stratum, stratum_df in metadata_df.groupby("type", sort=False):
        seen_splits = {}
        for marker in markers:
            pos_ids = tuple(sorted(stratum_df.loc[stratum_df[marker] == "pos", "ModelID"]))
            neg_ids = tuple(sorted(stratum_df.loc[stratum_df[marker] == "neg", "ModelID"]))
            if not pos_ids or not neg_ids:
                continue
            split_key = (pos_ids, neg_ids)
            if split_key in seen_splits:
                comparisons[seen_splits[split_key]]["marker"] += f"/{marker}"
                continue
            seen_splits[split_key] = len(comparisons)
            comparisons.append({
                "stratum": stratum,
                "marker": marker,
                "pos_ids": list(pos_ids),
                "neg_ids": list(neg_ids),
            })
    return comparisons


def differential_test(scores_df, id_col, score_col, comparisons, replicate_groups, d0_values=(2, 4)):
    """
    Runs the moderated t-test for every comparison against one feature
    family (pathways, or drugs), fitting one shared prior per family from
    replicate_groups and reusing it across all comparisons - see this
    module's docstring and derive_comparisons() for why. Returns a long
    dataframe with one row per (comparison, feature, d0): stratum, marker,
    feature, n_pos, n_neg, mean_pos, mean_neg, mean_diff, and
    t/p/q/df columns suffixed by each d0 in d0_values.
    """
    variance_arrays = [
        np.var(pivot_score_matrix(scores_df, id_col, score_col, group).to_numpy(), axis=1, ddof=1)
        for group in replicate_groups
    ]
    variance_dfs = [len(group) - 1 for group in replicate_groups]

    rows = []
    for comparison in comparisons:
        pos_matrix = pivot_score_matrix(scores_df, id_col, score_col, comparison["pos_ids"])
        neg_matrix = pivot_score_matrix(scores_df, id_col, score_col, comparison["neg_ids"])
        shared_features = pos_matrix.index.intersection(neg_matrix.index)
        pos_values = pos_matrix.loc[shared_features].to_numpy()
        neg_values = neg_matrix.loc[shared_features].to_numpy()
        mean_diff = pos_values.mean(axis=1) - neg_values.mean(axis=1)

        comparison_rows = {
            "stratum": comparison["stratum"], "marker": comparison["marker"],
            "feature": shared_features, "n_pos": len(comparison["pos_ids"]), "n_neg": len(comparison["neg_ids"]),
            "mean_pos": pos_values.mean(axis=1), "mean_neg": neg_values.mean(axis=1), "mean_diff": mean_diff,
        }
        for d0 in d0_values:
            prior_d0, prior_s0_sq = fit_shared_prior(variance_arrays, variance_dfs, d0=d0)
            t_stat, df, p_value = moderated_t_test(pos_values, neg_values, prior_d0, prior_s0_sq)
            q_value = benjamini_hochberg(p_value)
            comparison_rows[f"t_d0_{d0}"] = t_stat
            comparison_rows[f"p_d0_{d0}"] = p_value
            comparison_rows[f"q_d0_{d0}"] = q_value
            comparison_rows[f"df_d0_{d0}"] = df
        rows.append(pd.DataFrame(comparison_rows))
    return pd.concat(rows, ignore_index=True)
