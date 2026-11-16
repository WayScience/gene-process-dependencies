"""
A single, consolidated pinwheel-style view comparing all 8 Largaespada
samples at once, colored by cell type and marked by SUZ12 status - built
from scratch for this exploratory comparison rather than reusing the
shared utils/pinwheels.py plot_pinwheel(), which draws one sample's own
wedge-filled pinwheel at a time and isn't set up to overlay several
samples on shared angular positions. That function stays untouched for
its existing single-sample callers in modules 4 and 6.

This is an exploration tool, not a publication figure: legibility and a
single combined output file matter more than polish.

score_col values are signed (utils/latent_annotation.py's
compute_and_plot_latent_scores no longer takes abs()): a feature a sample
sits strongly on the "opposite" side of is just as notable as one it
sits strongly on the "same" side of, so both top_feature_union and
plot_overlay_pinwheel below select/plot by magnitude, not raw value.
"""

import numpy as np

TYPE_COLORS = {
    "tumor_derived": "#d62728",
    "immortalized_human_schwann_cell": "#1f77b4",
    "immortalized_peripheral_nerve": "#2ca02c",
}


def top_feature_union(scores_df, metadata_df, id_col, score_col, top_n_per_sample=10, max_features=30):
    """
    The shared set of angular positions for the overlay plot: the union of
    each sample's own top top_n_per_sample features (by score_col), capped
    at max_features by each feature's own max-across-samples score if the
    union would otherwise be larger.
    """
    model_ids = metadata_df["ModelID"]
    candidates = scores_df[scores_df["ModelID"].isin(model_ids)].copy()
    candidates["_abs_score"] = candidates[score_col].abs()
    per_sample_top = (
        candidates
        .sort_values("_abs_score", ascending=False)
        .groupby("ModelID")
        .head(top_n_per_sample)
    )
    feature_union = per_sample_top[id_col].unique()
    if len(feature_union) <= max_features:
        return list(feature_union)

    max_abs_score_by_feature = (
        candidates[candidates[id_col].isin(feature_union)]
        .groupby(id_col)["_abs_score"]
        .max()
        .sort_values(ascending=False)
    )
    return list(max_abs_score_by_feature.head(max_features).index)


def plot_overlay_pinwheel(scores_df, metadata_df, id_col, score_col, feature_subset, title, ax):
    """
    Draws one polar overlay plot onto ax: one line per sample in
    metadata_df, at fixed angular positions given by feature_subset
    (shared across every sample so they're directly comparable), colored
    by that sample's type and marker-styled by SUZ12 status. Takes
    dataframes directly - no file I/O - so it's callable from a test with
    a tiny synthetic frame.

    scores_df: long per-sample scores table with columns [ModelID, id_col, score_col].
    metadata_df: columns [ModelID, SUZ12, NF1, type].
    """
    if len(feature_subset) == 0:
        # e.g. finite-score filtering upstream removed every candidate
        # feature for this family - nothing to plot, but don't crash the
        # rest of the run (angles[0] below would IndexError on an empty
        # feature_subset).
        ax.text(0.5, 0.5, "No features available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, y=1.12)
        return

    angles = np.linspace(0, 2 * np.pi, len(feature_subset), endpoint=False)
    closed_angles = np.append(angles, angles[0])

    for _, sample in metadata_df.iterrows():
        sample_scores = scores_df[scores_df["ModelID"] == sample["ModelID"]].set_index(id_col)[score_col]
        # abs(): score_col is signed now, and a polar radius can't be
        # negative - magnitude is what's plotted, same reasoning as
        # utils/pinwheels.py's plot_pinwheel.
        radii = sample_scores.reindex(feature_subset).fillna(0.0).to_numpy()
        radii = np.abs(radii)
        closed_radii = np.append(radii, radii[0])

        color = TYPE_COLORS.get(sample["type"], "#7f7f7f")
        marker_style = dict(marker="o", markersize=5) if sample["SUZ12"] == "pos" else dict(marker="o", markersize=5, markerfacecolor="white")
        label = f"{sample['ModelID']} ({sample['type']}, SUZ12{sample['SUZ12']}/NF1{sample['NF1']})"
        ax.plot(closed_angles, closed_radii, color=color, linewidth=1.2, alpha=0.8, label=label, **marker_style)

    ax.set_xticks(angles)
    ax.set_xticklabels(feature_subset, fontsize=6)
    ax.set_title(title, y=1.12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=7)
