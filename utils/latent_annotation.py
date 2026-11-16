"""
Shared, top-level pathway/drug-to-latent-dimension annotation core: given
an ensemble-wide GSEA/drug-correlation candidate table, decide which
pathway/drug gets permanently pinned to which single (model, dim, init)
latent slot, and score a new sample against those slots. Used by module 8
(Largaespada analysis) and, nominally, by modules 4 and 6's own pinwheel
scripts (both currently broken for unrelated, pre-existing reasons as of
this writing, so not live consumers today).

Split out of utils/pinwheels.py, which now holds only plot_pinwheel - the
statistical/assignment logic here had outgrown a file named after one
specific plot type, with zero test coverage to match (see
utils/tests/test_latent_annotation.py).
"""

import numpy as np
import pandas as pd

from pinwheels import plot_pinwheel


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


def filter_significant_candidates(df, p_col="p_value", q_threshold=0.05, extra_mask=None, effect_col=None, min_abs_effect=None):
    """
    Filters a candidate table (every tested pathway/drug x signature pair,
    not a pre-filtered subset) to the ones significant after Benjamini-
    Hochberg correction, replacing a bare nominal p-value cutoff.

    extra_mask (optional): a boolean mask applied BEFORE correction - it
    defines the family of tests being corrected together (e.g. excluding
    shuffled/negative-control rows, which aren't real tests), not a
    post-hoc filter. Correcting an already-pre-filtered-by-p-value subset
    would answer the wrong question, so this function always expects the
    full candidate pool (minus any such family-defining exclusion) as
    input, never one already thresholded by p_col.

    effect_col/min_abs_effect (optional): an additional filter on
    |df[effect_col]| applied AFTER the BH correction - a plain effect-size
    gate (e.g. drugs' existing |pearson_correlation| > 0.15), a separate
    concern from multiple-testing correction, not folded into the q-value.

    Adds a "q_value" column to the returned rows.
    """
    candidates = df[extra_mask] if extra_mask is not None else df
    result = candidates.copy()
    result["q_value"] = benjamini_hochberg(result[p_col].to_numpy())
    result = result[result["q_value"] <= q_threshold]
    if effect_col is not None and min_abs_effect is not None:
        result = result[result[effect_col].abs() >= min_abs_effect]
    return result


def compute_and_plot_latent_scores(model_id, latent_df, comp_df, id, score, name, make_plot=True, metadata_cols=(), extra_cols=()):
    """
    Computes pathway scores based on latent dimensions and generates a pinwheel plot.

    Parameters:
        model_id (str): Identifier for the model.
        latent_df (DataFrame): DataFrame containing latent dimension scores.
        comp_df (DataFrame): DataFrame containing pathway or drug comparison scores.
        id (str): Column name for pathway or drug identifiers.
        score (str): Column name for the score used in calculations.
        name (str): Type of plot - e.g. pathway or drug.
        make_plot (bool): Whether to generate and save a pinwheel plot. Default True,
            matching prior behavior; pass False to compute scores without plotting.
        metadata_cols (iterable of str): any extra, non-latent columns latent_df
            carries beyond the standard ModelID/model/latent_dim_total/init/seed
            columns (e.g. module 8 attaches whatever per-sample metadata came
            with a new dataset). Must be passed explicitly by the caller, who
            knows what it attached - a metadata column can be numeric (so a
            dtype check doesn't exclude it) and isn't guaranteed to avoid any
            particular name pattern (so a label-pattern check doesn't either).
        extra_cols (iterable of str): any columns on comp_df beyond the
            standard id/latent_dim_total/init/model/z/score set that should
            also survive into the returned table (e.g. assign_unique_latent_dims's
            n_candidates/assignment_score/shared_slot_with) - without this,
            they're silently dropped by the column selection below.
    """
    # Filter by ModelID
    latent_filtered = latent_df[latent_df["ModelID"] == model_id].copy()

    # Melt latent dataframe. value_vars is restricted to the actual latent
    # columns, identified as "every column that isn't an id_var or a
    # caller-declared metadata column" - not by dtype or name pattern, since
    # a metadata column can be numeric and/or digit-named just as validly as
    # a real latent column (this is what broke before: an exclude list that
    # didn't know about module 8's SUZ12/NF1/type columns, then a dtype/regex
    # check that couldn't tell a numeric metadata column from a real one).
    id_vars = ["ModelID", "model", "latent_dim_total", "init", "seed"]
    exclude = set(id_vars) | set(metadata_cols)
    z_columns = [c for c in latent_filtered.columns if c not in exclude]
    latent_long = latent_filtered.melt(
        id_vars=id_vars,
        value_vars=z_columns,
        var_name="z",  # Latent dimension names
        value_name="latent_score"
    )

    # Convert z from string to int. This assumes every latent column is
    # named "z_<int>" (PCA/ICA/NMF) or plain "<int>" (VAE variants) - the
    # only two conventions any model type in this project currently uses.
    latent_long["z"] = latent_long["z"].astype(str).str.replace(r"^z_", "", regex=True).astype(int)

    # Prepare GSEA/drug dataframe
    comp_long = comp_df.rename(columns={"full_model_z": "latent_dim_total"}).copy()
    comp_long = comp_long[[id, "latent_dim_total", "init", "model", "z", score] + list(extra_cols)]
    comp_long["z"] = comp_long["z"].astype(str).str.replace(r"^z_", "", regex=True).astype(int)

    # Merge latent and GSEA data while keeping only one "model" and "init"
    merged_df = latent_long.merge(
        comp_long,
        on=["z", "model", "latent_dim_total", "init"],
        how="inner"
    )

    # Compute final pathway scores. Using merged_df[score] here, not comp_long[score]:
    # merged_df is the result of an inner join and has its own fresh index, which
    # doesn't line up with comp_long's original index or length. Multiplying against
    # comp_long[score] directly silently misaligns rows (and can introduce NaNs) rather
    # than raising, so it must use the copy of the column that actually survived the merge.
    #
    # Signed, not abs()'d: the sign now carries real information - positive
    # means this sample sits on the side of the latent axis that originally
    # drove the association, negative means the opposite side. Collapsing
    # that to magnitude-only threw away information the new sample's own
    # latent_score actually carries. Callers that need a non-negative value
    # (e.g. a pinwheel plot's radius) take abs()/clip at that point instead.
    merged_df["pathway_score"] = merged_df["latent_score"] * merged_df[score]
    merged_df = merged_df.dropna(subset=["pathway_score"])

    print(model_id)

    if make_plot:
        plot_pinwheel(model_id, merged_df, id, name)

    return merged_df


def _top_k_median(scores, ascending, k=3):
    """
    The median of a group's top-k most extreme scores (k or fewer if the
    group has fewer candidates than k) - "most extreme" meaning largest
    when ascending=False, smallest (most negative) when ascending=True,
    matching assign_unique_latent_dims's own is_drug convention. Used to
    rank a pathway/drug for assignment by more than just its single best
    hit, so one lucky/unlucky draw among many candidates doesn't look
    identical to a pathway where several independent hits agree.
    """
    ordered = sorted(scores, reverse=not ascending)
    top_k = ordered[: min(k, len(ordered))]
    return float(np.median(top_k))


def assign_unique_latent_dims(df, score_col, target_col, latent_col="z"):
    """
    Assigns each pathway/drug to the best available latent dimension
    without replacement, ranking pathways/drugs by a robustness-adjusted
    score (the median of their top-3 surviving candidates) rather than
    their single best hit - a pathway/drug with only one surviving
    candidate anywhere in the ensemble is not dropped (see n_candidates
    below), just ranked lower than one several independent hits agree on.

    Parameters:
    - df (pd.DataFrame): DataFrame containing pathways/drugs and their corresponding latent dimensions.
    - score_col (str): Column name for the enrichment score (e.g., "gsea_es_score", "drug_es_score").
    - target_col (str): Column name for the pathway or drug.
    - latent_col (str): Column name for the latent dimension (default: "z").

    Returns:
    - pd.DataFrame: one assigned row per pathway/drug, with two added
      columns: n_candidates (how many surviving candidates it had,
      ensemble-wide, before this assignment step - a corroboration
      count, not gated on) and assignment_score (the robustness score
      it was ranked by).
    """
    # Check to see if score is pearson correlation or not, so the ordering is the correct direction. GSEA results are non-directional, pearson correlation is
    if score_col != "pearson_correlation":
        df[score_col] = df[score_col].abs()
        is_drug = False
    else:
        is_drug = True

    # Sort so that the most important values are first (highest abs for GSEA, most negative for pearson correlation)
    df_sorted = df.sort_values(score_col, ascending=is_drug).copy()

    # Track used latent dimensions and their associated pathways
    used_latents = {}
    assigned_rows = []

    # Group by pathway or drug to process each one sequentially
    grouped = df_sorted.groupby(target_col)

    sorted_groups = sorted(
        grouped,
        key=lambda group: _top_k_median(group[1][score_col].tolist(), ascending=is_drug),
        reverse=True,
    )

    for pathway, group in sorted_groups:
        print(f"Processing pathway: {pathway}")

        # Filter out rows with already used latent dimensions
        available_latents = group[~group.apply(
            lambda r: (r[latent_col], r['model'], r['full_model_z'], r['init']) in used_latents.keys(), axis=1)]

        if available_latents.empty:
            print(f"No unused latent dimensions left for pathway {pathway}.")
            continue  # Skip to the next pathway

        # Sort by GSEA score (descending) to pick the top available latent dimension
        best_latent = available_latents.sort_values(by=score_col, ascending=is_drug).head(1).iloc[0].copy()
        best_latent["n_candidates"] = len(group)
        best_latent["assignment_score"] = _top_k_median(group[score_col].tolist(), ascending=is_drug)

        # Record the assignment
        latent_key = (best_latent[latent_col], best_latent['model'], best_latent['full_model_z'], best_latent['init'])
        used_latents[latent_key] = pathway  # Assign the latent dimension to the pathway
        assigned_rows.append(best_latent)  # Add to assigned rows

        print(f"Assigned latent dimension {best_latent[latent_col]} for model {best_latent['model']}, {best_latent['full_model_z']}, init {best_latent['init']} to pathway {pathway}.")

    # Convert list of assigned rows back to DataFrame
    return pd.DataFrame(assigned_rows)


def flag_shared_latent_slots(categories, latent_col="z"):
    """
    Cross-checks the independently-assigned results of several categories
    (e.g. CORUM, Reactome, drugs - each its own assign_unique_latent_dims()
    call) and flags where two categories landed on the identical
    (model, full_model_z, init, latent_col) slot - CORUM/Reactome/drugs
    are assigned independently by design (not one combined assignment),
    so this is possible and silent otherwise: two "independent" hits for
    a sample could be mechanically the same latent_score under two names.
    Purely diagnostic - never changes any assignment.

    Parameters:
    - categories: list of (label, assigned_df, target_col) tuples, one
      per category, each assigned_df the result of assign_unique_latent_dims.
    - latent_col (str): Column name for the latent dimension (default: "z").

    Returns:
    - list of dataframes, same order/shape as the input assigned_dfs, each
      with a new "shared_slot_with" column: a list of "label: name"
      strings for every other category sharing that row's exact slot
      (empty list if none).
    """
    def slot_key(row):
        return (row["model"], row["full_model_z"], row["init"], row[latent_col])

    slot_index = {}
    for label, assigned_df, target_col in categories:
        for _, row in assigned_df.iterrows():
            slot_index.setdefault(slot_key(row), []).append((label, row[target_col]))

    results = []
    for label, assigned_df, target_col in categories:
        shared_slot_with = []
        for _, row in assigned_df.iterrows():
            key = slot_key(row)
            others = [
                f"{other_label}: {other_name}"
                for other_label, other_name in slot_index[key]
                if not (other_label == label and other_name == row[target_col])
            ]
            shared_slot_with.append(others)
        out = assigned_df.copy()
        out["shared_slot_with"] = shared_slot_with
        results.append(out)
    return results
