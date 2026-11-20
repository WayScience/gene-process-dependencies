#!/usr/bin/env python
# coding: utf-8

# ## Annotate the new data's latent scores with pathways and drugs
#
# Links each new sample's BioBombe latent scores to the same GSEA pathway
# results and drug correlations used everywhere else in this project, via
# `utils/latent_annotation.py`'s `compute_and_plot_latent_scores` /
# `assign_unique_latent_dims` / `filter_significant_candidates` /
# `flag_shared_latent_slots` — the shared pathway/drug-to-latent
# assignment core (split out of `utils/pinwheels.py`, which now holds
# only its plotting function). `4.drug-dependency/7.pinwheel_plots.py`
# and `6.collab-data/3.pinwheel_plots.py` nominally use the same
# functions but are both currently broken for unrelated, pre-existing
# reasons (checked directly - not touched here).
#
# Run this script with `8.apply-biobombe-new-data/` as the working directory.

# In[ ]:


import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "../utils/")
from latent_annotation import (
    assign_unique_latent_dims,
    compute_and_plot_latent_scores,
    filter_significant_candidates,
    flag_shared_latent_slots,
)

EXTRA_ASSIGNMENT_COLS = ["n_candidates", "assignment_score", "shared_slot_with"]


# In[ ]:


data_results_dir = pathlib.Path("data/largaespada/results")

combined_latent_df = pd.read_parquet(data_results_dir / "NF1_data_latent_scores.parquet")
print(f"Latent scores: {combined_latent_df.shape}")

# Sample metadata columns (e.g. SUZ12, NF1, type) that
# 4.apply_biobombe_ensemble.py attached to combined_latent_df and recorded
# explicitly, so compute_and_plot_latent_scores below can exclude them by
# name instead of guessing from dtype or a naming pattern. Required, not
# optional: an older NF1_data_latent_scores.parquet predating this sidecar
# would otherwise silently default to an empty metadata list, letting
# SUZ12/NF1/type leak into the latent melt and fail integer conversion.
metadata_cols_file = data_results_dir / "NF1_data_latent_scores_metadata_cols.json"
if not metadata_cols_file.exists():
    raise FileNotFoundError(
        f"{metadata_cols_file} not found - this is required to know which columns "
        "in NF1_data_latent_scores.parquet are sample metadata rather than latent "
        "dimensions. Re-run 4.apply_biobombe_ensemble.py (which writes it) before "
        "this script."
    )
metadata_cols = json.loads(metadata_cols_file.read_text())
print(f"Sample metadata columns to exclude from latent melting: {metadata_cols}")


# ### Pathway annotation: CORUM and Reactome, kept separate
#
# Both raw, per-latent-dimension GSEA hit tables are produced by
# `3.run-biobombe/2.interpret_biobombe_with_gsea.py`'s
# `GSEA_LIBRARIES` loop: CORUM at its original, unsuffixed
# `combined_z_matrix_gsea_results.parquet` (unchanged, for backward
# compatibility with every existing consumer of that path), Reactome
# (library `Reactome_Pathways_2024`) at the new, separate
# `combined_z_matrix_gsea_results_reactome.parquet`. Note that this
# library version's term names don't carry an `R-HSA-...` ID suffix
# (unlike the older `Reactome_2022` library, or the pre-existing
# `4.drug-dependency/results/all_reactome_results.parquet`) - still
# genuine Reactome pathways, just a newer-library naming convention.
#
# Both results use a column literally called "reactome_pathway" for the
# pathway name — that's `2.interpret_biobombe_with_gsea.py`'s schema
# regardless of which library was actually run, not a labeling mistake here.
#
# Candidates are filtered to Benjamini-Hochberg significant (q<0.05, not a
# bare nominal p<0.05) before `assign_unique_latent_dims` ranks and
# assigns them - see `utils/latent_annotation.py` for why a nominal
# cutoff over this many tests isn't enough, and why that function now
# ranks by a robustness-adjusted score rather than each pathway/drug's
# single best hit.

# In[ ]:


gsea_sources = {
    "CORUM": pathlib.Path("../3.run-biobombe/gsea_results/combined_z_matrix_gsea_results.parquet"),
    "Reactome": pathlib.Path("../3.run-biobombe/gsea_results/combined_z_matrix_gsea_results_reactome.parquet"),
}

pathway_max_by_library = {}
for library, gsea_path in gsea_sources.items():
    if not gsea_path.exists():
        print(f"{library}: {gsea_path} not found, skipping.")
        continue

    gsea_df = pd.read_parquet(gsea_path)
    # Drop the shuffled negative-control rows (not real tests) before BH-
    # correcting the rest. This file has no fdr column of its own (unlike
    # the single-VAE GSEA results elsewhere in the project).
    significant_gsea_df = filter_significant_candidates(gsea_df, extra_mask=~gsea_df["shuffled"])
    print(f"{library}: {significant_gsea_df.shape[0]} / {gsea_df.shape[0]} significant hits (BH q<0.05)")

    # nes_score (normalized enrichment score), not gsea_es_score: NES adjusts
    # for gene-set-size effects, and it's what the previously-run ensemble-wide
    # annotation (4.drug-dependency/results/all_corum_results.parquet /
    # all_reactome_results.parquet) actually used.
    pathway_max_by_library[library] = assign_unique_latent_dims(
        significant_gsea_df, score_col="nes_score", target_col="reactome_pathway"
    )


# ### Drug annotation
#
# This step needs `4.drug-dependency/results/combined_latent_drug_correlations.parquet`,
# the ensemble-wide drug correlation table `4.drug-dependency/nbconverted/5.drug_latent_dim_correlation.py`
# produces. That script depends on `saved_models/`, which only exists locally
# now — if it hasn't been run yet, drug annotation is skipped here rather than failing.
# Computed here, before scoring either family, so `flag_shared_latent_slots`
# below can cross-check CORUM/Reactome/drugs together.

# In[ ]:


drug_correlation_path = pathlib.Path("../4.drug-dependency/results/combined_latent_drug_correlations.parquet")

if drug_correlation_path.exists():
    drug_df = pd.read_parquet(drug_correlation_path)
    significant_drug_df = filter_significant_candidates(
        drug_df, extra_mask=~drug_df["shuffled"], effect_col="pearson_correlation", min_abs_effect=0.15,
    )
    drug_max = assign_unique_latent_dims(
        significant_drug_df, score_col="pearson_correlation", target_col="name"
    )
    print(f"Drug candidates assigned: {drug_max.shape[0]}")
else:
    drug_max = None
    print(
        f"{drug_correlation_path} not found — skipping drug annotation. "
        "Run 4.drug-dependency/nbconverted/5.drug_latent_dim_correlation.py first to enable it."
    )


# ### Cross-check for shared latent slots
#
# CORUM, Reactome, and drugs are each assigned independently above, so a
# pathway and a drug (say) can land on the identical (model, dim, init)
# slot without either annotation being "wrong" - see
# `utils/latent_annotation.py`'s `flag_shared_latent_slots` docstring.
# This doesn't change any assignment, just adds a `shared_slot_with`
# column so it's visible instead of silent.

# In[ ]:


assignment_categories = [
    (library, pathway_max, "reactome_pathway") for library, pathway_max in pathway_max_by_library.items()
]
if drug_max is not None:
    assignment_categories.append(("Drug", drug_max, "name"))

flagged_results = flag_shared_latent_slots(assignment_categories)
flagged_by_label = dict(zip((label for label, _, _ in assignment_categories), flagged_results))

n_shared = sum((flagged_by_label[label]["shared_slot_with"].apply(len) > 0).sum() for label in flagged_by_label)
print(f"Rows sharing a latent slot with another category: {n_shared}")


# In[ ]:


pathway_results = []
for library in pathway_max_by_library:
    pathway_max = flagged_by_label[library]
    for model_id in combined_latent_df["ModelID"].unique():
        df = compute_and_plot_latent_scores(
            model_id, combined_latent_df, pathway_max, "reactome_pathway", "nes_score",
            library, make_plot=False, metadata_cols=metadata_cols, extra_cols=EXTRA_ASSIGNMENT_COLS,
        )
        df["gene_set_library"] = library
        pathway_results.append(df)

pathway_results_df = pd.concat(pathway_results, ignore_index=True) if pathway_results else pd.DataFrame()
print(f"Pathway annotations (CORUM + Reactome combined): {pathway_results_df.shape}")


# In[ ]:


if drug_max is not None:
    drug_max_flagged = flagged_by_label["Drug"]
    drug_results = []
    for model_id in combined_latent_df["ModelID"].unique():
        df = compute_and_plot_latent_scores(
            model_id, combined_latent_df, drug_max_flagged, "name", "pearson_correlation", "Drug",
            make_plot=False, metadata_cols=metadata_cols, extra_cols=EXTRA_ASSIGNMENT_COLS,
        )
        drug_results.append(df)
    drug_results_df = pd.concat(drug_results, ignore_index=True) if drug_results else pd.DataFrame()
    print(f"Drug annotations: {drug_results_df.shape}")
else:
    drug_results_df = pd.DataFrame()


# ### Save the full per-sample tables, not just the top hits
#
# Notebook 6 needs every sample's score for every assigned pathway/drug (to
# compare samples feature-by-feature), not just each sample's own top 20 -
# so the full tables are persisted here rather than only existing as
# in-memory variables during this run. Restricted to features with a
# finite score for every sample: a handful of CORUM pathways have an
# `inf` NES score from GSEA itself (an upstream data-quality issue, not
# introduced here), and any downstream variance-based comparison would be
# corrupted by one `inf`/missing value - better to drop those features now,
# explicitly, than silently skip over them later.
#
# Always written, even when empty: if an earlier run produced
# NF1_data_all_drugs.parquet (say) and this run has no drug-correlation
# input to work with, skipping the write here would leave that stale file
# in place, and notebook 6 would silently read it as if it were this
# run's result.

# In[ ]:


n_samples = combined_latent_df["ModelID"].nunique()


def _complete_finite_features(df, group_cols):
    """Rows for features (identified by group_cols) with a finite
    pathway_score for every one of the n_samples samples."""
    finite_df = df[np.isfinite(df["pathway_score"])]
    coverage = finite_df.groupby(group_cols)["ModelID"].transform("nunique")
    complete_df = finite_df[coverage == n_samples]
    n_total_features = df[group_cols].drop_duplicates().shape[0]
    n_complete_features = complete_df[group_cols].drop_duplicates().shape[0]
    return complete_df, n_total_features - n_complete_features


if not pathway_results_df.empty:
    complete_pathway_df, n_dropped = _complete_finite_features(pathway_results_df, ["reactome_pathway", "gene_set_library"])
    print(f"Pathways dropped for an inf/missing score in at least one sample: {n_dropped}")
else:
    complete_pathway_df = pathway_results_df
complete_pathway_df.to_parquet(data_results_dir / "NF1_data_all_pathways.parquet", index=False)

if not drug_results_df.empty:
    complete_drug_df, n_dropped = _complete_finite_features(drug_results_df, ["name"])
    print(f"Drugs dropped for an inf/missing score in at least one sample: {n_dropped}")
else:
    complete_drug_df = drug_results_df
complete_drug_df.to_parquet(data_results_dir / "NF1_data_all_drugs.parquet", index=False)


# ### Save top associations per sample
#
# "Top" means largest magnitude, not largest raw value: pathway_score is
# signed now (see utils/latent_annotation.py), and a large negative score
# is just as notable as a large positive one - sorting by raw value would
# silently drop every strongly-negative hit from these per-sample lists.

# In[ ]:


if not pathway_results_df.empty:
    pathway_results_df["_abs_pathway_score"] = pathway_results_df["pathway_score"].abs()
    top_pathways_per_sample = (
        pathway_results_df.sort_values(["ModelID", "gene_set_library", "_abs_pathway_score"], ascending=[True, True, False])
        .groupby(["ModelID", "gene_set_library"])
        .head(20)
        .drop(columns="_abs_pathway_score")
    )
    pathway_results_df = pathway_results_df.drop(columns="_abs_pathway_score")
else:
    top_pathways_per_sample = pathway_results_df
top_pathways_per_sample.to_parquet(data_results_dir / "NF1_data_top_pathways.parquet", index=False)

if not drug_results_df.empty:
    drug_results_df["_abs_pathway_score"] = drug_results_df["pathway_score"].abs()
    top_drugs_per_sample = (
        drug_results_df.sort_values(["ModelID", "_abs_pathway_score"], ascending=[True, False])
        .groupby("ModelID")
        .head(20)
        .drop(columns="_abs_pathway_score")
    )
    drug_results_df = drug_results_df.drop(columns="_abs_pathway_score")
else:
    top_drugs_per_sample = drug_results_df
top_drugs_per_sample.to_parquet(data_results_dir / "NF1_data_top_drugs.parquet", index=False)

print(f"Saved annotated results to {data_results_dir}")

