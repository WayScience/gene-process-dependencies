#!/usr/bin/env python
# coding: utf-8

# ## Annotate the new data's latent scores with pathways and drugs
# 
# Links each new sample's BioBombe latent scores to the same GSEA pathway
# results and drug correlations used everywhere else in this project, via
# `utils/pinwheels.py`'s `compute_and_plot_latent_scores` /
# `assign_unique_latent_dims` — the same functions
# `6.collab-data/scripts/3.pinwheel_plots.py` uses for collaborator data.
# 
# Run this script with `8.apply-biobombe-new-data/` as the working directory.

# In[ ]:


import pathlib
import sys

import pandas as pd

sys.path.insert(0, "../utils/")
from pinwheels import assign_unique_latent_dims, compute_and_plot_latent_scores


# In[ ]:


data_results_dir = pathlib.Path("data/largaespada/results")

combined_latent_df = pd.read_parquet(data_results_dir / "NF1_data_latent_scores.parquet")
print(f"Latent scores: {combined_latent_df.shape}")


# ### Pathway annotation: CORUM and Reactome, kept separate
#
# A real, ensemble-wide Reactome run did happen for this project at some
# point — 4.drug-dependency/results/all_reactome_results.parquet holds genuine
# Reactome pathways (R-HSA-tagged names, e.g. "rRNA Modification In Nucleus
# And Cytosol R-HSA-6790901") spanning all 6 model types, so this isn't
# hypothetical. What's missing is the raw, per-latent-dimension GSEA hit table
# that would have fed it (the direct analog of combined_z_matrix_gsea_results.parquet,
# but for Reactome) — it isn't in 3.run-biobombe/gsea_results/ or anywhere
# else in this repo's git history. `reactome_gsea_path` below is where
# `2.interpret_biobombe_with_gsea.py` would need to write one (by re-running
# it with a Reactome library) for that half to populate again.
#
# `3.run-biobombe/gsea_results/combined_z_matrix_gsea_results.parquet` — the
# one raw file that does exist — is CORUM, confirmed directly: every one of
# its 170 pathway names matches CORUM's complex-naming convention ("X complex
# (human)"), never Reactome's R-HSA- IDs. `4.drug-dependency/scripts/7.pinwheel_plots.py`
# actually has this backwards (it labels this file `reactome_df` and expects
# a nonexistent "_corum"-suffixed file for CORUM) — that script has never
# successfully run against its current code, since the file it wants for
# CORUM was never committed either.
#
# Both results, when present, use a column literally called "reactome_pathway"
# for the pathway name — that's `2.interpret_biobombe_with_gsea.py`'s schema
# regardless of which library was actually run, not a labeling mistake here.


# ### Pathway annotation (CORUM GSEA results)

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
    # Drop the shuffled negative-control rows and keep only nominally significant
    # hits. This file has no fdr column (unlike the single-VAE GSEA results
    # elsewhere in the project), so significance here is by p_value alone.
    significant_gsea_df = gsea_df[(~gsea_df["shuffled"]) & (gsea_df["p_value"] < 0.05)].copy()
    print(f"{library}: {significant_gsea_df.shape[0]} / {gsea_df.shape[0]} significant hits")

    # nes_score (normalized enrichment score), not gsea_es_score: NES adjusts
    # for gene-set-size effects, and it's what the previously-run ensemble-wide
    # annotation (4.drug-dependency/results/all_corum_results.parquet /
    # all_reactome_results.parquet) actually used.
    pathway_max_by_library[library] = assign_unique_latent_dims(
        significant_gsea_df, score_col="nes_score", target_col="reactome_pathway"
    )


# In[ ]:


pathway_results = []
for library, pathway_max in pathway_max_by_library.items():
    for model_id in combined_latent_df["ModelID"].unique():
        df = compute_and_plot_latent_scores(
            model_id, combined_latent_df, pathway_max, "reactome_pathway", "nes_score",
            library, make_plot=False,
        )
        df["gene_set_library"] = library
        pathway_results.append(df)

pathway_results_df = pd.concat(pathway_results, ignore_index=True) if pathway_results else pd.DataFrame()
print(f"Pathway annotations (CORUM + Reactome combined): {pathway_results_df.shape}")


# ### Drug annotation
#
# This step needs `4.drug-dependency/results/combined_latent_drug_correlations.parquet`,
# the ensemble-wide drug correlation table `4.drug-dependency/scripts/5.drug_latent_dim_correlation.py`
# produces. That script depends on `saved_models/`, which only exists locally
# now — if it hasn't been run yet, drug annotation is skipped here rather than failing.


# ### Drug annotation
# 
# This step needs `4.drug-dependency/results/combined_latent_drug_correlations.parquet`,
# the ensemble-wide drug correlation table `4.drug-dependency/scripts/5.drug_latent_dim_correlation.py`
# produces. That script depends on `saved_models/`, which only exists locally
# now — if it hasn't been run yet, drug annotation is skipped here rather than failing.

# In[ ]:


drug_correlation_path = pathlib.Path("../4.drug-dependency/results/combined_latent_drug_correlations.parquet")

if drug_correlation_path.exists():
    drug_df = pd.read_parquet(drug_correlation_path)
    significant_drug_df = drug_df[
        (drug_df["pearson_correlation"].abs() > 0.15) & (drug_df["p_value"] < 0.05)
    ].copy()
    drug_max = assign_unique_latent_dims(
        significant_drug_df, score_col="pearson_correlation", target_col="name"
    )

    drug_results = []
    for model_id in combined_latent_df["ModelID"].unique():
        df = compute_and_plot_latent_scores(
            model_id, combined_latent_df, drug_max, "name", "pearson_correlation", "Drug",
            make_plot=False,
        )
        drug_results.append(df)
    drug_results_df = pd.concat(drug_results, ignore_index=True) if drug_results else pd.DataFrame()
    print(f"Drug annotations: {drug_results_df.shape}")
else:
    drug_results_df = pd.DataFrame()
    print(
        f"{drug_correlation_path} not found — skipping drug annotation. "
        "Run 4.drug-dependency/scripts/5.drug_latent_dim_correlation.py first to enable it."
    )


# ### Save top associations per sample


# ### Save top associations per sample

# In[ ]:


top_pathways_per_sample = (
    pathway_results_df.sort_values(["ModelID", "gene_set_library", "pathway_score"], ascending=[True, True, False])
    .groupby(["ModelID", "gene_set_library"])
    .head(20)
    if not pathway_results_df.empty else pathway_results_df
)
top_pathways_per_sample.to_parquet(data_results_dir / "NF1_data_top_pathways.parquet", index=False)

if not drug_results_df.empty:
    top_drugs_per_sample = (
        drug_results_df.sort_values(["ModelID", "pathway_score"], ascending=[True, False])
        .groupby("ModelID")
        .head(20)
    )
    top_drugs_per_sample.to_parquet(data_results_dir / "NF1_data_top_drugs.parquet", index=False)

print(f"Saved annotated results to {data_results_dir}")

