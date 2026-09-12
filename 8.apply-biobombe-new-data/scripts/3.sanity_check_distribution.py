#!/usr/bin/env python
# coding: utf-8

# ## Sanity-check the new data's distribution against DepMap
# 
# Runs before any saved model sees the new data, on purpose: a formatting bug
# or a mismatched value scale should be visible here, not discovered later as
# an implausible latent score.
# 
# Two checks:
# - A shared UMAP of the new samples projected against the DepMap cohort, to
#   see whether the new data lands inside the DepMap distribution or off to
#   the side of it.
# - Per-gene range, missingness, and overall-scale checks on the raw (pre-scaling)
#   values, to catch a unit mismatch or a formatting bug that a UMAP plot alone
#   might not make obvious.
# 
# Run this script with `8.apply-biobombe-new-data/` as the working directory.

# In[ ]:


import pathlib
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap

sys.path.insert(0, "utils")
import data_prep as dp


# In[ ]:


data_directory = pathlib.Path("../0.data-download/data").resolve()
scaler_path = pathlib.Path("results/depmap_gene_scaler.joblib")
data_results_dir = pathlib.Path("data/largaespada/results")
data_results_dir.mkdir(parents=True, exist_ok=True)


# In[ ]:


new_scaled_df = pd.read_parquet(data_results_dir / "NF1_data_model_ready_scaled.parquet")
depmap_df, gene_dict_df = dp.load_depmap_reference(data_directory)
scaler_bundle = joblib.load(scaler_path)

depmap_scaled_df = dp.scale_depmap_reference(depmap_df, gene_dict_df, scaler_bundle)

print(f"New data (scaled): {new_scaled_df.shape}")
print(f"DepMap (scaled): {depmap_scaled_df.shape}")


# ### UMAP: new data projected against the DepMap cohort


# ### UMAP: new data projected against the DepMap cohort

# In[ ]:


gene_order = scaler_bundle["gene_order"]
combined = pd.concat(
    [depmap_scaled_df.assign(source="DepMap"), new_scaled_df.assign(source="New data")],
    ignore_index=True,
)

reducer = umap.UMAP(random_state=0)
embedding = reducer.fit_transform(combined[gene_order].values)
combined["umap_1"] = embedding[:, 0]
combined["umap_2"] = embedding[:, 1]


# In[ ]:


# A quick look here, not the final figure — the publication-ready version of
# this plot gets built later in a dedicated R notebook, from the coordinates
# saved below, so this cell only displays inline and saves no image.
plt.figure(figsize=(8, 7))
sns.scatterplot(
    data=combined.sort_values("source"),  # draw DepMap first so new points sit on top
    x="umap_1", y="umap_2", hue="source",
    palette={"DepMap": "#B0B0B0", "New data": "#E29578"},
    s=25, alpha=0.8, edgecolor="none",
)
plt.title("New data vs. DepMap cohort (UMAP of scaled, trained-gene-order dependency scores)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend(title="Source")
plt.tight_layout()
plt.show()

umap_coords_file = data_results_dir / "NF1_data_umap_coordinates.parquet"
combined[["ModelID", "source", "umap_1", "umap_2"]].to_parquet(umap_coords_file, index=False)
print(f"Saved UMAP coordinates (for the later R figure) to {umap_coords_file}")


# ### Per-gene sanity checks on the raw (pre-scaling) values


# ### Per-gene sanity checks on the raw (pre-scaling) values

# In[ ]:


new_raw_df = pd.read_parquet(data_results_dir / "NF1_data_depmap_aligned.parquet")
new_raw_values = new_raw_df.drop(columns=["ModelID"])
depmap_raw_values = depmap_df.drop(columns=["ModelID"])

new_stats = new_raw_values.stack().describe()
depmap_stats = depmap_raw_values.stack().describe()

comparison = pd.DataFrame({"new_data": new_stats, "depmap": depmap_stats})
print("Overall value distribution, new data vs. DepMap (raw, pre-scaling):")
comparison


# In[ ]:


# Flag genes where the new data's value range looks implausible relative to
# DepMap's, e.g. off by an order of magnitude, or a unit mismatch.
depmap_gene_range = depmap_raw_values.agg(["min", "max"]).T
new_gene_range = new_raw_values.agg(["min", "max"]).T

range_comparison = depmap_gene_range.join(new_gene_range, lsuffix="_depmap", rsuffix="_new", how="inner")
range_comparison["depmap_span"] = range_comparison["max_depmap"] - range_comparison["min_depmap"]
range_comparison["new_span"] = range_comparison["max_new"] - range_comparison["min_new"]
range_comparison["span_ratio"] = range_comparison["new_span"] / range_comparison["depmap_span"].replace(0, np.nan)

implausible_genes = range_comparison[
    (range_comparison["span_ratio"] > 3) | (range_comparison["span_ratio"] < (1 / 3))
].sort_values("span_ratio", ascending=False)

print(f"{len(implausible_genes)} / {len(range_comparison)} shared genes have a value range "
      "more than 3x wider or narrower than DepMap's for that gene.")
implausible_genes.head(20)


# In[ ]:


missingness = new_raw_values.isna().mean().sort_values(ascending=False)
print(f"Genes entirely missing from the new data: {(missingness == 1.0).sum()} / {len(missingness)}")
print(f"Genes with any missing values in the new data: {(missingness > 0).sum()} / {len(missingness)}")

sanity_summary = pd.DataFrame({
    "check": [
        "genes_fully_missing", "genes_partially_missing",
        "genes_with_implausible_range", "n_new_samples", "n_shared_genes",
    ],
    "value": [
        int((missingness == 1.0).sum()), int((missingness > 0).sum()),
        len(implausible_genes), new_raw_df.shape[0], range_comparison.shape[0],
    ],
})
sanity_summary.to_parquet(data_results_dir / "distribution_sanity_summary.parquet", index=False)
print(f"\nSaved sanity summary to {data_results_dir / 'distribution_sanity_summary.parquet'}")

