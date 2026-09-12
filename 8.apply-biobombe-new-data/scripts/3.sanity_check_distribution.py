#!/usr/bin/env python
# coding: utf-8

# ## Sanity-check the new data's distribution against BioBombe's training data
#
# Runs before any saved model sees the new data, on purpose: a formatting bug
# or a mismatched value scale should be visible here, not discovered later as
# an implausible latent score.
#
# The comparison population is the 670-model train split
# (1.data-exploration/data/VAE_train_df.parquet), not DepMap's full ~1150-model
# cohort: that's the exact data the persisted scaler was fit on and the models
# were trained on, so it's what "in-distribution" actually means here. It's
# already in the trained gene order by construction (trained_gene_order.parquet
# is derived from these same columns), so no gene-id resolution is needed to
# use it, unlike DepMap's full CRISPRGeneEffect.csv.
#
# Two checks:
# - A shared UMAP of the new samples projected against the train cohort, to
#   see whether the new data lands inside the training distribution or off to
#   the side of it.
# - Per-gene range, missingness, and overall-scale checks on the raw (pre-scaling)
#   values, to catch a unit mismatch or a formatting bug that a UMAP plot alone
#   might not make obvious.
#
# Run this script with `8.apply-biobombe-new-data/` as the working directory.

# In[1]:


import pathlib
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap

sys.path.insert(0, "../utils")
from data_loader import load_train_test_data

sys.path.insert(0, "utils")
import data_prep as dp


# In[2]:


train_test_data_directory = pathlib.Path("../1.data-exploration/data").resolve()
scaler_path = pathlib.Path("results/depmap_gene_scaler.joblib")
data_results_dir = pathlib.Path("data/largaespada/results")
data_results_dir.mkdir(parents=True, exist_ok=True)


# In[3]:


new_scaled_df = pd.read_parquet(data_results_dir / "NF1_data_model_ready_scaled.parquet")
scaler_bundle = joblib.load(scaler_path)
gene_order = scaler_bundle["gene_order"]

# drop_columns=False keeps ModelID and every gene (unfiltered, unscaled);
# subsetting to gene_order here gives exactly what the scaler was fit on.
train_raw_df = load_train_test_data(train_test_data_directory, train_or_test="train", drop_columns=False)
train_raw_df = train_raw_df[["ModelID"] + gene_order]

train_scaled = scaler_bundle["scaler"].transform(train_raw_df[gene_order].values.astype(np.float32))
train_scaled_df = pd.DataFrame(train_scaled, columns=gene_order)
train_scaled_df.insert(0, "ModelID", train_raw_df["ModelID"].values)

print(f"New data (scaled): {new_scaled_df.shape}")
print(f"Train data (scaled): {train_scaled_df.shape}")


# ### UMAP: new data projected against the training cohort

# In[4]:


combined = pd.concat(
    [train_scaled_df.assign(source="Train data"), new_scaled_df.assign(source="New data")],
    ignore_index=True,
)

reducer = umap.UMAP(random_state=0)
embedding = reducer.fit_transform(combined[gene_order].values)
combined["umap_1"] = embedding[:, 0]
combined["umap_2"] = embedding[:, 1]


# In[5]:


# A quick look here, not the final figure — the publication-ready version of
# this plot gets built later in a dedicated R notebook, from the coordinates
# saved below, so this cell only displays inline and saves no image.
plt.figure(figsize=(8, 7))
sns.scatterplot(
    data=combined.sort_values("source"),  # draw training points first so new points sit on top
    x="umap_1", y="umap_2", hue="source",
    palette={"Train data": "#B0B0B0", "New data": "#E29578"},
    s=25, alpha=0.8, edgecolor="none",
)
plt.title("New data vs. BioBombe's training cohort (UMAP of scaled, trained-gene-order dependency scores)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend(title="Source")
plt.tight_layout()
plt.show()

umap_coords_file = data_results_dir / "NF1_data_umap_coordinates.parquet"
combined[["ModelID", "source", "umap_1", "umap_2"]].to_parquet(umap_coords_file, index=False)
print(f"Saved UMAP coordinates (for the later R figure) to {umap_coords_file}")


# ### Per-gene sanity checks on the raw (pre-scaling) values

# In[6]:


# Restricted to the trained genes specifically (scaler_bundle["gene_order"]),
# not DepMap's full ~18,443-gene set: those are the only genes that actually
# reach model inference, and NF1_data_depmap_aligned.parquet's gene-symbol
# harmonization only ever targeted the trained set (see 0.preprocess_raw_data),
# so anything outside it was never checked for a correct alignment in the
# first place and would just be noise here.
new_raw_df = pd.read_parquet(data_results_dir / "NF1_data_depmap_aligned.parquet")
new_raw_values = new_raw_df[gene_order]
train_raw_values = train_raw_df[gene_order]

new_stats = new_raw_values.stack().describe()
train_stats = train_raw_values.stack().describe()

comparison = pd.DataFrame({"new_data": new_stats, "train_data": train_stats})
print("Overall value distribution, new data vs. training data (raw, pre-scaling):")
comparison


# In[7]:


# Flag genes where the new data's value range looks implausible relative to
# the training data's, e.g. off by an order of magnitude, or a unit mismatch.
train_gene_range = train_raw_values.agg(["min", "max"]).T
new_gene_range = new_raw_values.agg(["min", "max"]).T

range_comparison = train_gene_range.join(new_gene_range, lsuffix="_train", rsuffix="_new", how="inner")
range_comparison["train_span"] = range_comparison["max_train"] - range_comparison["min_train"]
range_comparison["new_span"] = range_comparison["max_new"] - range_comparison["min_new"]
range_comparison["span_ratio"] = range_comparison["new_span"] / range_comparison["train_span"].replace(0, np.nan)

implausible_genes = range_comparison[
    (range_comparison["span_ratio"] > 3) | (range_comparison["span_ratio"] < (1 / 3))
].sort_values("span_ratio", ascending=False)

print(f"{len(implausible_genes)} / {len(range_comparison)} trained genes have a value range "
      "more than 3x wider or narrower than the training data's for that gene.")
implausible_genes.head(20)


# In[8]:


missingness = new_raw_values.isna().mean().sort_values(ascending=False)
print(f"Genes entirely missing from the new data: {(missingness == 1.0).sum()} / {len(missingness)}")
print(f"Genes with any missing values in the new data: {(missingness > 0).sum()} / {len(missingness)}")

sanity_summary = pd.DataFrame({
    "check": [
        "genes_fully_missing", "genes_partially_missing",
        "genes_with_implausible_range", "n_new_samples", "n_trained_genes",
    ],
    "value": [
        int((missingness == 1.0).sum()), int((missingness > 0).sum()),
        len(implausible_genes), new_raw_df.shape[0], range_comparison.shape[0],
    ],
})
sanity_summary.to_parquet(data_results_dir / "distribution_sanity_summary.parquet", index=False)
print(f"\nSaved sanity summary to {data_results_dir / 'distribution_sanity_summary.parquet'}")
