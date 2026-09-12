#!/usr/bin/env python
# coding: utf-8

# ## Format new data two ways, and scale it for the trained ensemble
# 
# Produces two aligned versions of the new data:
# - depmap_aligned: reindexed to DepMap's full gene set and column order,
#   useful for any DepMap-adjacent comparison beyond just BioBombe
# - model_ready_scaled: reindexed to exactly the genes and order the trained
#   ensemble expects, then scaled with the persisted MinMaxScaler
# 
# The scaler is fit on the original DepMap training data the first time this
# runs, and persisted to results/depmap_gene_scaler.joblib so every later run
# (on this dataset or a future one) reuses the same scale rather than
# rescaling relative to whatever new cohort happens to be loaded.
# 
# Run this script with `8.apply-biobombe-new-data/` as the working directory.

# In[ ]:


import pathlib
import sys

import pandas as pd

sys.path.insert(0, "utils")
import data_prep as dp


# In[ ]:


data_directory = pathlib.Path("../0.data-download/data").resolve()
train_test_data_directory = pathlib.Path("../1.data-exploration/data").resolve()
model_save_dir = pathlib.Path("../3.run-biobombe/saved_models").resolve()
NF1_data_path = pathlib.Path("data/largaespada/NF1_data.parquet")

# The scaler is derived only from DepMap's already-public training data, not
# from the new dataset, so it's the one artifact in this module that's committed.
scaler_path = pathlib.Path("results/depmap_gene_scaler.joblib")

data_results_dir = pathlib.Path("data/largaespada/results")
data_results_dir.mkdir(parents=True, exist_ok=True)


# In[ ]:


prepared = dp.prepare_new_data_for_biobombe(
    new_data_path=NF1_data_path,
    data_directory=data_directory,
    train_test_data_directory=train_test_data_directory,
    model_save_dir=model_save_dir,
    scaler_path=scaler_path,
)

overlap_report = prepared["overlap_report"]
depmap_aligned = prepared["depmap_aligned"]
model_ready_scaled = prepared["model_ready_scaled"]

print(f"DepMap-aligned: {depmap_aligned.shape}")
print(f"Model-ready, scaled: {model_ready_scaled.shape}")


# In[ ]:


n_imputed_per_gene = pd.Series(model_ready_scaled.attrs["n_imputed_per_gene"])
n_genes_needing_imputation = (n_imputed_per_gene > 0).sum()
print(f"{n_genes_needing_imputation} / {len(n_imputed_per_gene)} trained genes needed imputation "
      "(missing in the new data, filled with the DepMap training mean before scaling)")

if n_genes_needing_imputation:
    print(n_imputed_per_gene.sort_values(ascending=False).head(20))


# In[ ]:


depmap_aligned.to_parquet(data_results_dir / "NF1_data_depmap_aligned.parquet", index=False)
model_ready_scaled.to_parquet(data_results_dir / "NF1_data_model_ready_scaled.parquet", index=False)

print(f"Saved formatted data to {data_results_dir}")
print(f"Scaler bundle available at {scaler_path}")

