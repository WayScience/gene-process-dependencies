#!/usr/bin/env python
# coding: utf-8

# ## Extract the trained ensemble's gene order
#
# The BioBombe ensemble in `saved_models/` is a downloaded, pretrained
# artifact that won't be retrained, so the exact, ordered list of genes it
# was fit on is fixed too. Rather than have every consumer (e.g.
# `8.apply-biobombe-new-data`) re-derive that order through
# `load_train_test_data()` each time, this notebook derives it once and
# writes it to `data/trained_gene_order.parquet`, verified against a real
# saved PCA model's dimensionality.
#
# Run this script with `3.run-biobombe/` as the working directory.

# In[1]:


import pathlib
import sys

import joblib
import pandas as pd

script_directory = pathlib.Path("../utils/").resolve()
sys.path.insert(0, str(script_directory))

from data_loader import load_train_test_data


# In[2]:


data_directory = pathlib.Path("../1.data-exploration/data").resolve()
model_save_dir = pathlib.Path("saved_models").resolve()
output_file = pathlib.Path("data/trained_gene_order.parquet")
output_file.parent.mkdir(parents=True, exist_ok=True)


# In[3]:


# Column names survive this call; they don't survive the zero_one_normalize=True
# call used for the actual training array, so this is the closest thing to
# reading the order off the training tensor directly. The same call
# 1.train_biobombe_ensemble.py uses to build the array every model in the
# ensemble is actually fit on.
weight_data = load_train_test_data(data_directory, train_or_test="train")
trained_gene_order = weight_data.columns.tolist()

print(f"Derived gene order for {len(trained_gene_order)} genes")


# In[4]:


# A real saved PCA model is loaded only to confirm its dimensionality
# matches, as a sanity check that the derivation above is still correct.
pca_models = sorted(model_save_dir.glob("pca_*.joblib"))
if not pca_models:
    raise FileNotFoundError(
        f"No pca_*.joblib model found in {model_save_dir} to sanity-check the gene order against."
    )
pca_model = joblib.load(pca_models[0])
if pca_model.n_features_in_ != len(trained_gene_order):
    raise ValueError(
        f"{pca_models[0].name} expects {pca_model.n_features_in_} genes, but "
        f"load_train_test_data(train_or_test='train') produced {len(trained_gene_order)}. "
        "The gene-order recipe above no longer matches what the ensemble was trained on."
    )

print(f"Verified against {pca_models[0].name}: {pca_model.n_features_in_} genes")


# In[5]:


# Split each "SYMBOL (ENTREZID)" column name into its own entrez_id and
# symbol_id, the same convention 0.data-download/scripts/2.construct_gene_dictionary.py
# uses for CRISPR_gene_dictionary.parquet. Storing all three columns here
# means consumers can look up entrez_id/symbol_id directly instead of
# re-parsing dependency_column strings themselves.
entrez_ids = [x[1].strip(")").strip() for x in weight_data.columns.str.split("(")]
symbol_ids = [x[0].strip() for x in weight_data.columns.str.split("(")]

trained_gene_dict = pd.DataFrame({
    "entrez_id": entrez_ids,
    "symbol_id": symbol_ids,
    "dependency_column": trained_gene_order,
})

print(trained_gene_dict.shape)
trained_gene_dict.head()


# In[6]:


trained_gene_dict.to_parquet(output_file, index=False)
print(f"Saved trained gene order to {output_file}")
