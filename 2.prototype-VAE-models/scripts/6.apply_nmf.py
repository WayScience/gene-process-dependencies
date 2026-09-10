#!/usr/bin/env python
# coding: utf-8

# ## Apply NMF to gene dependencies
# 
# Non-negative Matrix Factorization (NMF) is a dimensionality reduction technique that factors a non-negative matrix into two non-negative matrices. This is particularly useful when the data is inherently non-negative.
# 
# We apply it to GeneEffect scores here and save latent representations (NMF components) for downstream comparative analyses (to compare with BetaVAE).

# In[1]:


import sys
import pathlib
import pandas as pd
import plotnine as gg
from sklearn.decomposition import NMF

sys.path.insert(0, "../utils/")
from data_loader import load_model_data


# In[2]:


data_directory = pathlib.Path("../0.data-download/data").resolve()
dependency_file = pathlib.Path(f"{data_directory}/CRISPRGeneEffect.parquet").resolve()
gene_dict_file = pathlib.Path(f"{data_directory}/CRISPR_gene_dictionary.parquet").resolve()


output_dir = pathlib.Path("results").resolve()
nmf_output_file = pathlib.Path(f"{output_dir}/nmf_latent.parquet.gz").resolve()
output_nmf_weights_file = pathlib.Path(f"{output_dir}/NMF_weight_matrix_gsea.parquet").resolve()


# In[3]:


nmf_components = 50


# In[4]:


# Load data
dependency_df, gene_dict_df = load_model_data(dependency_file, gene_dict_file)


# In[5]:


# Ensure all values are non-negative by shifting the data
min_value = dependency_df.drop(columns=["ModelID"]).min().min()
if min_value < 0:
   dependency_df_non_negative = dependency_df.drop(columns=["ModelID"]) - min_value
else:
   dependency_df_non_negative = dependency_df.drop(columns=["ModelID"])


# # Perform NMF

# In[6]:


nmf = NMF(n_components=nmf_components, random_state=0)
W = nmf.fit_transform(dependency_df_non_negative)
H = nmf.components_


# In[7]:


# Transform models into NMF space (W matrix)
dependency_df_transformed = pd.DataFrame(W)


# In[8]:


# Recode column space and add back model IDs
dependency_df_transformed.columns = [f"NMF_{x}" for x in range(0, dependency_df_transformed.shape[1])]
dependency_df_transformed = pd.concat([dependency_df.loc[:, "ModelID"], dependency_df_transformed], axis="columns")


dependency_df_transformed.to_parquet(nmf_output_file, index=False)


print(dependency_df_transformed.shape)
dependency_df_transformed.head(3)


# In[9]:


# Obtain weights (H matrix), which can be used in GSEA
nmf_weights = pd.DataFrame(H, columns=dependency_df_non_negative.columns.tolist()).transpose()
nmf_weights.columns = [f"NMF_{x}" for x in range(nmf_weights.shape[1])]


nmf_weights = nmf_weights.reset_index().rename(columns={"index": "genes"})


nmf_weights.to_parquet(output_nmf_weights_file, index=False)


print(nmf_weights.shape)
nmf_weights.head(3)

