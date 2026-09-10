#!/usr/bin/env python
# coding: utf-8

# ## Apply PCA to gene dependencies
# 
# Principal component analysis (PCA) is a commonly-used dimensionality reduction method that finds components that explain orthogonal variation in the data in a descending fashion.
# 
# We apply it to GeneEffect scores here and save latent representations (PCA components) for downstream comparative analyses (to compare with BetaVAE).

# In[1]:


import sys
import pathlib
import pandas as pd
import plotnine as gg
from sklearn.decomposition import PCA

sys.path.insert(0, "../utils/")
from data_loader import load_model_data


# In[2]:


data_directory = pathlib.Path("../0.data-download/data").resolve()
dependency_file = pathlib.Path(f"{data_directory}/CRISPRGeneEffect.parquet").resolve()
gene_dict_file = pathlib.Path(f"{data_directory}/CRISPR_gene_dictionary.parquet").resolve()

output_dir = pathlib.Path("results").resolve()
pca_output_file = pathlib.Path(f"{output_dir}/pca_latent.parquet.gz").resolve()
output_pca_weights_file = pathlib.Path(f"{output_dir}/PCA_weight_matrix_gsea.parquet").resolve()


# In[3]:


pca_components = 50


# In[4]:


# Load data
dependency_df, gene_dict_df = load_model_data(dependency_file, gene_dict_file)


# # Perform PCA

# In[5]:


pca = PCA(n_components=pca_components)
pca.fit(dependency_df.drop(columns=["ModelID"]))


# In[6]:


# Output explained variance and quickly visualize
explained_var = pd.DataFrame(pca.explained_variance_ratio_, columns=["explained_var"]).reset_index()

(
    gg.ggplot(explained_var, gg.aes(x="index", y="explained_var"))
    + gg.geom_bar(stat="identity")
)


# In[7]:


# Transform models into pca space
dependency_df_transformed = pd.DataFrame(
    pca.transform(dependency_df.drop(columns=["ModelID"]))
)

# Recode column space and add back model IDs
dependency_df_transformed.columns = [f"PCA_{x}" for x in range(0, dependency_df_transformed.shape[1])]
dependency_df_transformed = pd.concat([dependency_df.loc[:, "ModelID"], dependency_df_transformed], axis="columns")

dependency_df_transformed.to_parquet(pca_output_file, index=False)

print(dependency_df_transformed.shape)
dependency_df_transformed.head(3)


# In[8]:


# Obtain weights, which can be used in GSEA
pca_weights = pd.DataFrame(pca.components_, columns=dependency_df.drop(columns=["ModelID"]).columns.tolist()).transpose()
pca_weights.columns = [f"PCA_{x}" for x in range(0, pca_weights.shape[1])]

pca_weights = pca_weights.reset_index().rename(columns={"index": "genes"})

pca_weights.to_parquet(output_pca_weights_file, index=False)

print(pca_weights.shape)
pca_weights.head(3)

