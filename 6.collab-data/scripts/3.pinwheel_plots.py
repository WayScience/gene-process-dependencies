#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import pathlib
import sys
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, "../utils/")
from pinwheels import compute_and_plot_latent_scores, extract_top_pathways_with_cancer, plot_stacked_bar_chart

sys.path.append("../4.drug-dependency")
from utils import load_utils


# In[18]:


corum_dir = pathlib.Path("../3.run-biobombe/gsea_results/combined_z_matrix_gsea_results_corum.parquet")
corum_df = pd.read_parquet(corum_dir)

reactome_dir = pathlib.Path("../3.run-biobombe/gsea_results/combined_z_matrix_gsea_results_1.parquet")
reactome_df = pd.read_parquet(reactome_dir)

total_drugs = pd.read_csv("../4.drug-dependency/results/total_drug_df.csv")
merged_df = pd.read_csv("../6.collab-data/results/final_drug_w_paths.csv")


# In[19]:


# Drop duplicates to count unique drug appearances per ModelID
unique_drug_model = total_drugs[["ModelID", "name"]].drop_duplicates()

total_modelids = unique_drug_model["ModelID"].nunique()
print(f"Total unique ModelIDs (all cancers): {total_modelids}")

# 1. OVERALL drug appearance frequency
total_modelids = unique_drug_model["ModelID"].nunique()
overall_counts = unique_drug_model["name"].value_counts().reset_index()
overall_counts.columns = ["drug", "count"]
overall_counts["percent"] = overall_counts["count"] / total_modelids * 100

# 2. Subset for brain tumors: Neuroblastoma and Diffuse Glioma
brain_df = total_drugs[total_drugs["OncotreePrimaryDisease"].isin(["Neuroblastoma", "Diffuse Glioma"])]

# Drop duplicates within this subset
brain_unique = brain_df[["ModelID", "name"]].drop_duplicates()

# Count again for brain tumor model IDs
brain_modelids = brain_unique["ModelID"].nunique()
brain_counts = brain_unique["name"].value_counts().reset_index()
brain_counts.columns = ["drug", "count"]
brain_counts["percent"] = brain_counts["count"] / brain_modelids * 100

# Optionally merge both to compare
comparison_df = pd.merge(
    overall_counts,
    brain_counts,
    on="drug",
    how="outer",
    suffixes=("_overall", "_brain_tumors")
).fillna(0)

# Sort by highest percentage in brain tumors
comparison_df = comparison_df.sort_values(by="percent_brain_tumors", ascending=False)

# Save to file if needed
comparison_df.to_csv("drug_appearance_comparison.csv", index=False)

print(comparison_df.head(50))


# In[20]:


#RNA-seq predicted latent dataframe
latent_dir = pathlib.Path("../6.collab-data/results/phgg_latent_predictions.parquet").resolve()
latent_df = pd.read_parquet(latent_dir)
latent_df['latent_score'] = pd.to_numeric(latent_df['latent_score'], errors='coerce').clip(lower=0)


# In[21]:


latent_df.head()


# In[22]:


reactome_dims = pathlib.Path("../4.drug-dependency/results/reactome_paths")
reactome_max = pd.read_parquet(reactome_dims)

corum_dims = pathlib.Path("../4.drug-dependency/results/corum_paths")
corum_max = pd.read_parquet(corum_dims)

drug_dims = pathlib.Path("../4.drug-dependency/results/drug_results")
drug_max = pd.read_parquet(drug_dims)


# In[23]:


pathway_merge_df = []
corum_merge_df = []
for sample in latent_df['ModelID'].unique():
    p_df = compute_and_plot_latent_scores(sample, latent_df, reactome_max, "reactome_pathway", "nes_score", "Reactome")
    c_df = compute_and_plot_latent_scores(sample, latent_df, corum_max, "reactome_pathway", "nes_score", "CORUM")
    pathway_merge_df.append(p_df)
    corum_merge_df.append(c_df)


# In[24]:


drug_merge_df = []
for sample in latent_df['ModelID'].unique():
    df = compute_and_plot_latent_scores(sample, latent_df, drug_max, "name", "pearson_correlation", "Drug")
    drug_merge_df.append(df)


# In[25]:


drug_merge_df = pd.concat(drug_merge_df, ignore_index=True)


# In[26]:


drug_merge_df.head()


# In[27]:


hist_df = pathlib.Path("../4.drug-dependency/results/histogram_results")
drug_hist = pd.read_parquet(hist_df)


# In[28]:


drug_merge_df['OncotreePrimaryDisease'] = "Pediatric High-Grade Glioma"
for col in drug_hist.columns:
    if col not in drug_merge_df.columns:
        drug_merge_df[col] = pd.NA  # or np.nan depending on downstream needs

# Reorder columns to match drug_hist
drug_merge_df = drug_merge_df[drug_hist.columns]

# Append the modified dataframe
drug_hist = pd.concat([drug_hist, drug_merge_df], ignore_index=True)


# In[29]:


drug_merge_df.head()


# In[30]:


merge_keys = ['z', 'model', 'init', 'latent_dim_total']


# In[31]:


correlation_df = pd.read_parquet("../4.drug-dependency/results/drug_correlation.parquet")
correlation_df = correlation_df.rename(columns={'full_model_z': 'latent_dim_total'})


# In[32]:


top_drugs_df = (
    drug_merge_df
    .sort_values(["ModelID", "latent_score"], ascending=[True, False])
    .groupby("ModelID")
    .head(200)
    .reset_index(drop=True)
)


# In[33]:


merged_df = pd.merge(
    top_drugs_df,
    correlation_df[['z', 'model', 'init', 'latent_dim_total', 'name', 'moa', 'target', 'indication', 'phase', 'Associated Pathways']],
    on=merge_keys + ['name'],
    how='left'
)


# In[34]:


merged_df.to_csv("results/final_drug_w_paths.csv")

