#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from scipy.stats import pearsonr
import pathlib
import sys
import numpy as np

sys.path.append("../")
from utils import load_utils


# In[2]:


# Load PRISM data
top_dir = "../4.drug-dependency"
data_dir = "data"

prism_df, prism_cell_df, prism_trt_df = load_utils.load_prism(
    top_dir=top_dir,
    data_dir=data_dir,
    secondary_screen=False,
    load_cell_info=True,
    load_treatment_info=True,
)

# Reset the index and name it ModelID
prism_df.reset_index(inplace=True)
prism_df.rename(columns={'index': 'ModelID'}, inplace=True)

# Check the result
print(prism_df.shape)
prism_df.head(3)


# In[3]:


#Load reactome pathways
pathway_dir = pathlib.Path("../2.prototype-VAE-models/results/significant_gsea_results.parquet.gz").resolve()
pathway_df = pd.read_parquet(pathway_dir)
pathway_df.head()


# In[4]:


# load the latent dim matrix 
latent_dir = pathlib.Path("../2.prototype-VAE-models/results/latent_df.parquet").resolve()
latent_df = pd.read_parquet(latent_dir)
latent_df.head()
print(latent_df.shape)


# In[5]:


# Ensure ModelID is the index for both dataframes to align 
latent_df.set_index('ModelID', inplace=True)
prism_df.set_index('ModelID', inplace=True)
latent_df.head()


# In[6]:


# Align both dataframes based on the ModelID
common_model_ids = latent_df.index.intersection(prism_df.index)


# In[7]:


# Filter both dataframes to keep only common ModelIDs
latent_df_filtered = latent_df.loc[common_model_ids]
prism_df_filtered = prism_df.loc[common_model_ids]


# In[8]:


# Check the variance of each latent dimension and drug response column
latent_variance = latent_df_filtered.var()
prism_variance = prism_df_filtered.var()

# Filter out constant columns (variance == 0)
latent_df_filtered = latent_df_filtered.loc[:, latent_variance != 0]
prism_df_filtered = prism_df_filtered.loc[:, prism_variance != 0]


# In[9]:


latent_df_filtered.head()


# In[10]:


prism_df_filtered.head()


# In[11]:


# Create a dataframe to store the Pearson correlation results
correlation_results = []

# Iterate over each latent dimension and drug column
for latent_col in latent_df_filtered.columns:
    for drug_col in prism_df_filtered.columns:
        latent_values = latent_df_filtered[latent_col]
        drug_values = prism_df_filtered[drug_col]

        # Check if either column is constant
        if latent_values.nunique() <= 1 or drug_values.nunique() <= 1:
            corr = np.nan
        else:
            # Drop missing values for both columns
            valid_data = pd.concat([latent_values, drug_values], axis=1).dropna()
            latent_values_valid = valid_data[latent_col]
            drug_values_valid = valid_data[drug_col]

            if len(latent_values_valid) > 1 and len(drug_values_valid) > 1:
                # Calculate Pearson correlation
                corr, _ = pearsonr(latent_values_valid, drug_values_valid)
            else:
                corr = np.nan
                print("nan")
        
        # Store the result
        correlation_results.append({
            'latent_dimension': latent_col,
            'drug': drug_col,
            'correlation': corr
        })

# Convert the results to a dataframe for easier analysis
correlation_df = pd.DataFrame(correlation_results)

# Display the correlation dataframe
correlation_df.sort_values(by='correlation', key=abs, ascending=False).head(50)


# In[12]:


#Sort pathways by NES score (ascending order)
pathway_df.sort_values(by='nes', ascending=True)

ranked_gsea = pathway_df.sort_values(by='nes', key=abs, ascending=False)

#Group by 'z_dim' and aggregate 'Term' into a list of associated pathways
grouped_pathway_df = ranked_gsea.groupby('z_dim').apply(lambda x: x.nlargest(10, 'nes')['Term'].tolist()).reset_index(drop=False)

# remove z_
grouped_pathway_df['z_dim'] = grouped_pathway_df['z_dim'].str.replace('z_', '', regex=False)

grouped_pathway_df.columns = ['latent dimension', 'Associated Pathways']

grouped_pathway_df.head()


# In[13]:


# Assuming 'drug_column_name' is the column in prism_trt_df that matches the 'drug' column in correlation_df
prism_trt_df_filtered = prism_trt_df[['column_name', 'name', 'moa', 'target', 'indication', 'phase']]

# Merge correlation_df with prism_trt_df based on the 'drug' column in correlation_df and the matching column in prism_trt_df
correlation_df_merged1 = pd.merge(correlation_df, prism_trt_df_filtered, how='left', left_on='drug', right_on='column_name')

# Drop the redundant drug_column_name column after the merge if needed
correlation_df_merged1 = correlation_df_merged1.drop(columns=['column_name'])

# Merge correlation_df with prism_trt_df based on the 'drug' column in correlation_df and the matching column in prism_trt_df
correlation_df_merged = pd.merge(correlation_df_merged1, grouped_pathway_df, how='left', left_on='latent_dimension', right_on='latent dimension')

# Drop the redundant drug_column_name column after the merge if needed
correlation_df_merged = correlation_df_merged.drop(columns=['latent dimension'])

significant_corr_df = correlation_df_merged[
    (correlation_df_merged['correlation'].abs() > 0.1)
]
# saving results as single output file
correlation_dir = pathlib.Path("./results/drug_correlation.parquet")
significant_corr_df.to_parquet(correlation_dir)

# Display the updated dataframe with the new columns
correlation_df_merged.sort_values(by='correlation', key=abs, ascending=False).head(50)

