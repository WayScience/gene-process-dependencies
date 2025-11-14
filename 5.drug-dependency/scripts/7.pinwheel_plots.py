#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pathlib

import pandas as pd
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Add project-specific paths
sys.path.insert(0, "../utils/")
sys.path.append("../5.drug-dependency")

# Import local modules
from data_loader import load_model_data
from pinwheels import compute_and_plot_latent_scores, assign_unique_latent_dims
from model_utils import extract_latent_dims
from utils import load_utils


# In[2]:


data_directory = pathlib.Path("../0.data-download/data").resolve(strict=True)
dependency_file = pathlib.Path(f"{data_directory}/CRISPRGeneEffect.parquet").resolve(strict=True)
gene_dict_file = pathlib.Path(f"{data_directory}/CRISPR_gene_dictionary.parquet").resolve(strict=True)


# In[3]:


# Load metadata
metadata_df_dir = pathlib.Path("../0.data-download/data/metadata_df.parquet").resolve(strict=True)
metadata = pd.read_parquet(metadata_df_dir)
print(metadata.shape)

#Load dependency data
dependency_df, gene_dict_df = load_model_data(dependency_file, gene_dict_file)
dependency_df.head()

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Apply the scaler to the numeric columns
dependency_df[dependency_df.select_dtypes(include=['float64', 'int']).columns] = scaler.fit_transform(
    dependency_df.select_dtypes(include=['float64', 'int'])
)


# In[4]:


train_and_test_subbed_dir = pathlib.Path("../0.data-download/data/train_and_test_subbed.parquet").resolve(strict=True)
train_and_test_subbed = pd.read_parquet(train_and_test_subbed_dir)

# Convert DataFrame to NumPy and then Tensor
train_test_array = train_and_test_subbed.to_numpy()
train_test_tensor = torch.tensor(train_test_array, dtype=torch.float32)

#Create TensorDataset and DataLoader
tensor_dataset = TensorDataset(train_test_tensor)
train_and_test_subbed_loader = DataLoader(tensor_dataset, batch_size=32, shuffle=False)


# In[5]:


corum_dir = pathlib.Path("../4.gene-expression-signatures/gsea_results/combined_z_matrix_gsea_results_corum.parquet")
corum_df = pd.read_parquet(corum_dir)

reactome_dir = pathlib.Path("../4.gene-expression-signatures/gsea_results/combined_z_matrix_gsea_results.parquet")
reactome_df = pd.read_parquet(reactome_dir)


# In[6]:


# Define cut-offs
lfc_cutoff = 0.584
fdr_cutoff = 0.05

# Filter data for significant results
significant_corum_df = corum_df[
    (corum_df['gsea_es_score'].abs() > lfc_cutoff) & 
    (corum_df['p_value'] < fdr_cutoff)
]

significant_reactome_df = reactome_df[
    (reactome_df['gsea_es_score'].abs() > lfc_cutoff) & 
    (reactome_df['p_value'] < fdr_cutoff)
]


# In[7]:


# Define the location of the saved models and output directory for results
model_save_dir = pathlib.Path("../4.gene-expression-signatures/saved_models")
output_dir = pathlib.Path("results")
output_dir.mkdir(parents=True, exist_ok=True)


# In[ ]:


# Placeholder for storing all latent representations
latent_dfs = []
cache_file = pathlib.Path("./results/combined_latent_df.parquet")

# Check if the cached file exists
if cache_file.exists():
    print(f"Loading cached latent representations from {cache_file}")
    combined_latent_df = pd.read_parquet(cache_file)

else:
    # Iterate over all saved model files
    for model_file in model_save_dir.glob("*.joblib"):
        model_file_name = model_file.stem
        try:
            parts = model_file_name.split("_")
            model_name = parts[0]
            num_components = int(parts[3])  # Example: Extract number of components
            init = int(parts[7])  # Extract initialization value
            seed = int(parts[9])  # Extract seed value
        except (IndexError, ValueError):
            print(f"Skipping file {model_file} due to unexpected filename format.")
            continue

        # Load the model
        print(f"Loading model from {model_file}")
        try:
            model = joblib.load(model_file)
        except Exception as e:
            print(f"Failed to load model from {model_file}: {e}")
            continue
        

        # Extract latent dimensions
        latent_df = extract_latent_dims(model_name, model, dependency_df, train_and_test_subbed_loader, metadata)

        # Add metadata to the latent dataframe
        latent_df["model"] = model_name
        latent_df["latent_dim_total"] = num_components
        latent_df["init"] = init
        latent_df["seed"] = seed

        # Move metadata columns to the front
        metadata_columns = ["model", "latent_dim_total", "init", "seed"]
        latent_df = latent_df.loc[:, ~latent_df.columns.duplicated()]
        latent_df.columns = latent_df.columns.astype(str)
        latent_df = latent_df[metadata_columns + [col for col in latent_df.columns if col not in metadata_columns]]
        latent_dfs.append(latent_df)

    # Combine all latent representations into one dataframe
    combined_latent_df = pd.concat(latent_dfs, ignore_index=True)
    combined_latent_df.to_parquet(cache_file)


# In[8]:


drug_dir = pathlib.Path("../5.drug-dependency/results/combined_latent_drug_correlations.parquet")
drug_df = pd.read_parquet(drug_dir)

# Load PRISM data
top_dir = "../5.drug-dependency"
data_dir = "data"

prism_df, prism_cell_df, prism_trt_df = load_utils.load_prism(
    top_dir=top_dir,
    data_dir=data_dir,
    secondary_screen=False,
    load_cell_info=True,
    load_treatment_info=True,
)


# In[9]:


# Merge drug_df with prism_trt_df to replace drug IDs with drug names
drug_df = drug_df.merge(prism_trt_df[['column_name', 'name']], 
                        left_on='drug', 
                        right_on='column_name', 
                        how='left')

# Drop the redundant column_name column
drug_df = drug_df.drop(columns=['column_name'])

# Display the updated dataframe
drug_df.head()


# In[10]:


# Define cut-offs
corr_cutoff = 0.15
pval_cutoff = 0.05

# Filter data for significant results
significant_drug_df = drug_df[
    (drug_df['pearson_correlation'].abs() > corr_cutoff) & 
    (drug_df['p_value'] < pval_cutoff)
]


# In[11]:


reactome_max = assign_unique_latent_dims(significant_reactome_df, score_col="gsea_es_score", target_col="reactome_pathway")
corum_max = assign_unique_latent_dims(significant_corum_df, score_col="gsea_es_score", target_col="reactome_pathway")
drug_max = assign_unique_latent_dims(significant_drug_df, score_col="pearson_correlation", target_col="name")


# In[12]:


corum_merge_df = []
for model_id in combined_latent_df['ModelID'].unique():
    df = compute_and_plot_latent_scores(model_id, combined_latent_df, corum_max, "reactome_pathway", "gsea_es_score", "CORUM_01")
    corum_merge_df.append(df)


# In[13]:


pathway_merge_df = []
for model_id in combined_latent_df['ModelID'].unique():
    df = compute_and_plot_latent_scores(model_id, combined_latent_df, reactome_max, "reactome_pathway", "gsea_es_score", "Reactome_01")
    pathway_merge_df.append(df)


# In[14]:


drug_merge_df = []
for model_id in combined_latent_df['ModelID'].unique():
    df = compute_and_plot_latent_scores(model_id, combined_latent_df, drug_max, "name", "pearson_correlation", "Drug_01")
    drug_merge_df.append(df)


# In[15]:


pathway_final = pd.concat(pathway_merge_df, ignore_index=True)
drug_final = pd.concat(drug_merge_df, ignore_index=True)
corum_final = pd.concat(corum_merge_df, ignore_index=True)


# In[16]:


drug_final.to_parquet("./results/all_drug_results.parquet")
corum_final.to_parquet("./results/all_corum_results.parquet")
pathway_final.to_parquet("./results/all_reactome_results.parquet")

