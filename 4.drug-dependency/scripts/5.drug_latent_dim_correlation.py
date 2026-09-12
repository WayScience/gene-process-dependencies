#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import sys

import joblib
import logging
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from utils import load_utils

sys.path.insert(0, "../utils/")
from data_loader import load_model_data
from model_utils import extract_latent_dims


# In[2]:


# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def perform_correlation(latent_df: pd.DataFrame, 
    drug_df: pd.DataFrame, 
    model_name: str, 
    num_components: int, 
    shuffle: bool = False):
    """
    Perform Pearson correlation between latent dimensions and drug dependency scores.
    
    Parameters:
    latent_df (pd.DataFrame): Dataframe containing latent dimensions (e.g., PCA/ICA/NMF).
    drug_df (pd.DataFrame): Dataframe containing drug dependency scores with `ModelID`.
    model_name (str): Name of the model used to extract the latent dimensions (PCA, ICA, NMF).
    num_components (int): Number of latent dimensions/components in the model.
    
    Returns:
    pd.DataFrame: Correlation results between latent dimensions and drug scores.
    """
    logging.info(f"Starting correlation analysis for model: {model_name} with {num_components} components.")

    correlation_results = []
    
    # Set index to ModelID if present
    if 'ModelID' in latent_df.columns:
        latent_df = latent_df.set_index('ModelID')
        logging.info("Set ModelID as index for latent_df.")
    
    # Align both dataframes based on the ModelID
    common_model_ids = latent_df.index.intersection(drug_df.index)
    logging.info(f"Found {len(common_model_ids)} common ModelIDs.")

    # Filter both dataframes to keep only common ModelIDs
    latent_df_filtered = latent_df.loc[common_model_ids]
    prism_df_filtered = drug_df.loc[common_model_ids]

    # Check and log the variance of each latent dimension and drug response column
    latent_variance = latent_df_filtered.var()
    prism_variance = prism_df_filtered.var()
    
    logging.info(f"Number of latent dimensions with non-zero variance: {(latent_variance != 0).sum()}.")
    logging.info(f"Number of drug columns with non-zero variance: {(prism_variance != 0).sum()}.")

    # Filter out constant columns (variance == 0)
    latent_df_filtered = latent_df_filtered.loc[:, latent_variance != 0]
    prism_df_filtered = prism_df_filtered.loc[:, prism_variance != 0]

    # Loop over each latent dimension and calculate correlation with each drug
    for latent_col in latent_df_filtered.columns:
        logging.info(f"Processing latent dimension: {latent_col}")
        for drug_col in prism_df_filtered.columns:
            latent_values = latent_df_filtered[latent_col]
            drug_values = prism_df_filtered[drug_col]
            
            # Check if either column is constant
            if latent_values.nunique() <= 1 or drug_values.nunique() <= 1:
                corr = np.nan
                logging.warning(f"Skipping correlation for {latent_col} and {drug_col} due to constant values.")
            else:
                # Drop missing values for both columns
                valid_data = pd.concat([latent_values, drug_values], axis=1).dropna()
                latent_values_valid = valid_data[latent_col]
                drug_values_valid = valid_data[drug_col]
                
                if len(latent_values_valid) > 1 and len(drug_values_valid) > 1:
                    # Calculate Pearson correlation
                    corr, p_value = pearsonr(latent_values_valid, drug_values_valid)
                else:
                    corr = np.nan
                    p_value = np.nan
                    logging.warning(f"Insufficient valid data for correlation between {latent_col} and {drug_col}.")
        
            # Store the results
            result_row = {
                "z": int(latent_col.replace("z_", "")),
                "full_model_z": num_components,
                "model": str(model_name),
                "drug": str(drug_col),
                "pearson_correlation": corr,
                "p_value": p_value,
                "shuffled": shuffle
            }
            correlation_results.append(result_row)
    
    # Convert results into a dataframe
    correlation_results_df = pd.DataFrame(correlation_results)
    
    logging.info("Correlation analysis completed.")
    return correlation_results_df


# In[3]:


data_directory = pathlib.Path("../0.data-download/data").resolve()
dependency_file = pathlib.Path(f"{data_directory}/CRISPRGeneEffect.csv").resolve()
gene_dict_file = pathlib.Path(f"{data_directory}/CRISPR_gene_dictionary.parquet").resolve()


# In[4]:


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
prism_df.set_index('ModelID', inplace=True)
prism_df.head()


# In[5]:


# Create a copy of the prism dataframe to shuffle the values without removing the ModelID column
prism_df_shuffled = prism_df.copy()

# Iterate over each drug column (except 'ModelID') and shuffle its values
for drug_col in prism_df_shuffled.columns:
    if drug_col != 'ModelID':
        # Shuffle the values of the column without resetting the index
        prism_df_shuffled[drug_col] = prism_df_shuffled[drug_col].sample(frac=1, random_state=None).values

prism_df_shuffled.head()


# In[6]:


# Load metadata
metadata_df_dir = pathlib.Path("../1.data-exploration/data/metadata_df.parquet")
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


# In[7]:


train_and_test_subbed_dir = pathlib.Path("../1.data-exploration/data/train_and_test_subbed.parquet")
train_and_test_subbed = pd.read_parquet(train_and_test_subbed_dir)


# Convert DataFrame to NumPy and then Tensor
train_test_array = train_and_test_subbed.to_numpy()
train_test_tensor = torch.tensor(train_test_array, dtype=torch.float32)

#Create TensorDataset and DataLoader
tensor_dataset = TensorDataset(train_test_tensor)
train_and_test_subbed_loader = DataLoader(tensor_dataset, batch_size=32, shuffle=False)


# In[8]:


# Define the location of the saved models and output directory for correlation results
model_save_dir = pathlib.Path("../3.run-biobombe/saved_models")
output_dir = pathlib.Path("results")
output_dir.mkdir(parents=True, exist_ok=True)

# Latent dimensions and model names to iterate over
latent_dims = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 150, 200]
model_names = ["pca", "ica", "nmf", "vanillavae", "betavae", "betatcvae"]

# File to store the combined correlation results
final_output_file = output_dir / "combined_latent_drug_correlations.parquet"
try:
    combined_results_df = pd.read_parquet(final_output_file)
    print(f"Loaded existing results from {final_output_file}")
except FileNotFoundError:
    # If the file doesn't exist, initialize an empty DataFrame
    combined_results_df = pd.DataFrame()
    logging.error("FileNotFoundError: No existing file found. Initialized an empty DataFrame.")

for num_components in latent_dims:
    for model_name in model_names:
        # Check if this model and latent dimension have already been processed
        if not combined_results_df.empty:
            if ((combined_results_df['model'] == model_name) & 
                (combined_results_df['full_model_z'] == num_components)).any():
                print(f"Skipping {model_name} with {num_components} dimensions as it is already processed.")
                continue  # Skip to the next iteration if this combination is already present
        
        # Load the saved model
        model_filename = model_save_dir / f"{model_name}_{num_components}_components_model.joblib"
        if model_filename.exists():
            print(f"Loading model from {model_filename}")
            model = joblib.load(model_filename)
            
            latent_df = extract_latent_dims(model_name, model, dependency_df, train_and_test_subbed_loader, metadata)

            latent_df.columns = ['ModelID'] + [f'z_{col}' if isinstance(col, int) else col for col in latent_df.columns[1:]]
            # Perform Pearson correlation between latent dimensions and drug data
            correlation_results_df = perform_correlation(latent_df, prism_df, model_name, num_components)
            # Perform Pearson correlation for shuffled data (negative control)
            negative_control_results_df = perform_correlation(latent_df, prism_df_shuffled, model_name, num_components, shuffle=True)
            # Concatenate results to the combined dataframe
            combined_results_df = pd.concat([combined_results_df, correlation_results_df, negative_control_results_df], ignore_index=True)
        else:
            raise FileNotFoundError(f"Model file {model_filename} not found. Script will terminate.")


# Save the combined results to a parquet file
combined_results_df.to_parquet(final_output_file)
print(f"Saved combined results to {final_output_file}")


# In[9]:


# Assuming 'drug_column_name' is the column in prism_trt_df that matches the 'drug' column in correlation_df
prism_trt_df_filtered = prism_trt_df[['column_name', 'name', 'moa', 'target', 'indication', 'phase']]

# Merge correlation_df with prism_trt_df based on the 'drug' column in correlation_df and the matching column in prism_trt_df
correlation_df_merged = pd.merge(combined_results_df, prism_trt_df_filtered, how='left', left_on='drug', right_on='column_name')

# Drop the redundant drug_column_name column after the merge if needed
correlation_df_merged = correlation_df_merged.drop(columns=['column_name'])

significant_corr_df = correlation_df_merged[
    (correlation_df_merged['pearson_correlation'].abs() > 0.1)
]


# In[10]:


combined_results_df.sort_values(by='pearson_correlation', key=abs, ascending = False).head(50)

