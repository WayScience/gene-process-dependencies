#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import sys

# Add custom module paths
sys.path.append("../5.RNAseq/utils")
sys.path.append("../4.drug-dependency")

import pandas as pd

from analyze_utils import find_drugs
from utils import load_utils


# In[2]:


# Define the location of the saved models and output directory for results
output_dir = pathlib.Path("results")
output_dir.mkdir(parents=True, exist_ok=True)

# File to store the combined correlation results
final_test_results_file = output_dir / "test_r2.parquet"
final_test_predictions_file = output_dir /  "test_preds.parquet"


# In[ ]:


#load RNAseq predictions and R2 values
final_test_results_df = pd.read_parquet(final_test_results_file)
final_test_predictions_df = pd.read_parquet(final_test_predictions_file)


# In[4]:


meta_cols = ['model', 'latent_dimension', 'z_dimension', 'type', 'init', 'seed']

# Get all other columns, sorted alphabetically
data_cols = sorted([col for col in final_test_predictions_df.columns if col not in meta_cols])

# Reorder the dataframe
final_test_predictions_df = final_test_predictions_df[meta_cols + data_cols]

final_test_predictions_df.head()


# In[5]:


# Load pre-split RNASeq data
train = pd.read_parquet("./data/RNASeq_train_zscored.parquet")
val = pd.read_parquet("./data/RNASeq_val_zscored.parquet")
test = pd.read_parquet("./data/RNASeq_test_zscored.parquet")


# In[ ]:


prism_dir = pathlib.Path("../4.drug-dependency/results")
prism_dir.mkdir(parents=True, exist_ok=True)
final_output_file = (prism_dir / "combined_latent_drug_correlations.parquet").resolve(strict=True)
combined_results_df = pd.read_parquet(final_output_file)


# In[8]:


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


# In[9]:


#Load metadata

# Set i/o paths and files
data_dir = pathlib.Path("../0.data-download/data")

# Input files
model_input_file = pathlib.Path(f"{data_dir}/Model.csv")

model_df = pd.read_csv(model_input_file)


# In[ ]:


#check the number of ModelIDs that are in both the test RNAseq dataset and the PRISM dataset
overlap_model_ids = set(test['SampleID']).intersection(set(prism_df['ModelID']))

print(f"Number of overlapping ModelIDs: {len(overlap_model_ids)}")

# check which ModelIDs are missing in each dataframe
missing_in_prism = set(test['SampleID']) - overlap_model_ids
missing_in_test = set(prism_df['ModelID']) - overlap_model_ids

print(f"ModelIDs in test but not in prism: {missing_in_prism}")
print(f"ModelIDs in prism but not in test: {missing_in_test}")


# In[ ]:


#find the best correlated drug for each z in final_test_predictions_df
drug_df = find_drugs(final_test_predictions_df, final_test_results_df, combined_results_df)


# In[12]:


drug_results_file = output_dir / "drug_correlations.parquet"
drug_df.to_parquet(drug_results_file)


# In[13]:


drug_df.head()


# In[ ]:


# Get the list of ModelIDs that are already columns in final_test_predictions_df
existing_model_ids = [col for col in drug_df.columns if col in prism_df['ModelID'].tolist()]

# Loop through each "drug" row in final_test_predictions_df
for idx, drug_row in drug_df[drug_df['type'] == 'drug'].iterrows():

    drug_name = drug_row['drug']  # Get the drug name

    # Ensure the drug exists as a column in prism_df
    if drug_name in prism_df.columns:

        # Extract the scores for this drug
        drug_scores = prism_df.set_index('ModelID')[drug_name]

        # Find the min and max drug scores
        min_score = drug_scores.min()
        max_score = drug_scores.max()

        print(f"Drug: {drug_name} | Min score: {min_score} | Max score: {max_score}")

        # Update only the existing ModelID columns in the drug row
        for model_id in existing_model_ids:
            if model_id in drug_scores.index:
                drug_df.loc[idx, model_id] = drug_scores[model_id]
    else:
        print("No drug found")
    

