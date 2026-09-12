#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import sys

import joblib
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "../utils/")
from data_loader import load_model_data
from model_utils import extract_latent_dims

sys.path.insert(0, "./utils")
from optimizing_utils import model_training


# In[2]:


def align_dataframes(
    train: pd.DataFrame, 
    test: pd.DataFrame, 
    val: pd.DataFrame, 
    latent_df: pd.DataFrame
    ):
    """
    Aligns gene expression datasets (train, test, val) with latent representations from latent_df.

    This function merges latent dimension data with RNASeq gene expression datasets based on `SampleID`,
    separates features from target latent dimensions, and removes latent dimensions that contain only zeros.
    Additionally, it combines validation and test features/targets for final evaluation.

    Parameters:
    - train (pd.DataFrame): Training dataset containing RNASeq gene expression data.
    - test (pd.DataFrame): Test dataset containing RNASeq gene expression data.
    - val (pd.DataFrame): Validation dataset containing RNASeq gene expression data.
    - latent_df (pd.DataFrame): Dataframe containing latent dimensions with `ModelID` as the key.

    Returns:
    - X_train_features (pd.DataFrame): Training dataset features (gene expression).
    - X_val_features (pd.DataFrame): Validation dataset features (gene expression).
    - X_test_features (pd.DataFrame): Combined validation & test dataset features (gene expression).
    - y_train (pd.DataFrame): Training dataset target (latent dimensions).
    - y_val (pd.DataFrame): Validation dataset target (latent dimensions).
    - y_test (pd.DataFrame): Combined validation & test dataset target (latent dimensions).
    """
    
    # Identify latent_columns as columns with numeric names
    latent_columns = [col for col in latent_df.columns if col != 'ModelID']

    train = train.merge(latent_df, left_on='SampleID', right_on='ModelID', how='left')
    val = val.merge(latent_df, left_on='SampleID', right_on='ModelID', how='left')
    test = test.merge(latent_df, left_on='SampleID', right_on='ModelID', how='left')

    # Separate the features (RNASeq gene expression) and target (latent dimensions)
    X_train_features = train.drop(columns=['SampleID', 'ModelID'] + latent_columns)
    X_val_features = val.drop(columns=['SampleID', 'ModelID'] + latent_columns)
    X_test_features = test.drop(columns=['SampleID', 'ModelID'] + latent_columns)

    # Set targets as the latent dimensions corresponding to each sample
    y_train = train[latent_columns]
    y_val = val[latent_columns]
    y_test = test[latent_columns]

    # Identify zero-only columns in y_train
    zero_columns = y_train.columns[(y_train == 0).all()]

    # Drop these columns from all y datasets
    y_train = y_train.drop(columns=zero_columns)
    y_val = y_val.drop(columns=zero_columns)
    y_test = y_test.drop(columns=zero_columns)

    y_test = pd.concat([y_val, y_test], axis=0)

    return X_train_features, X_val_features, X_test_features, y_train, y_val, y_test
    


# In[3]:


# Load pre-split RNASeq data
train = pd.read_parquet("./data/RNASeq_train_zscored.parquet")
val = pd.read_parquet("./data/RNASeq_val_zscored.parquet")
test = pd.read_parquet("./data/RNASeq_test_zscored.parquet")


# In[4]:


data_directory = pathlib.Path("../0.data-download/data").resolve()
dependency_file = pathlib.Path(f"{data_directory}/CRISPRGeneEffect.csv").resolve()
gene_dict_file = pathlib.Path(f"{data_directory}/CRISPR_gene_dictionary.parquet").resolve()


# In[5]:


# Load metadata
metadata_df_dir = pathlib.Path("../1.data-exploration/data/metadata_df.parquet").resolve()
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


# In[6]:


train_and_test_subbed_dir = pathlib.Path("../1.data-exploration/data/train_and_test_subbed.parquet").resolve()
train_and_test_subbed = pd.read_parquet(train_and_test_subbed_dir)

# Convert DataFrame to NumPy and then Tensor
train_test_array = train_and_test_subbed.to_numpy()
train_test_tensor = torch.tensor(train_test_array, dtype=torch.float32)

#Create TensorDataset and DataLoader
tensor_dataset = TensorDataset(train_test_tensor)
train_and_test_subbed_loader = DataLoader(tensor_dataset, batch_size=32, shuffle=False)


# In[7]:


# Define the location of the saved models and output directory for results
model_save_dir = pathlib.Path("../3.run-biobombe/saved_models").resolve()
output_dir = pathlib.Path("results").resolve()
output_dir.mkdir(parents=True, exist_ok=True)

# File to store the combined correlation results
final_test_results_file = output_dir / "test_r2.parquet"
final_test_predictions_file = output_dir /  "test_preds.parquet"
final_test_actual_file = output_dir /  "pred_vs_actual.parquet"


# In[8]:


# Initialize lists to hold test results and predictions
all_test_results = []
all_test_predictions = []

# Load existing results if available
if pathlib.Path(final_test_results_file).exists():
    final_test_results_df = pd.read_parquet(final_test_results_file)
    final_test_predictions_df = pd.read_parquet(final_test_predictions_file)
    all_test_results.append(final_test_results_df)
    all_test_predictions.append(final_test_predictions_df)
    print(f"Loaded existing test results from {final_test_results_file}")
    print(f"Existing test results loaded: {final_test_results_df.shape}")
    print(f"Existing test predictions loaded: {final_test_predictions_df.shape}")
else:
    final_test_results_df = pd.DataFrame(columns=['model','latent_dimension', 'z_dimension', 'init', 'R2_score'])
    final_test_predictions_df = pd.DataFrame(columns=['model', 'latent_dimension', 'z_dimension', 'init', 'type']) 
    print("Starting with empty DataFrames.")

# Iterate over all files in the saved_models directory
for model_file in model_save_dir.glob("*.joblib"):
    model_file_name = model_file.stem
    try:
        parts = model_file_name.split("_")
        model_name = parts[0]
        num_components = int(parts[3])
        init = int(parts[7])
        seed = int(parts[9])
    except (IndexError, ValueError):
        print(f"Skipping file {model_file} due to unexpected filename format.")
        continue

    expected_z_dimensions = {int(f"{i}") for i in range(num_components)}

    # Get already processed z_dimensions
    processed_z_dimensions = set(final_test_predictions_df[
        (final_test_predictions_df['model'] == model_name) &
        (final_test_predictions_df['init'] == init) &
        (final_test_predictions_df['latent_dimension'] == num_components)
    ]['z_dimension'].unique())

    processed_z_dimensions = {int(col.lstrip('z_')) for col in processed_z_dimensions}

    # Determine which z_dimensions still need processing
    missing_z_dimensions = expected_z_dimensions - processed_z_dimensions

    if not missing_z_dimensions:
        print(f"Skipping {model_name} init {init} with {num_components} dimensions. All z_dimensions are processed.")
        continue
    else:
        print(f"Processing missing z_dimensions: {missing_z_dimensions}")

    # Load the model
    print(f"Loading model from {model_file}")
    try:
        model = joblib.load(model_file)
    except Exception as e:
        print(f"Failed to load model from {model_file}: {e}")
        continue

    # Extract latent dimensions
    latent_df = extract_latent_dims(model_name, model, dependency_df, train_and_test_subbed_loader, metadata)

    # Ensure missing_z_dimensions is of type set of integers
    missing_z_dimensions = {int(dim) for dim in missing_z_dimensions}  # Make sure the missing_z_dimensions are integers

    # Ensure latent_df.columns are integers for comparison
    latent_df.columns = [int(col) if isinstance(col, str) and col.isdigit() else col for col in latent_df.columns]

    # Filter latent_df to only include missing z_dimensions (ensure the column names are integers)
    latent_df = latent_df[[col for col in latent_df.columns if (isinstance(col, int) and col in missing_z_dimensions) or col == 'ModelID']]

    # Align dataframes for training and testing
    x_train, x_val, x_test, y_train, y_val, y_test = align_dataframes(train, test, val, latent_df)

    # Perform model training and prediction only for missing z_dimensions
    test_results_df, test_predictions_df = model_training(
        x_train, x_val, x_test, y_train, y_val, y_test, model_name, num_components, init, seed
    )

    # Add metadata columns for test results and predictions
    test_results_df = test_results_df.assign(init=init, seed=seed)
    test_predictions_df = test_predictions_df.assign(init=init, seed=seed)

    # Append only new results to avoid duplication
    all_test_results.append(test_results_df)
    all_test_predictions.append(test_predictions_df)

    # Incrementally update final dataframes
    final_test_results_df = pd.concat(all_test_results, ignore_index=True).drop_duplicates()
    final_test_predictions_df = pd.concat(all_test_predictions, ignore_index=True).drop_duplicates()

    # Clean up z_dimension naming
    final_test_results_df['z_dimension'] = final_test_results_df['z_dimension'].astype(str)
    final_test_predictions_df['z_dimension'] = final_test_predictions_df['z_dimension'].astype(str)
    final_test_results_df['z_dimension'] = final_test_results_df['z_dimension'].str.replace(r'^z_z_', 'z_', regex=True)
    final_test_predictions_df['z_dimension'] = final_test_predictions_df['z_dimension'].str.replace(r'^z_z_', 'z_', regex=True)

    # Save results and predictions
    final_test_results_df.to_parquet(final_test_results_file)
    final_test_predictions_df.to_parquet(final_test_predictions_file)

print(f"Saved test results to {final_test_results_file}")
print(f"Saved test predictions to {final_test_predictions_file}")


# In[9]:


# Sort test results for R² in descending order and get the top 10 latent dimensions
top_20_r2 = final_test_results_df.sort_values(by='R2_score', ascending=False).head(20)
print("Top 20 R² scores:")
print(top_20_r2)


# In[10]:


print(final_test_predictions_df.shape)
final_test_predictions_df.head()

