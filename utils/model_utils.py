import pathlib
import sys
import numpy as np
import pandas as pd

script_directory = pathlib.Path("../2.prototype-VAE-models/utils/").resolve()
sys.path.insert(0, str(script_directory))
from betatcvae import tc_extract_latent_dimensions
from betavae import extract_latent_dimensions
from vanillavae import vvae_extract_latent_dimensions

script_directory = pathlib.Path("../2.prototype-VAE-models/utils/").resolve()
sys.path.insert(0, str(script_directory))
from betavae import weights
from betatcvae import tc_weights
from vanillavae import vanilla_weights

# Function to extract latent dimensions for all models
def extract_latent_dims(model_name, model, dependency_df, train_and_test_subbed_loader, metadata):
    """
    Extracts latent dimensions from a given model and returns a DataFrame containing the latent representations.

    Args:
        model_name (str): The name of the model ('pca', 'ica', 'nmf', 'betavae', 'betatcvae', 'vanillavae').
        model (object): The trained dimensionality reduction model.
        dependency_df (pd.DataFrame): DataFrame containing input features for transformation.
        train_and_test_subbed_loader (object): Data loader for VAE models.
        metadata (object): Metadata required for VAE models.

    Returns:
        pd.DataFrame: A DataFrame containing latent dimensions with ModelID if applicable.
    """
    if model_name in ["pca", "ica", "nmf"]:
        # Extract latent dimensions for PCA, ICA, or NMF models
        numerical_columns = dependency_df.select_dtypes(include=[np.number]).columns
        numerical_df = dependency_df[numerical_columns].astype(np.float32)  # Ensure float32 type
        
        # Transform data into the reduced space
        latent_matrix = model.transform(numerical_df)
        
        # Create a DataFrame with the latent dimensions
        latent_df = pd.DataFrame(
            latent_matrix,
            columns=[f'z_{i}' for i in range(latent_matrix.shape[1])]
        )
        
        # Add ModelID column if present in the original DataFrame
        if 'ModelID' in dependency_df.columns:
            latent_df.insert(0, 'ModelID', dependency_df['ModelID'].values)
            
    elif model_name == "betavae":
        latent_df = extract_latent_dimensions(model, train_and_test_subbed_loader, metadata)
    elif model_name == "betatcvae":
        latent_df = tc_extract_latent_dimensions(model, train_and_test_subbed_loader, metadata)
    elif model_name == "vanillavae":
        latent_df = vvae_extract_latent_dimensions(model, train_and_test_subbed_loader, metadata)
    
    return latent_df

def extract_weights(
    model: object, 
    model_name: str, 
    weight_data: pd.DataFrame = None,
    dependency_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Extracts the weight matrix from a trained model, organizing genes as rows and components as columns.

    Args:
        model (object): A trained dimensionality reduction model (PCA, ICA, NMF, or VAE).
        model_name (str): The name of the model ('pca', 'ica', 'nmf', 'betavae', 'betatcvae', 'vanillavae').
        weight_data (pd.DataFrame, optional): Data required for weight extraction in VAE models.
        dependency_df (pd.DataFrame, optional): DataFrame containing feature names for PCA, ICA, or NMF models.

    Returns:
        pd.DataFrame: A DataFrame containing model weights with genes as rows and components as columns.
    """
    if model_name in ["pca", "ica", "nmf"]:
        weights_df = pd.DataFrame(
            model.components_,
            columns=dependency_df.drop(columns=["ModelID"]).columns.tolist()
        ).transpose()
        
        # Reset index to include gene names as a column
        weights_df = weights_df.reset_index()
        weights_df.rename(columns={"index": "genes"}, inplace=True)
    elif model_name == "betavae":
        weights_df = weights(model, weight_data)
    elif model_name == "betatcvae":
        weights_df = tc_weights(model, weight_data)
    elif model_name == "vanillavae":
        weights_df = vanilla_weights(model, weight_data)
    
    print(weights_df)
    
    # Ensure there are no duplicate columns
    weights_df = weights_df.loc[:, ~weights_df.columns.duplicated()]
    
    # Rename the first column to 'genes' if it is not already named so
    if weights_df.columns[0] != "genes":
        weights_df.rename(columns={weights_df.columns[0]: "genes"}, inplace=True)
        weights_df = weights_df.reset_index(drop=True)
    
    return weights_df