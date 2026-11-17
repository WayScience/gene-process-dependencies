#!/usr/bin/env python
# coding: utf-8

# ## GSEA Analysis Pipeline for Dimensionality Reduction Models

# This script performs Gene Set Enrichment Analysis (GSEA) on weight matrices  extracted from various dimensionality reduction models (PCA, ICA, NMF, VanillaVAE,  BetaVAE, and BetaTCVAE). It iterates over different latent dimensions and model types, extracts the weight matrices, and computes GSEA scores. Runs once per library in GSEA_LIBRARIES below (currently CORUM and Reactome), each to its own output file, for downstream analysis.

# In[1]:


import joblib
import pandas as pd
import blitzgsea as blitz
import random
import pathlib
import sys

script_directory = pathlib.Path("../utils/").resolve()
sys.path.insert(0, str(script_directory))
from data_loader import load_train_test_data, load_model_data
from model_utils import extract_weights


# In[2]:


# Load data
data_directory = pathlib.Path("../0.data-download/data").resolve()
train_test_data_directory = pathlib.Path("../1.data-exploration/data").resolve()

train_df, test_df, val_df, load_gene_stats = load_train_test_data(
    train_test_data_directory, train_or_test="all", load_gene_stats=True, zero_one_normalize=True
)
train_data = pd.DataFrame(train_df)

dependency_file = pathlib.Path(f"{data_directory}/CRISPRGeneEffect.csv").resolve()
gene_dict_file = pathlib.Path(f"{data_directory}/CRISPR_gene_dictionary.parquet").resolve()
dependency_df, gene_dict_df= load_model_data(dependency_file, gene_dict_file)
gene_dict_df = pd.DataFrame(gene_dict_df)


# In[3]:


#Load weight data for VAEs
weight_df = load_train_test_data(
    train_test_data_directory, train_or_test="train"
)

gene_list_passed_qc = gene_dict_df.loc[
    gene_dict_df["qc_pass"], "dependency_column"
].tolist()

weight_data = weight_df.filter(gene_list_passed_qc, axis=1)
weight_data.head()


# In[4]:


def perform_gsea(weights_df: pd.DataFrame, model_name: str, num_components: int, init: int, modelseed:int, lib: str = "CORUM") -> pd.DataFrame:
    """
    Performs Gene Set Enrichment Analysis (GSEA) for a given weight matrix.

    Args:
        weights_df (pd.DataFrame): DataFrame containing genes and their associated weights.
        model_name (str): Name of the model being analyzed.
        num_components (int): Number of components used in the model.
        lib (str): Name of the GSEA library (default: 'CORUM').

    Returns:
        pd.DataFrame: Results of GSEA with columns for pathway, enrichment scores, and other metrics.
    """

    library = blitz.enrichr.get_library(lib)
    random.seed(0)
    seed = random.random()
    gsea_results = []
    for col in weights_df.columns[1:]:  # Skip 'genes' column
        gene_signature = weights_df[['genes', col]]
        if gene_signature.shape[0] > 0:
            try:
                # processes=1: blitz.gsea() otherwise spins up its own
                # multiprocessing.Pool(processes=4) on every single call (once
                # per latent dim, per model - thousands of times across a full
                # run). Repeated Pool creation/teardown at that volume is what
                # deadlocked a prior run partway through, with zero progress
                # for 8+ hours. Single-process is slower per call but reliable.
                gsea_result = blitz.gsea(gene_signature, library, seed=seed, processes=1)
                gsea_result = gsea_result.reset_index()
                for _, pathway_result in gsea_result.iterrows():
                    result_row = {
                        "z": int(col),
                        "full_model_z": num_components,
                        "init" : int(init),
                        "modelseed" : int(modelseed),
                        "model": str(model_name),
                        "reactome_pathway": str(pathway_result['Term']),
                        "gsea_es_score": pathway_result['es'],
                        "nes_score": pathway_result['nes'],
                        "p_value": pathway_result['pval'],
                        "shuffled": False
                    }
                    gsea_results.append(result_row)

            except ZeroDivisionError:
                print(f"Skipping GSEA for {col} due to zero division error.")

    gsea_results_df = pd.DataFrame(gsea_results)
    return gsea_results_df

# Define the location of the saved models and output directory for GSEA results
model_save_dir = pathlib.Path("saved_models")
output_dir = pathlib.Path("gsea_results")
output_dir.mkdir(parents=True, exist_ok=True)

# Each library gets its own output file and its own independent, resumable
# pass over saved_models/ - rerunning one library later doesn't touch the
# other's file or "already processed" state. CORUM keeps its original,
# unsuffixed filename for backward compatibility with every existing
# downstream consumer of this file (e.g. module 8's annotation script).
GSEA_LIBRARIES = {
    "CORUM": {"enrichr_name": "CORUM", "output_file": "combined_z_matrix_gsea_results.parquet"},
    "Reactome": {"enrichr_name": "Reactome_Pathways_2024", "output_file": "combined_z_matrix_gsea_results_reactome.parquet"},
}

for library_key, library_config in GSEA_LIBRARIES.items():
    final_output_file = output_dir / library_config["output_file"]
    try:
        combined_results_df = pd.read_parquet(final_output_file)
        print(f"{library_key}: loaded existing results from {final_output_file}")
    except FileNotFoundError:
        # If the file doesn't exist, initialize an empty DataFrame
        combined_results_df = pd.DataFrame()
        print(f"{library_key}: no existing file found. Initialized empty DataFrame.")

    # Iterate over all files in the saved_models directory
    for model_file in model_save_dir.glob("*.joblib"):
        # Extract model name and number of components from the filename
        model_file_name = model_file.stem
        try:
            # Assuming the filename format includes model_name, num_components, and potentially a seed
            # Example: "BetaVAE_100_components_seed42_model.joblib"
            parts = model_file_name.split("_")
            model_name = parts[0]  # First part is the model name
            num_components = int(parts[3])  # Second part should indicate the number of components
            init = int(parts[7])
            seed = int(parts[9])
        except (IndexError, ValueError) as e:
            print(f"Skipping file {model_file} due to unexpected filename format.")
            continue
        # Check if this model, latent dimension, and initialization have already been processed
        if not combined_results_df.empty:
            if ((combined_results_df['model'] == model_name) &
                (combined_results_df['init'] == init) &
                (combined_results_df['modelseed'] == seed) &
                (combined_results_df['full_model_z'] == num_components)).any():
                print(f"{library_key}: skipping {model_name} init {init} with {num_components} dimensions as it is already processed.")
                continue

        # Load the model
        print(f"{library_key}: loading model from {model_file}")
        try:
            model = joblib.load(model_file)
        except Exception as e:
            print(f"Failed to load model from {model_file}: {e}")
            continue

        # Extract the weight matrix
        weight_matrix_df = extract_weights(model, model_name, weight_data, dependency_df)

        # Perform GSEA
        gsea_results_df = perform_gsea(
            weight_matrix_df, model_name, num_components, init, seed,
            lib=library_config["enrichr_name"],
        )
        combined_results_df = pd.concat([combined_results_df, gsea_results_df], ignore_index=True)

    # Save this library's results before moving on to the next library, so a
    # crash partway through the next one can't lose what's already done here.
    combined_results_df.to_parquet(final_output_file, index=False)
    print(f"{library_key}: saved combined GSEA results to {final_output_file}")


# In[5]:


# Quick inline peek at the strongest hits - reflects whichever library ran
# last in GSEA_LIBRARIES above, not a combined view of all libraries.
combined_results_df.sort_values(by='gsea_es_score', key=abs, ascending = False).head(50)
