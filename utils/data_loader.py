import pathlib
from cgi import test
from copy import deepcopy
from random import sample

import numpy as np
import pandas as pd
from matplotlib import testing
from numpy import ndarray
from sklearn.preprocessing import MinMaxScaler
import torch


def load_data(data_directory, adult_or_pediatric="all", id_column="ModelID"):
    """
    Load and preprocess model and gene effect data.

    Parameters:
    - data_directory (str): Directory containing the data files.
    - adult_or_pediatric (str, optional): Filter for adult or pediatric samples. Defaults to "all".
    - id_column (str, optional): Column name for IDs in the dataframes. Defaults to "ModelID".

    Returns:
    - model_df (DataFrame): Preprocessed model dataframe.
    - effect_df (DataFrame): Preprocessed gene effect dataframe.
    """
    # Define data paths
    data_directory = "../0.data-download/data/"
    model_file = pathlib.Path(data_directory, "Model.csv")
    effect_data_file = pathlib.Path(data_directory, "CRISPRGeneEffect.csv")

    # Load data. DepMap's own CSV leaves the gene-effect file's first column
    # (ModelID) unlabeled, so name it explicitly rather than relying on the header.
    model_df = pd.read_csv(model_file)
    effect_df = pd.read_csv(effect_data_file).dropna(axis=1)
    effect_df = effect_df.rename(columns={effect_df.columns[0]: id_column})

    # Rearrange model info and gene effect dataframe indices
    model_df = model_df.sort_index(ascending=True).reset_index()
    effect_df = effect_df.set_index(id_column).sort_index(ascending=True).reset_index()

    # Search for similar IDs between effect and model data
    eff_ids = effect_df[id_column].tolist()
    mod_ids = model_df[id_column].tolist()
    eff_vs_mod_ids = set(eff_ids) & set(mod_ids)
    mod_vs_eff_ids = set(mod_ids) & set(eff_ids)

    # Subset data to only matching IDs
    model_df = model_df.loc[model_df[id_column].isin(eff_vs_mod_ids)].reset_index(drop=True)
    effect_df = effect_df.loc[effect_df[id_column].isin(mod_vs_eff_ids)]

    if adult_or_pediatric != "all":
        model_df = model_df.query("age_categories == @adult_or_pediatric").reset_index(drop=True)
        model_to_keep = model_df.reset_index(drop=True).ast.literal_eval(id_column).tolist()
        effect_df = effect_df.query(id_column + " == @samples_to_keep").reset_index(drop=True)

    model_df = model_df.set_index(id_column).reindex(index=list(mod_vs_eff_ids)).reset_index()
    effect_df = effect_df.set_index(id_column).reindex(index=list(mod_vs_eff_ids)).reset_index()

    return model_df, effect_df


def load_train_test_data(
    data_directory,
    train_file="VAE_train_df.parquet",
    test_file="VAE_test_df.parquet",
    val_file="VAE_val_df.parquet",
    train_or_test="all",
    load_gene_stats=False,
    zero_one_normalize=False,
    drop_columns=True
    ):
    """
    Load and preprocess training, testing, and validation data.

    Parameters:
    - data_directory (str): Directory containing the data files.
    - train_file (str, optional): Filename for the training data. Defaults to "VAE_train_df.parquet".
    - test_file (str, optional): Filename for the testing data. Defaults to "VAE_test_df.parquet".
    - val_file (str, optional): Filename for the validation data. Defaults to "VAE_val_df.parquet".
    - train_or_test (str, optional): Specify whether to return 'train', 'test', 'validation', or 'all' data. Defaults to "all".
    - load_gene_stats (bool, optional): If True, loads gene statistics. Defaults to False.
    - zero_one_normalize (bool, optional): If True, normalizes data to [0, 1]. Defaults to False.
    - drop_columns (bool, optional): If True, drops specified columns. Defaults to True.

    Returns:
    - DataFrame(s): Depending on train_or_test, returns one or multiple DataFrames.
    """
    # Define directory paths
    training_data_file = pathlib.Path(data_directory, train_file)
    testing_data_file = pathlib.Path(data_directory, test_file)
    validation_data_file = pathlib.Path(data_directory, val_file)

    # Load the data
    train_file = pd.read_parquet(training_data_file)
    test_file = pd.read_parquet(testing_data_file)
    val_file = pd.read_parquet(validation_data_file)

    # Load gene statistics if specified
    if load_gene_stats:
        gene_statistics_file = pathlib.Path(data_directory, "genes_variances_and_t-tests_df.parquet")
        load_gene_stats = pd.read_parquet(gene_statistics_file)
    else:
        load_gene_stats = None

    if drop_columns:
        # Prepare data for training by dropping specific columns
        train_file = train_file.drop(columns=["ModelID", "age_and_sex"])
        test_file = test_file.drop(columns=["ModelID", "age_and_sex"])
        val_file = val_file.drop(columns=["ModelID", "age_and_sex"])

        # Load gene dictionary and filter data based on QC
        gene_dict_df = pd.read_parquet("../0.data-download/data/CRISPR_gene_dictionary.parquet")
        gene_list_passed_qc = gene_dict_df.loc[gene_dict_df["qc_pass"], "dependency_column"].tolist()

        # Filter the data to include only genes that passed QC
        train_file = train_file.filter(gene_list_passed_qc, axis=1)
        test_file = test_file.filter(gene_list_passed_qc, axis=1)
        val_file = val_file.filter(gene_list_passed_qc, axis=1)

    # Normalize data if specified
    if zero_one_normalize:
        train_file = train_file.values.astype(np.float32)
        test_file = test_file.values.astype(np.float32)
        val_file = val_file.values.astype(np.float32)

        scaler = MinMaxScaler()
        train_file = scaler.fit_transform(train_file)
        test_file = scaler.transform(test_file)
        val_file = scaler.transform(val_file)

    # Return the requested data
    if train_or_test == "test":
        return test_file
    elif train_or_test == "train":
        return train_file
    elif train_or_test == "validation":
        return val_file
    elif train_or_test == "all":
        return train_file, test_file, val_file, load_gene_stats


def load_model_data(dependency_file, gene_dict_file):
    """
    Load and preprocess gene dependency data and gene dictionary.

    Parameters:
    - dependency_file (str): Path to the gene dependency data CSV.
    - gene_dict_file (str): Path to the gene dictionary parquet file.

    Returns:
    - dependency_df (DataFrame): Preprocessed gene dependency data.
    - gene_dict_df (DataFrame): Preprocessed gene dictionary.
    """
    # Load gene dependency data
    dependency_df = pd.read_csv(dependency_file)

    print(dependency_df.shape)
    dependency_df.head(3)

    # Load gene dictionary and filter by QC
    gene_dict_df = pd.read_parquet(gene_dict_file).query("qc_pass").reset_index(drop=True)
    gene_dict_df.entrez_id = gene_dict_df.entrez_id.astype(str)

    # Recode column names to entrez ids
    entrez_genes = [x[1].strip(")").strip() for x in dependency_df.iloc[:, 1:].columns.str.split("(")]
    entrez_intersection = list(set(gene_dict_df.entrez_id).intersection(set(entrez_genes)))

    gene_dict_df = gene_dict_df.set_index("entrez_id").reindex(entrez_intersection)

    # Subset dependencies to the genes that passed QC
    dependency_df.columns = ["ModelID"] + entrez_genes
    dependency_df = dependency_df.loc[:, ["ModelID"] + gene_dict_df.index.tolist()]
    dependency_df.columns = ["ModelID"] + gene_dict_df.symbol_id.tolist()
    dependency_df = dependency_df.dropna(axis="columns")

    return dependency_df, gene_dict_df
