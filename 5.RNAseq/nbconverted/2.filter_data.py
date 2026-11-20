#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pathlib
import sys
from sklearn.preprocessing import StandardScaler

script_directory = pathlib.Path("../utils/").resolve()
sys.path.insert(0, str(script_directory))
from data_loader import load_train_test_data


# In[2]:


def split_rna_seq(rna_data, split_sample_ids, split_name, output_dir, top_genes):
    """
    Filters RNASeq data for a specific split (train, test, val) using precomputed top genes.
    Args:
        rna_data (pd.DataFrame): Full RNASeq data with 'SampleID' and gene columns.
        split_sample_ids (list): List of SampleIDs for the current split.
        split_name (str): Name of the split (e.g., 'train', 'test', 'val').
        output_dir (str): Directory to save the filtered split.
        top_genes (list): List of top genes selected globally by variance.
    Returns:
        pd.DataFrame: Filtered RNASeq data for the given split.
    """
    print(f"\nProcessing {split_name} split...")

    # Align RNASeq data to the current split's SampleIDs
    aligned_rna_data = rna_data[rna_data['SampleID'].isin(split_sample_ids)].copy()

    # Filter to include only the top genes
    filtered_rna_split = aligned_rna_data[['SampleID'] + list(top_genes)]

    # Save the filtered split
    if output_dir:
        output_path = pathlib.Path(output_dir) / f"RNASeq_{split_name}_filtered.parquet"
        filtered_rna_split.to_parquet(output_path, index=False)
        print(f"Saved {split_name} split to {output_path}")
        print(filtered_rna_split.shape)

    return filtered_rna_split


# In[3]:


def zscore_normalize_split(rna_split_data, split_name, output_dir=None):
    """
    Z-score normalizes RNASeq data for a given split.

    Args:
        rna_split_data (pd.DataFrame): RNASeq data for the split with 'SampleID' and gene columns.
        split_name (str): Name of the split (e.g., 'train', 'test', 'val').
        output_dir (str, optional): Directory to save the z-scored data.

    Returns:
        pd.DataFrame: Z-score normalized RNASeq data for the split.
    """
    print(f"\nZ-Score Normalizing {split_name} split...")
    
    # Separate SampleID and gene data
    sample_ids = rna_split_data['SampleID']
    gene_data = rna_split_data.drop(columns=['SampleID'])

    # Z-score normalization
    scaler = StandardScaler()
    zscored_data = pd.DataFrame(
        scaler.fit_transform(gene_data),
        columns=gene_data.columns,
        index=gene_data.index
    )

    # Add SampleID back to the z-scored data
    zscored_data.insert(0, 'SampleID', sample_ids.values)

    # Save z-scored data if output directory is specified
    if output_dir:
        output_path = pathlib.Path(output_dir) / f"RNASeq_{split_name}_zscored.parquet"
        zscored_data.to_parquet(output_path, index=False)
        print(f"Saved {split_name} split to {output_path}")

    print(zscored_data.head())
    print(zscored_data.shape)
    return zscored_data


# In[4]:


# Load RNAseq data
rna_seq_data = pd.read_parquet("../5.RNAseq/data/RNASeq.parquet")  # Replace with your file path
rna_seq_data.rename(columns={'Unnamed: 0': 'SampleID'}, inplace=True)

# Load gene dependency data
data_directory = pathlib.Path("../1.data-exploration/data").resolve()

train_df, test_df, val_df, load_gene_stats = load_train_test_data(
    data_directory, train_or_test="all", load_gene_stats=True, zero_one_normalize=False, drop_columns=False
)

train_data = pd.DataFrame(train_df)
test_data = pd.DataFrame(test_df)
val_data = pd.DataFrame(val_df)


# In[5]:


rna_seq_data.head()


# In[6]:


# Set the number of top genes by variance to select
top_n_genes = 2000

# Identify common genes between RNASeq and dependency data
dependency_genes = set(load_gene_stats['gene_ID'].unique())  # Extract gene names from dependency data
rna_genes = set(rna_seq_data.columns) - {'SampleID'}         # Extract gene columns (excluding SampleID)
common_genes = dependency_genes.intersection(rna_genes)

# Filter RNASeq data to include only common genes
rna_common_data = rna_seq_data[['SampleID'] + list(common_genes)]

# Calculate gene variances globally
gene_variances = rna_common_data.drop(columns=['SampleID']).var(axis=0)

# Select top N genes by variance
top_genes = gene_variances.nlargest(top_n_genes).index
print(f"Top {top_n_genes} genes selected")


# In[7]:


# Define splits
splits = [
    {"name": "train", "sample_ids": train_df['ModelID'].tolist()},
    {"name": "test", "sample_ids": test_df['ModelID'].tolist()},
    {"name": "val", "sample_ids": val_df['ModelID'].tolist()}
]

# Output directory
output_directory = "../5.RNAseq/data"

# Process and save RNASeq data for each split
split_rna_results = {}
for split in splits:
    split_rna_results[split["name"]] = split_rna_seq(
        rna_data=rna_common_data,
        split_sample_ids=split["sample_ids"],
        split_name=split["name"],
        output_dir=output_directory,
        top_genes=top_genes
    )


# In[8]:


#z score normalize 
# Define output directory
output_directory = "../5.RNAseq/data"

# Perform Z-score normalization on already split data
zscored_train = zscore_normalize_split(split_rna_results["train"], "train", output_dir=output_directory)
zscored_test = zscore_normalize_split(split_rna_results["test"], "test", output_dir=output_directory)
zscored_val = zscore_normalize_split(split_rna_results["val"], "val", output_dir=output_directory)

