#!/usr/bin/env python
# coding: utf-8

# ## Correlate BioBombe's latent dimensions with PRISM drug response
#
# For every latent dimension of every saved BioBombe model, correlates its
# score across DepMap cell lines with every PRISM (primary screen) drug's
# viability score for the same cell lines, plus a shuffled negative control,
# so later steps (e.g. `8.apply-biobombe-new-data/5.annotate_latent_scores.py`)
# can identify which latent dimensions track real drug sensitivity.
#
# Every saved model under `3.run-biobombe/saved_models/` is discovered
# directly (via `glob`, parsing each filename) rather than assuming one file
# per model type and dimension: that's true for PCA/ICA/NMF, but the VAE
# variants have 5 files per dimension, one per hyperparameter-search seed.
# The dependency data is reindexed to `3.run-biobombe/data/trained_gene_order.parquet`'s
# exact gene order before scoring, since every PCA/ICA/NMF component's
# weights are tied to a specific gene position. The `MinMaxScaler` is fit
# directly on this full ~1,150-model DepMap cohort and applied back to that
# same cohort, consistent with how this script uses the already-trained,
# saved models purely for inference here, not retraining.
#
# Correlations are computed with `pairwise_compare` (vectorized Pearson
# correlation via normalize-then-dot-product), batching drugs that share an
# identical set of non-missing cell lines into one call each — PRISM's
# ~4,686 drugs fall into only ~232 distinct missingness patterns — since
# missingness has to be handled before the call (the library treats any row
# with a missing feature value as entirely uncorrelatable, rather than
# dropping values pairwise per comparison).
#
# Run this script with `4.drug-dependency/` as the working directory.

# In[1]:


import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from pairwise_compare import compare_pairs, PearsonsCorrelation
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from utils import load_utils

sys.path.insert(0, "../utils/")
from data_loader import load_model_data
from model_utils import extract_latent_dims


# In[2]:


data_directory = pathlib.Path("../0.data-download/data").resolve()
dependency_file = data_directory / "CRISPRGeneEffect.csv"
gene_dict_file = data_directory / "CRISPR_gene_dictionary.parquet"
trained_gene_order_file = pathlib.Path("../3.run-biobombe/data/trained_gene_order.parquet").resolve()
model_save_dir = pathlib.Path("../3.run-biobombe/saved_models").resolve()

output_dir = pathlib.Path("results")
output_dir.mkdir(parents=True, exist_ok=True)
final_output_file = output_dir / "combined_latent_drug_correlations.parquet"


# In[3]:


dependency_df, gene_dict_df = load_model_data(dependency_file, gene_dict_file)

trained_gene_order_df = pd.read_parquet(trained_gene_order_file)
gene_order = trained_gene_order_df["symbol_id"].tolist()
dependency_df = dependency_df[["ModelID"] + gene_order]

scaler = MinMaxScaler()
dependency_df[gene_order] = scaler.fit_transform(dependency_df[gene_order])
print(f"dependency_df: {dependency_df.shape}")


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

prism_df.reset_index(inplace=True)
prism_df.rename(columns={'index': 'ModelID'}, inplace=True)
prism_df.set_index('ModelID', inplace=True)
print(f"prism_df: {prism_df.shape}")


# In[5]:


# A shuffled copy of every drug column, as a negative control baseline.
prism_df_shuffled = prism_df.copy()
for drug_col in prism_df_shuffled.columns:
    prism_df_shuffled[drug_col] = prism_df_shuffled[drug_col].sample(frac=1, random_state=None).values


# In[6]:


# Load metadata and the train/test/val tensor loader the VAE variants need
metadata_df_dir = pathlib.Path("../1.data-exploration/data/metadata_df.parquet")
metadata = pd.read_parquet(metadata_df_dir)

train_and_test_subbed_dir = pathlib.Path("../1.data-exploration/data/train_and_test_subbed.parquet")
train_and_test_subbed = pd.read_parquet(train_and_test_subbed_dir)
train_test_tensor = torch.tensor(train_and_test_subbed.to_numpy(), dtype=torch.float32)
train_and_test_subbed_loader = DataLoader(TensorDataset(train_test_tensor), batch_size=32, shuffle=False)


# In[7]:


# Discover every saved model file directly rather than guessing a filename -
# see the module docstring above for why the original filename pattern never
# matched anything, and why one file per (model, dimension) can't be assumed.
all_model_files = sorted(model_save_dir.glob("*.joblib"))
print(f"Found {len(all_model_files)} saved models")

latent_frames = []
skipped_models = []
for model_file in all_model_files:
    parts = model_file.stem.split("_")
    model_name, dim, init = parts[0], int(parts[3]), int(parts[7])
    try:
        model = joblib.load(model_file)
        latent_df = extract_latent_dims(model_name, model, dependency_df, train_and_test_subbed_loader, metadata)
    except Exception as e:
        skipped_models.append((model_file.name, str(e)))
        continue

    latent_df = latent_df.set_index("ModelID")
    # Columns come back as plain ints for PCA/ICA/NMF or "z_i"-named for the
    # VAE variants (utils/model_utils.py); normalize both to one parseable
    # "model|dim|init|z" key so results can be split back into
    # model/full_model_z/init/z columns after the correlation step below.
    latent_df.columns = [f"{model_name}|{dim}|{init}|{str(c).replace('z_', '')}" for c in latent_df.columns]
    latent_frames.append(latent_df)

print(f"Scored {len(latent_frames)} / {len(all_model_files)} models")
if skipped_models:
    print(f"Skipped {len(skipped_models)}:")
    for name, reason in skipped_models[:10]:
        print(f"  {name}: {reason}")

combined_latent_df = pd.concat(latent_frames, axis=1)
print(f"combined_latent_df: {combined_latent_df.shape}")


# In[8]:


common_model_ids = combined_latent_df.index.intersection(prism_df.index)
print(f"Common ModelIDs between latent scores and PRISM: {len(common_model_ids)}")

latent_common = combined_latent_df.loc[common_model_ids]
latent_common = latent_common.loc[:, latent_common.var(skipna=True) != 0]
prism_common = prism_df.loc[common_model_ids]
prism_shuffled_common = prism_df_shuffled.loc[common_model_ids]
print(f"Latent dimensions with non-zero variance: {latent_common.shape[1]}")


# In[9]:


def correlate_against_drugs(latent_df, drug_df, shuffle_flag):
    """
    Pearson correlation (and its p-value) between every latent-dim column and
    every drug column, for the ModelIDs both have a value for.

    Missingness is handled before calling pairwise_compare, not by it:
    pairwise_compare treats any row with a missing feature value as entirely
    uncorrelatable (NaN for every comparison involving that row), rather than
    dropping missing values pairwise per comparison the way the original
    per-pair loop did. PRISM's missingness is concentrated on the drug side
    (checked directly: the latent scores have none), and drugs that share an
    identical set of non-missing cell lines can be batched into one call
    together - only ~232 distinct patterns exist across PRISM's ~4,686 drugs,
    so this is ~232 calls rather than 4,686, each handling every latent
    dimension against every drug in its batch in one vectorized step.
    """
    patterns = drug_df.notna().apply(lambda col: frozenset(col[col].index), axis=0)
    groups = patterns.groupby(patterns).groups

    results = []
    for pattern, drugs in groups.items():
        valid_ids = [x for x in latent_df.index if x in pattern]
        if len(valid_ids) < 3:
            continue
        sub_latent = latent_df.loc[valid_ids].dropna(axis=1)
        if sub_latent.shape[1] == 0:
            continue
        sub_drugs = drug_df.loc[valid_ids, list(drugs)]

        values = np.vstack([sub_latent.T.values, sub_drugs.T.values])
        entity_types = ["latent_dim"] * sub_latent.shape[1] + ["drug"] * sub_drugs.shape[1]
        entity_ids = sub_latent.columns.tolist() + list(drugs)
        tidy_df = pd.DataFrame(values, columns=[str(x) for x in valid_ids])
        tidy_df.insert(0, "entity_type", entity_types)
        tidy_df.insert(1, "entity_id", entity_ids)

        result = compare_pairs(
            tidy_df, different=["entity_type", "entity_id"], features=[str(x) for x in valid_ids],
            comparator=PearsonsCorrelation(name="pearson_correlation"),
        )
        result["n"] = len(valid_ids)
        results.append(result)

    combined = pd.concat(results, ignore_index=True)

    is_latent0 = combined["entity_type__group0"] == "latent_dim"
    fit_key = np.where(is_latent0, combined["entity_id__group0"], combined["entity_id__group1"])
    drug = np.where(is_latent0, combined["entity_id__group1"], combined["entity_id__group0"])
    fit_parts = pd.DataFrame([str(k).split("|") for k in fit_key], columns=["model", "full_model_z", "init", "z"])

    r, n = combined["pearson_correlation"].to_numpy(), combined["n"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = r * np.sqrt((n - 2) / (1 - r ** 2))
    p_value = 2 * stats.t.sf(np.abs(t_stat), df=n - 2)

    return pd.DataFrame({
        "model": fit_parts["model"],
        "full_model_z": fit_parts["full_model_z"].astype(int),
        "init": fit_parts["init"].astype(int),
        "z": fit_parts["z"].astype(int),
        "drug": drug,
        "pearson_correlation": r,
        "p_value": p_value,
        "shuffled": shuffle_flag,
    })


real_results_df = correlate_against_drugs(latent_common, prism_common, shuffle_flag=False)
print(f"Real correlations: {real_results_df.shape}")

shuffled_results_df = correlate_against_drugs(latent_common, prism_shuffled_common, shuffle_flag=True)
print(f"Shuffled (negative control) correlations: {shuffled_results_df.shape}")

combined_results_df = pd.concat([real_results_df, shuffled_results_df], ignore_index=True)


# In[10]:


# Merge in drug display names and metadata, and save *that* - not the
# pre-merge table - since downstream consumers (e.g. module 8's
# 5.annotate_latent_scores.py) key off the "name" column this merge adds,
# not the raw PRISM column identifier.
prism_trt_df_filtered = prism_trt_df[['column_name', 'name', 'moa', 'target', 'indication', 'phase']]
combined_results_df = pd.merge(
    combined_results_df, prism_trt_df_filtered, how='left', left_on='drug', right_on='column_name',
).drop(columns=['column_name'])

combined_results_df.to_parquet(final_output_file)
print(f"Saved combined results to {final_output_file}")

significant_corr_df = combined_results_df[
    (~combined_results_df["shuffled"]) & (combined_results_df['pearson_correlation'].abs() > 0.1)
]
print(f"Significant (non-shuffled, |r| > 0.1) correlations: {significant_corr_df.shape[0]} / {(~combined_results_df['shuffled']).sum()}")


# In[11]:


combined_results_df[~combined_results_df["shuffled"]].sort_values(by='pearson_correlation', key=abs, ascending=False).head(50)
