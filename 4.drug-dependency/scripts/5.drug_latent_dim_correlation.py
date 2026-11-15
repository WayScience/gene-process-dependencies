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
# weights are tied to a specific gene position.
#
# The scaler is the same persisted, train-split-fit bundle
# `8.apply-biobombe-new-data/utils/data_prep.py` maintains, not a fresh fit on
# this script's own full ~1,150-model cohort: the saved PCA/ICA/NMF models'
# weights are tied to the specific per-gene scale they were trained under
# (the 670-model train split), and a full-cohort fit shifts that scale for
# roughly 40% of genes, which changes what the models' latent scores actually
# mean, not just their numeric range. Every model type - including the VAE
# variants, which this script scores directly against this same scaled
# ~1,150-model cohort rather than the narrower 958-model population
# `1.data-exploration/data/train_and_test_subbed.parquet` represents - sees
# the identical scaled input and population, so no latent column is ever
# missing a value for a cell line another model type has one for.
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
import re
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from pairwise_compare import compare_pairs, PearsonsCorrelation
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset

from utils import load_utils

sys.path.insert(0, "../utils/")
from data_loader import load_model_data
from model_utils import extract_latent_dims

sys.path.insert(0, "../8.apply-biobombe-new-data/utils/")
from data_prep import get_or_create_scaler


# In[2]:


data_directory = pathlib.Path("../0.data-download/data").resolve()
dependency_file = data_directory / "CRISPRGeneEffect.csv"
gene_dict_file = data_directory / "CRISPR_gene_dictionary.parquet"
trained_gene_order_file = pathlib.Path("../3.run-biobombe/data/trained_gene_order.parquet").resolve()
train_test_data_directory = pathlib.Path("../1.data-exploration/data").resolve()
model_save_dir = pathlib.Path("../3.run-biobombe/saved_models").resolve()

output_dir = pathlib.Path("results")
output_dir.mkdir(parents=True, exist_ok=True)
final_output_file = output_dir / "combined_latent_drug_correlations.parquet"

# The one scaler artifact module 8 already maintains, fit on the correct
# 670-model train split and self-validating (regenerates if its fit
# population or gene schema ever falls out of sync with the current code) -
# reused here rather than duplicating that fitting logic.
scaler_path = pathlib.Path("../8.apply-biobombe-new-data/results/depmap_gene_scaler.joblib").resolve()


# In[3]:


dependency_df, gene_dict_df = load_model_data(dependency_file, gene_dict_file)

trained_gene_order_df = pd.read_parquet(trained_gene_order_file)
# dependency_column ("SYMBOL (ENTREZID)") is the format get_or_create_scaler's
# bundle stores gene_order in; symbol_id (bare symbol) is what load_model_data
# returns as dependency_df's actual column names. Both are the same row order
# in trained_gene_order_df (one row per gene), so symbol_id can be used to
# select/reindex dependency_df's columns while dependency_column is used only
# to match the scaler bundle's own schema.
dependency_column_order = trained_gene_order_df["dependency_column"].tolist()
gene_order = trained_gene_order_df["symbol_id"].tolist()
dependency_df = dependency_df[["ModelID"] + gene_order]

scaler_bundle = get_or_create_scaler(train_test_data_directory, dependency_column_order, scaler_path)
dependency_df[gene_order] = scaler_bundle["scaler"].transform(dependency_df[gene_order].values.astype(np.float32))
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


# One scaled tensor and one ModelID frame, built from this same scaled
# dependency_df, shared by every model type - including the VAE variants,
# which otherwise have no direct use for dependency_df (their own
# extract_latent_dimensions functions only need a DataLoader and a ModelID
# frame in the same row order). Scoring every model against this identical
# ~1,150-model population means no latent column is ever missing a value for
# a cell line another model type has one for.
full_tensor = torch.tensor(dependency_df[gene_order].values, dtype=torch.float32)
full_loader = DataLoader(TensorDataset(full_tensor), batch_size=32, shuffle=False)
full_metadata = dependency_df[["ModelID"]].reset_index(drop=True)


# In[7]:


# Discover every saved model file directly rather than guessing a filename,
# and validate each one against the actual naming template
# ({model}_latent_dims_{N}_trial_{T}_init_{I}_seed_{S}.joblib) before parsing
# it, so a stray file that doesn't match it is skipped rather than crashing
# the whole run on a parts-index error. A file that *does* match the
# template but fails to load or score is a different kind of problem - that
# fails the run below, rather than silently shipping a partial result.
model_filename_pattern = re.compile(
    r"^(?P<model>[a-z]+)_latent_dims_(?P<dim>\d+)_trial_(?P<trial>[^_]+)_init_(?P<init>\d+)_seed_(?P<seed>-?\d+)$"
)

all_model_files = sorted(model_save_dir.glob("*.joblib"))
print(f"Found {len(all_model_files)} saved models")

latent_frames = []
skipped_models = []
unrecognized_files = []
for model_file in all_model_files:
    match = model_filename_pattern.match(model_file.stem)
    if not match:
        unrecognized_files.append(model_file.name)
        continue
    model_name, dim, init = match["model"], int(match["dim"]), int(match["init"])

    try:
        model = joblib.load(model_file)
        latent_df = extract_latent_dims(model_name, model, dependency_df, full_loader, full_metadata)
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
if unrecognized_files:
    print(f"{len(unrecognized_files)} files didn't match the expected naming template and were skipped: {unrecognized_files[:10]}")
if skipped_models:
    raise RuntimeError(
        f"{len(skipped_models)} models matching the expected naming template failed to load or score "
        f"(first few: {skipped_models[:5]}) - fix the cause rather than exporting a partial ensemble."
    )

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
print(f"Latent dimensions with any missing value among these {len(common_model_ids)} cell lines: "
      f"{latent_common.isna().any().sum()}")


# In[9]:


def correlate_against_drugs(latent_df, drug_df, shuffle_flag):
    """
    Pearson correlation (and its p-value) between every latent-dim column and
    every drug column, for the ModelIDs both have a value for.

    Missingness is handled before calling pairwise_compare, not by it:
    pairwise_compare treats any row with a missing feature value as entirely
    uncorrelatable (NaN for every comparison involving that row), rather than
    dropping missing values pairwise per comparison the way a naive per-pair
    loop would. PRISM's missingness is concentrated on the drug side (every
    model type now scores the identical population, so latent_df itself
    should have none - checked explicitly below rather than assumed), and
    drugs that share an identical set of non-missing cell lines can be
    batched into one call together - only ~232 distinct patterns exist
    across PRISM's ~4,686 drugs, so this is ~232 calls rather than 4,686,
    each handling every latent dimension against every drug in its batch in
    one vectorized step.
    """
    patterns = drug_df.notna().apply(lambda col: frozenset(col[col].index), axis=0)
    groups = patterns.groupby(patterns).groups

    results = []
    for pattern, drugs in groups.items():
        valid_ids = [x for x in latent_df.index if x in pattern]
        if len(valid_ids) < 3:
            continue
        sub_latent_full = latent_df.loc[valid_ids]
        sub_latent = sub_latent_full.dropna(axis=1)
        n_dropped = sub_latent_full.shape[1] - sub_latent.shape[1]
        if n_dropped:
            raise ValueError(
                f"{n_dropped} latent columns have a missing value for this drug group's "
                f"{len(valid_ids)} cell lines - every model type is expected to score the same "
                "population, so this means that assumption broke, not that it's safe to drop silently."
            )
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
