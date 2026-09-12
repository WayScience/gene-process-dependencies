#!/usr/bin/env python
# coding: utf-8

# ## Apply the trained BioBombe ensemble to the new data
# 
# Loads every saved model in `3.run-biobombe/saved_models/` and runs the new,
# formatted-and-scaled data through it one at a time: `.transform()` for
# PCA/ICA/NMF, `.encode()` for the VAE variants, via the same per-model-type
# dispatch (`utils/model_utils.py`) `3.run-biobombe`'s own CKA and
# reconstruction-quality scripts already use.
# 
# Run this script with `8.apply-biobombe-new-data/` as the working directory.

# In[ ]:


import pathlib
import sys

import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "../utils/")
from model_utils import extract_latent_dims

sys.path.insert(0, "utils")
import data_prep as dp


# In[ ]:


model_save_dir = pathlib.Path("../3.run-biobombe/saved_models")
data_results_dir = pathlib.Path("data/largaespada/results")


# In[ ]:


new_scaled_df = pd.read_parquet(data_results_dir / "NF1_data_model_ready_scaled.parquet")
gene_columns = [c for c in new_scaled_df.columns if c != "ModelID"]

# metadata and the tensor loader must stay in the same row order as new_scaled_df,
# the same contract extract_latent_dimensions/tc_extract_latent_dimensions/
# vvae_extract_latent_dimensions rely on for the VAE variants.
metadata = new_scaled_df[["ModelID"]].reset_index(drop=True)
new_tensor = torch.tensor(new_scaled_df[gene_columns].values, dtype=torch.float32)
new_loader = DataLoader(TensorDataset(new_tensor), batch_size=32, shuffle=False)

print(f"New data ready for the ensemble: {new_scaled_df.shape[0]} samples, {len(gene_columns)} genes")


# In[ ]:


def parse_model_filename(model_file):
    """Matches the {model}_latent_dims_{N}_trial_{T}_init_{I}_seed_{S}.joblib
    convention used throughout 3.run-biobombe and 4.drug-dependency."""
    parts = model_file.stem.split("_")
    return {
        "model": parts[0],
        "latent_dim_total": int(parts[3]),
        "init": int(parts[7]),
        "seed": int(parts[9]),
    }


# In[ ]:


all_latent_scores = []
skipped = []

for model_file in sorted(model_save_dir.glob("*.joblib")):
    try:
        model_info = parse_model_filename(model_file)
    except (IndexError, ValueError):
        skipped.append((model_file.name, "unexpected filename format"))
        continue

    try:
        model = joblib.load(model_file)
        latent_df = extract_latent_dims(
            model_info["model"], model, new_scaled_df, new_loader, metadata
        )
    except Exception as e:
        skipped.append((model_file.name, str(e)))
        continue

    latent_df["model"] = model_info["model"]
    latent_df["latent_dim_total"] = model_info["latent_dim_total"]
    latent_df["init"] = model_info["init"]
    latent_df["seed"] = model_info["seed"]
    all_latent_scores.append(latent_df)

print(f"Applied {len(all_latent_scores)} / {len(all_latent_scores) + len(skipped)} saved models")
if skipped:
    print(f"Skipped {len(skipped)} models, first 10:")
    for name, reason in skipped[:10]:
        print(f"  {name}: {reason}")


# In[ ]:


combined_latent_df = pd.concat(all_latent_scores, ignore_index=True)
print(combined_latent_df.shape)
combined_latent_df.head()


# ### Attach sample metadata
#
# Whatever per-sample metadata comes with the new dataset (clinical info,
# tumor type, etc.) — anything keyed by ModelID — gets carried through here
# rather than dropped. Point sample_metadata_path at that file once it exists;
# every column in it (other than ModelID) rides along on every row of the
# combined latent scores.


# ### Attach sample metadata
# 
# Whatever per-sample metadata comes with the new dataset (clinical info,
# tumor type, etc.) — anything keyed by ModelID — gets carried through here
# rather than dropped. Point sample_metadata_path at that file once it exists;
# every column in it (other than ModelID) rides along on every row of the
# combined latent scores.

# In[ ]:


sample_metadata_path = pathlib.Path("data/largaespada/NF1_metadata.parquet")

if sample_metadata_path.exists():
    sample_metadata_df = pd.read_parquet(sample_metadata_path)
    before_cols = set(combined_latent_df.columns)
    combined_latent_df = combined_latent_df.merge(sample_metadata_df, on="ModelID", how="left")
    added_cols = [c for c in combined_latent_df.columns if c not in before_cols]
    print(f"Attached sample metadata from {sample_metadata_path}: {added_cols}")
else:
    print(f"{sample_metadata_path} not found — proceeding without sample metadata.")


# In[ ]:


output_file = data_results_dir / "NF1_data_latent_scores.parquet"
combined_latent_df.to_parquet(output_file, index=False)
print(f"Saved combined latent scores to {output_file}")

