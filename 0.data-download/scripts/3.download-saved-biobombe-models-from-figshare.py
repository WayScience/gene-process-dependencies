#!/usr/bin/env python
# coding: utf-8

# ## Download Saved BioBombe Models
#
# This notebook downloads the pretrained BioBombe ensemble and places it at
# [`3.run-biobombe/saved_models/`](../3.run-biobombe/saved_models/),
# the location every downstream analysis script (`2.prototype-VAE-models`, `3.run-biobombe/2.interpret_biobombe_with_gsea.py`,
# `4.drug-dependency`, `5.RNAseq`) expects fitted models to live.
#
# The archive contains 486 `.joblib` models: PCA, ICA, NMF, VanillaVAE, BetaVAE, and BetaTCVAE,
# each swept across 27 latent dimensions (the VAE variants trained with 5 random-seed
# reinitializations per dimension; PCA/ICA/NMF are deterministic, so one fit each).
# These models were trained previously and archived on Figshare rather than checked into
# version control, since the combined archive is roughly 1.2 GB.
#
# Source: [Gene Process Dependencies - BioBombe Models](https://figshare.com/articles/dataset/Gene_Process_Dependencies_-_BioBombe_Models/33437659), Figshare, CC BY 4.0.
#
# > Curd, Julia; Way, Gregory (2026). Gene Process Dependencies - BioBombe Models. figshare. Dataset. https://doi.org/10.6084/m9.figshare.33437659.v1
#
# Described further in the companion preprint: [Characterizing the landscape of gene process dependencies in cancer](https://doi.org/10.1101/2025.11.14.688518).

# In[1]:


import hashlib
import pathlib
import re
import shutil
import zipfile

import requests


# ### Step 1: Set download constants

# In[2]:


# Figshare article: https://figshare.com/articles/dataset/Gene_Process_Dependencies_-_BioBombe_Models/33437659
figshare_id = "68212225"
figshare_url = "https://ndownloader.figshare.com/files/"

# Reported by the Figshare API for this file, used to confirm the download wasn't corrupted or truncated
expected_md5 = "b37f2da96699f9f96dd3a1bdbfb372d3"
expected_file_count = 486

# Where the raw zip lands, consistent with how 1.data_downloader.py stages large downloads
staging_dir = pathlib.Path("data")
zip_file = staging_dir / "gene_process_dependency_biobombe_models.zip"

# Canonical location every downstream script expects the fitted models to live
saved_models_dir = pathlib.Path("../3.run-biobombe/saved_models").resolve()


# ### Step 2: Download the archive, verifying its checksum
#
# The archive is streamed to disk in chunks rather than loaded into memory, since it's ~1.2 GB.
# The checksum is computed separately (a streamed read of the file already on disk), so that
# re-running this notebook without re-downloading still re-verifies the file that's present.

# In[3]:


def download_file(figshare_id, figshare_url, output_file, chunk_size=1024 * 1024):
    """
    Stream a figshare file to disk in chunks, to avoid holding the archive in memory.
    """
    download_url = f"{figshare_url}/{figshare_id}"

    with requests.get(download_url, stream=True) as response:
        response.raise_for_status()
        total_bytes = int(response.headers.get("Content-Length", 0))
        downloaded_bytes = 0

        with open(output_file, "wb") as output_file_writer:
            for chunk in response.iter_content(chunk_size=chunk_size):
                output_file_writer.write(chunk)
                downloaded_bytes += len(chunk)

                if total_bytes:
                    percent = downloaded_bytes / total_bytes * 100
                    print(
                        f"\rDownloaded {downloaded_bytes / 1e6:,.0f} / {total_bytes / 1e6:,.0f} MB ({percent:.1f}%)",
                        end="",
                    )
    print()


def compute_md5(file_path, chunk_size=1024 * 1024):
    """
    Compute the md5 checksum of a file on disk, reading it in chunks so large files
    never need to fully fit in memory.
    """
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


# In[4]:


staging_dir.mkdir(exist_ok=True)

if zip_file.exists():
    print(f"{zip_file} already exists, skipping download (delete it to force a re-download)")
else:
    print(f"Downloading {zip_file} (figshare file {figshare_id})...")
    download_file(figshare_id, figshare_url, zip_file)

md5_checksum = compute_md5(zip_file)
assert md5_checksum == expected_md5, (
    f"Checksum mismatch: expected {expected_md5}, got {md5_checksum}. "
    "The download may be corrupted or truncated; delete the file and retry."
)
print(f"Checksum verified: {md5_checksum}")


# ### Step 3: Extract into `3.run-biobombe/saved_models/`
#
# The archive contains the models inside a folder; we extract to a temporary staging directory
# first, then flatten every `.joblib` file directly into `saved_models/`, regardless of how it's
# nested in the archive, since every downstream script globs `saved_models/*.joblib` with no
# subfolders.

# In[5]:


extraction_staging_dir = staging_dir / "biobombe_models_extracted"

if saved_models_dir.exists() and any(saved_models_dir.glob("*.joblib")):
    print(f"{saved_models_dir} already contains .joblib files, skipping extraction")
else:
    print(f"Extracting {zip_file}...")
    with zipfile.ZipFile(zip_file) as zf:
        zf.extractall(extraction_staging_dir)

    saved_models_dir.mkdir(parents=True, exist_ok=True)

    joblib_files = list(extraction_staging_dir.rglob("*.joblib"))
    for joblib_file in joblib_files:
        shutil.move(str(joblib_file), str(saved_models_dir / joblib_file.name))

    # Clean up the now-empty extraction scratch space
    shutil.rmtree(extraction_staging_dir)

    print(f"Moved {len(joblib_files)} model files into {saved_models_dir}")


# ### Step 4: Sanity check the extracted models
#
# Confirm the count matches the archive's description (486), and that every filename matches
# the `{model}_latent_dims_{N}_trial_{T}_init_{I}_seed_{S}.joblib` convention that every
# downstream script (e.g. `3.run-biobombe/4.compare_model_weights_cka.py`, `5.RNAseq/3.models.py`) parses.

# In[6]:


filename_pattern = re.compile(
    r"^(?P<model>[a-z]+)_latent_dims_(?P<dims>\d+)_trial_(?P<trial>[\w-]+)_init_(?P<init>\d+)_seed_(?P<seed>\d+)\.joblib$"
)

model_files = sorted(saved_models_dir.glob("*.joblib"))
assert len(model_files) == expected_file_count, (
    f"Expected {expected_file_count} model files, found {len(model_files)} in {saved_models_dir}"
)

unmatched = [f.name for f in model_files if not filename_pattern.match(f.name)]
assert not unmatched, f"{len(unmatched)} file(s) don't match the expected naming convention: {unmatched[:5]}"

model_counts = {}
for f in model_files:
    model_name = filename_pattern.match(f.name).group("model")
    model_counts[model_name] = model_counts.get(model_name, 0) + 1

print(f"All {len(model_files)} model files verified at {saved_models_dir}")
for model_name, count in sorted(model_counts.items()):
    print(f"  {model_name}: {count}")
