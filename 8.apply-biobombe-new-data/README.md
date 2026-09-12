# Apply BioBombe to new data

## Purpose

This module applies the trained BioBombe ensemble to new, external gene dependency data.
The new data are DepMap-like: CRISPR gene knockout dependency scores, one row per sample and one column per gene, the same modality the ensemble was trained on.
Unlike `5.RNAseq` and `6.collab-data`, this module doesn't bridge across modalities — it runs new samples through the saved models directly.
The output is a set of latent scores per sample per model, annotated with the same GSEA pathway and drug correlation context used everywhere else in this project, so a new sample's vulnerabilities can be interpreted the same way an existing DepMap cell line's can.

## Contents

### `utils/data_prep.py`

The shared functions every numbered script below builds on: loading new data, resolving its gene identifiers against DepMap's, formatting it both ways, and fitting, persisting, and applying the training scaler.
`prepare_new_data_for_biobombe()` chains all of that into one call — load, assess overlap, format both ways, scale — for anyone who wants the finished, model-ready data without stepping through it, which is what `2.format_and_scale_data` itself calls; `1.load_and_assess_gene_overlap` calls the loading and overlap-assessment pieces directly, since its job is to surface that report on its own before scaling ever happens.

### `1.load_and_assess_gene_overlap`

Loads the new dataset and compares its gene membership and column order against `CRISPRGeneEffect.csv`.
DepMap gene identifiers and any new data's gene identifiers aren't guaranteed to match by construction, so this step is not a formality: it reports how many genes overlap, how many are missing on either side, and whether shared genes are already aligned in the same order.
It writes a gene-overlap report and a missing-gene list to `data/results/`, which later steps in this module can read before trusting any join.

### `2.format_and_scale_data`

Formats the new data two ways, via `data_prep.prepare_new_data_for_biobombe()`: once aligned to the full DepMap gene set and column order, and once subset and reordered to the QC-passed genes in the exact order the saved models expect.
Computes a `MinMaxScaler` from `0.data-download/data/CRISPRGeneEffect.csv`, the original data the ensemble was trained on, and persists it to `results/depmap_gene_scaler.joblib`.
Every downstream script in this module (and any future one) reuses that persisted scaler rather than refitting on new data, since refitting would rescale relative to the new cohort instead of the training distribution the models actually learned.
Genes the trained ensemble expects but the new data doesn't have are imputed with the DepMap training mean for that gene, not dropped, so every sample still has a value for every trained gene going into the ensemble.

### `3.sanity_check_distribution`

Compares the newly formatted data against the DepMap cohort before any model ever sees it.
Projects both into a shared UMAP space and plots the new samples against the DepMap distribution, saved to `data/visualize/umap_new_data_vs_depmap.png`.
Also checks simpler per-gene distributional sanity: value range, missingness, and overall scale, flagging genes or samples that look implausible for a real CRISPR screen rather than letting a formatting bug silently propagate into latent scores.

### `4.apply_biobombe_ensemble`

Loads each saved model from `3.run-biobombe/saved_models/` and applies it to the new data one at a time — `.transform()` for PCA/ICA/NMF, `.encode()` for the VAE variants — the same per-model-type dispatch `utils/model_utils.py` already uses elsewhere in this project.
Writes one combined latent-scores table to `data/results/new_data_latent_scores.parquet`, covering every model, latent dimension, and initialization in the ensemble.
Building this surfaced a real bug in `vvae_extract_latent_dimensions` (`2.prototype-VAE-models/utils/vanillavae.py`) that would have crashed on every VanillaVAE model in the ensemble — a signature mismatch with its caller, plus a numpy/tensor type mismatch inside the function — both fixed as part of this module, since they block inference for an already-trained model, not just training.

### `5.annotate_latent_scores`

Links each new sample's latent scores to the GSEA pathway results (`3.run-biobombe/gsea_results/`) for the corresponding model and latent dimension, the same pattern `6.collab-data/scripts/3.pinwheel_plots.py` uses for collaborator data, via `utils/pinwheels.py`.
Drug annotation follows the same pattern against `4.drug-dependency/results/combined_latent_drug_correlations.parquet` when that file exists; it's produced by `4.drug-dependency/scripts/5.drug_latent_dim_correlation.py`, which only became runnable once the BioBombe ensemble was downloaded, so this step skips drug annotation with a clear message rather than failing if it hasn't been generated yet.
Produces a per-sample ranked table of top associated pathways (and drugs, when available) in `data/results/`, plus pinwheel plots per sample in `visualize/pinwheels/`.

## Data

`data/` holds the specific new dataset this module is currently applied to, along with its intermediate and final results (`data/results/`) and figures (`data/visualize/`).
It's gitignored for now, the same way `0.data-download/data/` is: the dataset isn't ready to share yet, so it stays local until that changes.
`visualize/pinwheels/` is gitignored for the same reason — one plot per new sample.
The persisted scaler in `results/` is the exception — it's derived purely from DepMap's already-public training data, not from the new dataset, so it's safe to commit and reuse across whatever new data comes next.
