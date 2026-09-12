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

### `0.preprocess_raw_data`

Dataset-specific: converts whatever a new dataset actually arrives as into the shape every other script in this module expects (one row per sample, one column per gene, a `ModelID` column, gene columns named to match DepMap's current symbols). For the Largaespada lab MPNST data currently in `data/largaespada/raw_data/`, that means transposing the raw gene-effect matrix and harmonizing gene symbols HGNC has renamed since the data was generated (e.g. old aminoacyl-tRNA synthetase names, the 2019 ATP synthase subunit renames, the 2020-21 histone nomenclature overhaul) to whatever symbol BioBombe was actually trained on, using mygene.info's alias lookup, entrez id as the stable bridge across renames. Resolution works backward from the ~163 trained genes missing under their current symbol (asking each one's own former names) rather than forward from ~1,700 raw columns of uncertain identity, since the forward direction hits genuinely ambiguous aliases shared by unrelated genes (e.g. "STRA13" names both CENPX and the unrelated BHLHE40) with no principled way to pick.
Of BioBombe's 2,718 trained genes, 151 of the 163 initially missing were recovered this way; the remaining 12 (`CALM2`, `CARS2`, `CDK8`, `COQ4`, `KAT6A`, `MED20`, `NEPRO`, `PPP2R3C`, `SGO1`, `SPRTN`, `TAB2`, `TXNL4A`) have no match anywhere in the raw data under any name mygene.info knows — most likely because this MPNST-specific screen used a different guide library than DepMap's genome-wide Avana library, not a resolution failure. That list is written to `data/largaespada/results/genes_unresolved_after_alias_lookup.parquet` each time this runs, so it stays current if the raw data or BioBombe's trained gene set ever changes. `2.format_and_scale_data` imputes these with the DepMap training mean, same as any other gene missing from a new dataset.
Writes `NF1_data.parquet` and `NF1_metadata.parquet` to `data/largaespada/`, the exact filenames `1.load_and_assess_gene_overlap` and `2.format_and_scale_data` already hardcode.

### `1.load_and_assess_gene_overlap`

Loads the new dataset and compares its gene membership and column order against `CRISPRGeneEffect.csv`.
DepMap gene identifiers and any new data's gene identifiers aren't guaranteed to match by construction, so this step is not a formality: it reports how many genes overlap, how many are missing on either side, and whether shared genes are already aligned in the same order.
It writes a gene-overlap report and a missing-gene list to `data/results/`, which later steps in this module can read before trusting any join.

### `2.format_and_scale_data`

Formats the new data two ways, via `data_prep.prepare_new_data_for_biobombe()`: once aligned to the full DepMap gene set and column order, and once subset and reordered to the QC-passed genes in the exact order the saved models expect.
Computes a `MinMaxScaler` from `1.data-exploration/data/VAE_train_df.parquet`, the exact 670-model train split the ensemble was actually trained on (not DepMap's full ~1150-model `CRISPRGeneEffect.csv` — fitting on the broader cohort gives measurably different min/max values for roughly 40% of genes, which showed up as new data scaling outside [0, 1] and breaking NMF, which requires non-negative input), and persists it to `results/depmap_gene_scaler.joblib`.
Every downstream script in this module (and any future one) reuses that persisted scaler rather than refitting on new data, since refitting would rescale relative to the new cohort instead of the training distribution the models actually learned.
Genes the trained ensemble expects but the new data doesn't have are imputed with the training-split mean for that gene, not dropped, so every sample still has a value for every trained gene going into the ensemble.

### `3.sanity_check_distribution`

Compares the newly formatted data against BioBombe's actual 670-model training split (`1.data-exploration/data/VAE_train_df.parquet`) before any model ever sees it — not DepMap's broader ~1150-model cohort, since the training split is the exact population the persisted scaler was fit on and what "in-distribution" means for these specific models.
Projects both into a shared UMAP space and displays the new samples against the training distribution inline; the coordinates are saved to `data/largaespada/results/NF1_data_umap_coordinates.parquet` for a later, publication-ready R figure rather than a PNG from this script.
Also checks simpler per-gene distributional sanity, restricted to the trained genes specifically: value range, missingness, and overall scale, flagging genes or samples that look implausible for a real CRISPR screen rather than letting a formatting bug silently propagate into latent scores.

### `4.apply_biobombe_ensemble`

Loads each saved model from `3.run-biobombe/saved_models/` and applies it to the new data one at a time — `.transform()` for PCA/ICA/NMF, `.encode()` for the VAE variants — the same per-model-type dispatch `utils/model_utils.py` already uses elsewhere in this project.
Writes one combined latent-scores table to `data/results/new_data_latent_scores.parquet`, covering every model, latent dimension, and initialization in the ensemble.
Building this surfaced a real bug in `vvae_extract_latent_dimensions` (`2.prototype-VAE-models/utils/vanillavae.py`) that would have crashed on every VanillaVAE model in the ensemble — a signature mismatch with its caller, plus a numpy/tensor type mismatch inside the function — both fixed as part of this module, since they block inference for an already-trained model, not just training.

### `5.annotate_latent_scores`

Links each new sample's latent scores to the GSEA pathway results (`3.run-biobombe/gsea_results/`) for the corresponding model and latent dimension, the same pattern `6.collab-data/scripts/3.pinwheel_plots.py` uses for collaborator data, via `utils/pinwheels.py`.
Drug annotation follows the same pattern against `4.drug-dependency/results/combined_latent_drug_correlations.parquet` when that file exists; it's produced by `4.drug-dependency/scripts/5.drug_latent_dim_correlation.py`, which only became runnable once the BioBombe ensemble was downloaded, so this step skips drug annotation with a clear message rather than failing if it hasn't been generated yet.
Produces a per-sample ranked table of top associated pathways (and drugs, when available) in `data/results/`, plus pinwheel plots per sample in `visualize/pinwheels/`.

## Data

`data/` holds the specific new dataset this module is currently applied to (its untouched delivery under `data/<dataset>/raw_data/`, plus `0.preprocess_raw_data`'s output alongside it), its intermediate and final results (`data/results/`), and figures (`data/visualize/`).
It's gitignored for now, the same way `0.data-download/data/` is: the dataset isn't ready to share yet, so it stays local until that changes.
`visualize/pinwheels/` is gitignored for the same reason — one plot per new sample.
The persisted scaler in `results/` is the exception — it's derived purely from DepMap's already-public training data, not from the new dataset, so it's safe to commit and reuse across whatever new data comes next.
