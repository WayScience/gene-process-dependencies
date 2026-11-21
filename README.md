# Gene Process Dependencies

We define a **gene process dependency** as a group of genes in the same biological process, pathway, or protein complex that a cancer cell depends on together for survival.

![Overview of the BioBombe framework: six machine learning algorithms applied across many latent dimensionalities to DepMap gene dependency data, evaluated for reconstruction, gene set coverage, and stability, then interpreted via GSEA against Reactome pathways and CORUM complexes, and correlated with PRISM drug screen data to nominate drug candidates](figure1_overview.jpg)

> Curd, J.B., Balagopal, N.K., Green, A.L. Way, G.P. "Characterizing the landscape of gene process dependencies in cancer." *BMC Genomics* (2026). [https://doi.org/10.1186/s12864-026-13457-5](https://doi.org/10.1186/s12864-026-13457-5).

The figure above shows how this project finds them: many dimensionality-reduction models compress DepMap's CRISPR gene-dependency data into a shared "latent" representation, and each resulting dimension gets matched back to a real Reactome pathway or CORUM protein complex (panel C), and separately correlated with drug sensitivity data to nominate drug candidates.

## Goal

Precision oncology tries to match a drug to a tumor's genetics, but today that benefits fewer than 10% of cancer patients.
A big reason is that it searches for single-gene drug targets, and only about 15% of human proteins are even considered druggable.
The Cancer Dependency Map (DepMap) expanded that search by mapping which single genes cancer cells depend on for survival, across hundreds of cell lines and dozens of cancer types.
But a gene rarely acts alone: cancer is driven by coordinated, multi-gene *processes* (signaling pathways, protein complexes, metabolic networks), not single genes in isolation.
PARP inhibitors (synthetic lethality in DNA damage repair) and immune checkpoint blockade are proof that targeting a process, not a single gene, can work as a therapeutic strategy.

**Our goal is to characterize the landscape of gene process dependencies directly, and use them to nominate new cancer drug targets.**
We apply BioBombe, an ensemble of hundreds of dimensionality reduction models (PCA, ICA, NMF, and three VAE variants, each trained at many latent dimensionalities from k=2 to k=200), to DepMap's CRISPR knockout dependency data for 958 cancer cell lines spanning both pediatric and adult tumors.
No single model, or single latent dimensionality, recovers every gene process dependency on its own; together, the ensemble approaches a far more complete picture than any individual model could.
We interpret every latent dimension biologically via gene set enrichment against Reactome pathways and CORUM protein complexes, and correlate it with PRISM drug screen viability data, surfacing both expected biology and cancer type-specific vulnerabilities (TP53-related and mitochondrial dependencies in glioma, for instance) alongside previously unrecognized drug candidates.

Because most tumors are never profiled with a CRISPR screen, we also built a bridge: models that predict each gene process dependency directly from RNA-seq data.
Applied to an external pediatric high-grade glioma (pHGG) RNA-seq dataset, this approach predicted, and we then experimentally validated in the lab, that **cladribine** is a candidate therapeutic for killing these cells: a complete path from an unscreened tumor's RNA-seq profile, to a predicted vulnerability, to a validated drug.

## A visual tour

**A large, diverse starting point.**
958 DepMap cancer cell lines, spanning both pediatric and adult tumors and dozens of cancer types.
Every gene process dependency in this project is discovered from this cohort, so which cancer types and ages it covers shapes what BioBombe can and can't find.

![Cell line cohort by cancer type, age, and pediatric status](1.data-exploration/figures/cancer_type_age_and_ped_model_distributions.png)

**Why one model is not enough.**
Six different algorithms (PCA, ICA, NMF, and three VAE variants), each run at many latent dimensionalities, each recover different real biology.
Every point on this plot is one model's single best pathway match at one setting, and no single model or setting wins consistently.
Relying on just one, the typical approach, would mean missing most of the real gene process dependencies hiding in this data.

![Highest GSEA enrichment score across all pathways, by latent dimension, for every algorithm in the ensemble](3.run-biobombe/visualize/latent_plot.png)

**Turning a number into biology.**
A latent dimension learned by one of these models is just a column of numbers until it is matched back to a real pathway, protein complex, or drug.
This plot shows that match for three cancer samples at once: which Reactome pathways, CORUM complexes, and drugs line up with what each sample's model actually learned, making clear how one sample's biology differs from another's.

![Radar plots of per-sample Reactome pathway, CORUM complex, and drug response associations](4.drug-dependency/visualize/combined_radar_plots.png)

**Why this matters: a prediction that held up in the lab.**
Starting only from a tumor's RNA-seq profile, with no CRISPR screen at all, this approach predicted that pediatric high-grade glioma cells would be vulnerable to the drug cladribine.
We then tested that prediction directly on real glioma cells, and it worked reasonably well.
This is the project's proof of concept, a computational prediction, made from cheap and widely available data, that pointed to a real, working cancer drug candidate.

![Predicted drug latent score vs. measured cell-killing activity for three candidate drugs on pediatric high-grade glioma lines](6.collab-data/visualize/drug_latent_vs_killing.png)

## Data

All DepMap and PRISM data are publicly available.
- Cancer dependency data: [Cancer Dependency Map resource](https://depmap.org/portal/download/) (pinned to the 24Q2 release; see [0.data-download](0.data-download/)).
- Drug response data: the [PRISM Repurposing screen](https://depmap.org/repurposing/), also hosted on DepMap.
- The pretrained, 486-model BioBombe ensemble is hosted on [Figshare](https://figshare.com/articles/dataset/Gene_Process_Dependencies_-_BioBombe_Models/33437659) rather than checked into version control (~1.2 GB); see [0.data-download/3.download-saved-biobombe-models-from-figshare.ipynb](0.data-download/3.download-saved-biobombe-models-from-figshare.ipynb).
- RNA-seq data: DepMap's own RNA-seq profiles, matched to the same CRISPR-screened cell lines used everywhere else in this project, train the RNA-seq-to-latent-dimension bridge ([5.RNAseq](5.RNAseq/)); bulk RNA-seq from pediatric tumors shared by a project collaborator, Dr. Adam Green, via [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE231858), has no matched CRISPR screen at all, which is what that bridge is actually applied to ([6.collab-data](6.collab-data/)).

## Repository structure

This repository is a numbered pipeline, meant to be run in order.

| Order | Module | Description |
| :---- | :----- | :---------- |
| [0.data-download](0.data-download/) | Download data | Download CRISPR gene effect data, cell line metadata, a QC'd gene dictionary, and the pretrained BioBombe ensemble |
| [1.data-exploration](1.data-exploration/) | Explore and split data | Visualize cell line demographics and split gene effect data into train, test, and validation sets |
| [2.prototype-VAE-models](2.prototype-VAE-models/) | Train and validate a single Beta-VAE / Beta-TC-VAE | Optimize hyperparameters, train one representative Beta-VAE and Beta-TC-VAE alongside PCA/ICA/NMF baselines, then check every model with heatmaps, GSEA, and t-tests |
| [3.run-biobombe](3.run-biobombe/) | Train and validate the full BioBombe ensemble | Sweep PCA, ICA, NMF, VanillaVAE, BetaVAE, and BetaTCVAE across many latent dimensions, run GSEA against CORUM and Reactome on each, then check ensemble consistency with CKA and reconstruction quality |
| [4.drug-dependency](4.drug-dependency/) | Correlate with drug response | Correlate latent dimensions with PRISM drug screen viability data, then visualize the strongest per-sample hits with spider and pinwheel plots |
| [5.RNAseq](5.RNAseq/) | Bridge to RNA-seq | Train models that predict each latent dimension from RNA-seq expression, using DepMap cell lines that have both |
| [6.collab-data](6.collab-data/) | Apply to external data | Apply the RNA-seq bridge to real collaborator samples with no CRISPR screen, and compare predicted vulnerabilities to observed cell killing |
| [7.shiny-app](7.shiny-app/) | Prepare visualization data | Build the PCA projections used by the project's Shiny app |
| [8.apply-biobombe-new-data](8.apply-biobombe-new-data/) | Apply BioBombe to new data | Align a new gene-dependency dataset to the trained ensemble's genes, sanity-check its distribution against DepMap, apply every saved model, annotate the resulting latent scores with CORUM/Reactome pathways and drug correlations, and test for differences by sample group |

## Environment setup

This project uses two separate conda environments: `gene_dependency_representations`, for the Python/PyTorch analysis pipeline (every numbered module), and `gene_dependency_figures`, for the R-based figure scripts under a few modules' `nbconverted/` directories.
Most contributors only need the first.

### Step 1: Install mamba

These environments install PyTorch and JupyterLab, so they're large enough that conda's solver is slow.
[mamba](https://mamba.readthedocs.io/) is a drop-in, much faster replacement.

```sh
conda install -n base -c conda-forge mamba -y
```

### Step 2: Create the environment(s)

```sh
# conda version 24.5.0
mamba env create --yes --file environment.yml           # main pipeline (required)
mamba env create --yes --file figure_environment.yml    # R figure scripts only (optional)
```

Plain `conda env create` also works for either, if you'd rather skip mamba, just slower to resolve.

### Step 3: Activate an environment

```sh
conda activate gene_dependency_representations   # for everything except the R figure scripts
conda activate gene_dependency_figures            # for the .r scripts specifically
```
