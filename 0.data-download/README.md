# Description of the Data

## DepMap Public 24Q2

This DepMap release contains data from in vitro studies of genetic dependencies in cancer cell lines using CRISPR/Cas9 loss-of-function screens from project Achilles, project Sanger, and several other datasets (e.g. CCLE)

Data resource:
[DepMap 24Q2 Public](https://plus.figshare.com/articles/dataset/DepMap_24Q2_Public/25880521), DOI 10.25452/figshare.plus.25880521.v1.

The 24Q2 release notes are described here: https://forum.depmap.org/t/announcing-the-24q2-release/3312.

## Files used in our multivariate gene dependency project

See the following resource for more information: https://forum.depmap.org/t/depmap-genetic-dependencies-faq/131.

### CRISPRGeneEffect.csv

Integrated dataset processed by a joint Chronos run for the Achilles dataset (Avana library) combined with the Sanger dataset (KY library).
The two datasets were adjusted for by Chronos 2.0 and integrated using a ComBat-based algorithm called Harmonia.

Chronos adjusts for sgRNA efficacy, screen quality, differential cell growth rates, and copy number differences across cell models.

> Dempster, J.M., Boyle, I., Vazquez, F. et al. Chronos: a cell population dynamics model of CRISPR experiments that improves inference of gene fitness effects. Genome Biol 22, 343 (2021). https://doi.org/10.1186/s13059-021-02540-7

The dataset is also adjusted for gene knockout effects that occur on the same chromosome arm.

### CRISPRGeneDependency.csv

This data comprises scores from individual CRISPR gene knockout screens in cancer cell lines for every gene across many different ages and cancer types.
The values represent gene dependency probability estimates for cell survival and growth for all models in the integrated gene effect.
The higher the score, the greater the gene dependency. 

The probability estimate is derived from the CRISPRGeneEffect estimates.

Columns (18,444): gene
Rows (1,150): the ModelID

### Model.csv

This file gives details on the cell lines, type of cancer, sex, age, and unique IDs of the patient.

Columns (43): description of cancer, and patient's biological and ID information
Rows (1,960): the ModelID

## Constructing the gene filtering dictionary

Gene dependency file columns are formatted as `"gene symbol (entrez id)"`.
`2.construct_gene_dictionary.ipynb` splits `CRISPRGeneEffect.csv` column names into a six-column dictionary (`entrez_id`, `symbol_id`, the original `dependency_column`, two per-source QC pass/fail flags, and a QC summary column), using QC calls from `depmap_gene_meta.tsv` (see Pan et al. 2022). 
The result is written to `CRISPR_gene_dictionary.parquet`.

## Downloading the pretrained BioBombe ensemble

`3.download-saved-biobombe-models-from-figshare.ipynb` downloads the pretrained BioBombe ensemble and places it at `../3.run-biobombe/saved_models/`, the location every downstream analysis script expects fitted models to live.
The archive contains 486 `.joblib` models — PCA, ICA, NMF, VanillaVAE, BetaVAE, and BetaTCVAE, each swept across 27 latent dimensions.
These were trained previously and archived on Figshare rather than checked into version control, since the combined archive is roughly 1.2 GB.
The notebook verifies the download against Figshare's reported md5 checksum.
It also confirms every extracted filename matches the naming convention downstream scripts parse.

Source: [Gene Process Dependencies - BioBombe Models](https://figshare.com/articles/dataset/Gene_Process_Dependencies_-_BioBombe_Models/33437659), Figshare, CC BY 4.0.

> Curd, Julia; Way, Gregory (2026). Gene Process Dependencies - BioBombe Models. figshare. Dataset. https://doi.org/10.6084/m9.figshare.33437659.v1
