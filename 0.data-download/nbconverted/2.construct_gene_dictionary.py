#!/usr/bin/env python
# coding: utf-8

# ## Construct gene dictionary
# 
# The column names of the gene dependency files are of the format "gene symbol (entrez id)".
# 
# Additionally, the `depmap_gene_meta.tsv` contains genes that passed an initial QC (see Pan et al. 2022).
# 
# This notebook will create a six column matrix that separates symbol from entrez id, retains the original column name, two columns of if the gene passed two different QC, and a QC summary column.
# 
# Example:
# 
# | entrez_id | symbol_id | dependency_column | qc_pass_pan | qc_pass_other | qc_pass |
# | :-------: | :-------: | :---------------: | :---------: | :-----------: | :-----: |
# | 1 | A1BG |A1BG (1)| True | True | True |
# | 29974 | A1CF | A1CF (29974) | True | False | False |
# |	2 	| A2M | A2M (2) | False | True | False |
# 
# *Note, the example qc_pass column above is an example and may not reflect truth.*
# 
# ### Quality control columns
# 
# - `qc_pass_pan` refers to the genes QC'd by Pan et al. 2022
# - `qc_pass_other` refers to gene families filtered by saturated signals
#     - RPL - Ribosomal proteins (including mitochondrial)
#     - RPS - S Ribosomal proteins
# - `qc_pass` refers to genes that pass all qc metrics

# In[1]:


import pathlib
import pandas as pd


# In[2]:


base_dir = "data/"

dependency_file = pathlib.Path(f"{base_dir}/CRISPRGeneEffect.csv")
qc_gene_file = pathlib.Path(f"{base_dir}/depmap_gene_meta.tsv")

output_gene_dict_file = pathlib.Path(f"{base_dir}/CRISPR_gene_dictionary.parquet")


# In[3]:


# Load gene dependency data. DepMap's own CSV leaves the first column
# (ModelID) unlabeled, so read it in by position rather than by name.
dependency_df = pd.read_csv(dependency_file, index_col=0)

print(dependency_df.shape)
dependency_df.head()


# In[4]:


# Load depmap metadata
gene_meta_df = pd.read_csv(qc_gene_file, sep="\t")
gene_meta_df.entrezgene = gene_meta_df.entrezgene.astype(str)

print(gene_meta_df.shape)
gene_meta_df.head(3)


# ## Obtain the intersection of the genes
# 
# Comparing the current DepMap release and the previous gene set qc (19Q2 depmap release)

# In[5]:


# Recode column names to entrez ids from dependency file
# (ModelID is the index now, not a column, so every column here is a gene)
entrez_genes = [x[1].strip(")").strip() for x in dependency_df.columns.str.split("(")]

# Obtain intersection of entrez gene ids
entrez_intersection = list(
    set(gene_meta_df.entrezgene).intersection(set(entrez_genes))
)

print(f"The number of overlapping entrez gene ids: {len(entrez_intersection)}")

# Subset the gene metadata file to only those in common, which are ones that passed qc
gene_passed_qc_df = (
    gene_meta_df
    .query("entrezgene in @entrez_intersection")
    .set_index("entrezgene")
    .reindex(entrez_intersection)
    .reset_index()
    .loc[:, ["entrezgene", "Name", "symbol"]]
)

gene_passed_qc_df.head()


# ## Convert the initial dependency map input file to three parts
# 
# 1. Entrez ID
# 2. Symbol
# 3. The full column name

# In[6]:


entrez_genes = [x[1].strip(")").strip() for x in dependency_df.columns.str.split("(")]
symbol_genes = [x[0].strip() for x in dependency_df.columns.str.split("(")]

gene_dictionary_df = pd.DataFrame(
    [
        entrez_genes,
        symbol_genes,
        dependency_df.columns.tolist()
    ]
).transpose()

gene_dictionary_df.columns = ["entrez_id", "symbol_id", "dependency_column"]

print(gene_dictionary_df.shape)
gene_dictionary_df.head()


# ## Create the QC columns

# In[7]:


# These gene families consistently oversaturate signals in latent representations
qc_fail_other_genes = "RPL|RPS"


# In[8]:


gene_dictionary_qc_df = (
    # Merge gene dictionary with qc dataframe
    gene_dictionary_df.merge(
        gene_passed_qc_df,
        left_on="entrez_id",
        right_on="entrezgene",
        how="left"  # Note the left merge, to retain all genes from gene_dictionary_df
    )
    # Select only certain columns
    .loc[:, ["entrez_id", "symbol_id", "dependency_column", "entrezgene"]]
    # Values that are missing indicate genes that did not pass QC
    .fillna(value={"entrezgene": False})
    # Rename the column to be clearly defined
    .rename(columns={"entrezgene": "qc_pass_pan"})
)

# Convert genes with entrez entries to those that indicate QC pass
gene_dictionary_qc_df.loc[gene_dictionary_qc_df.qc_pass_pan != False, "qc_pass_pan"] = True

# Create the qc_pass_other column
gene_dictionary_qc_df = (
    gene_dictionary_qc_df.assign(
        qc_pass_other=~gene_dictionary_qc_df.symbol_id.str.contains(qc_fail_other_genes)
    )
)

# Create qc_pass summary column
gene_dictionary_qc_df = (
    gene_dictionary_qc_df.assign(
        qc_pass=(gene_dictionary_qc_df.qc_pass_pan & gene_dictionary_qc_df.qc_pass_other)
    )
)

# Output file
gene_dictionary_qc_df.to_parquet(output_gene_dict_file, index=False)

print(gene_dictionary_qc_df.qc_pass.value_counts())
print(gene_dictionary_qc_df.shape)

gene_dictionary_qc_df.head(3)

