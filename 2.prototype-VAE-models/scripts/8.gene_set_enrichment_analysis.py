#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import matplotlib.backends.backend_pdf

sns.set_theme(color_codes=True)

import blitzgsea as blitz


# Download the gene set library here: https://github.com/MaayanLab/blitzgsea


# In[2]:


random.seed(18)
seed = random.random()
print(random.random())


# In[3]:


# list available gene set libraries in Enrichr
blitz.enrichr.print_libraries()


# In[4]:


# use enrichr submodule to retrieve gene set library
# these libraries are finicky to work with--they usually work the first time but then may stop working. You may need to remove the library from your computer and trying to reimport it to work again.
library = blitz.enrichr.get_library("Reactome_2022")


# In[5]:


# load the weight matrix 
gene_weight_dir = pathlib.Path("results/weight_matrix_gsea.parquet").resolve()
signature = pd.read_parquet(gene_weight_dir)
print(signature.shape)
signature.head()


# In[6]:


# Running GSEA

all_GSEA_results = []
all_signatures = []
results = []

range = signature.shape[1]

for col in signature.iloc[:,1:range-1].columns:
    df = signature.loc[:, [signature.columns[0], col]]
    result = blitz.gsea(df, library, seed=seed)
    results.append(result)
    all_GSEA_results.append(result.assign(z_dim=f"z_{col}"))
    all_signatures.append(df)


# In[7]:


# Copying signature dataframe without gene column
neg_signature = signature.iloc[:, 1:].copy()

# Vertically shuffling the data in each column to create a negative control
for col in neg_signature.columns:
    neg_signature.loc[:, col] = np.random.permutation(neg_signature.loc[:, col].values)

# Adding gene column back to finalize negative control data
genes = signature.iloc[:,:1]
neg_signature.insert(0,'0', genes)

# Running GSEA with negative control data
neg_GSEA_results = []
negative_control = []

range = neg_signature.shape[1]

for col in neg_signature.iloc[:,1:range-1].columns:
    neg_df = neg_signature.loc[:, [neg_signature.columns[0], col]]
    neg_result = blitz.gsea(neg_df, library, seed=seed)
    neg_GSEA_results.append(neg_result.assign(z_dim=f"z_{col}"))
    negative_control.append(neg_df)


# In[8]:


# stack up all of the results to be analyzed
all_GSEA_results_df= pd.concat(all_GSEA_results)
neg_GSEA_results_df = pd.concat(neg_GSEA_results)

# merging real and negative control gsea results to single dataframe with column specifying source
all_GSEA_results_df['source'] = 'real'
neg_GSEA_results_df['source'] = 'negative control'

#Remove separate term row 
all_GSEA_results_df = all_GSEA_results_df.reset_index()
neg_GSEA_results_df = neg_GSEA_results_df.reset_index()

combo_gsea_df = pd.concat([all_GSEA_results_df, neg_GSEA_results_df])

# Define cut-offs
lfc_cutoff = 0.584
fdr_cutoff = 0.25

# Filter data for significant results
significant_gsea_df = all_GSEA_results_df[
    (all_GSEA_results_df['es'].abs() > lfc_cutoff) & 
    (all_GSEA_results_df['fdr'] < fdr_cutoff)
]
significant_negs = neg_GSEA_results_df[
    (neg_GSEA_results_df['es'].abs() > lfc_cutoff) & 
    (neg_GSEA_results_df['fdr'] < fdr_cutoff)
]


# In[9]:


# saving significant gsea results as single output file
significant_gsea_dir = pathlib.Path("./results/significant_gsea_results.parquet.gz")
significant_gsea_df.to_parquet(significant_gsea_dir, compression = 'gzip')

# saving gsea results as single output file
combo_gsea_dir = pathlib.Path("./results/combined_gsea_results.parquet.gz")
combo_gsea_df.to_parquet(combo_gsea_dir, compression = 'gzip')


# In[10]:


# sort by what you want to evaluate
combo_gsea_df.sort_values(by='nes', ascending = True)

idx = significant_gsea_df.groupby('Term')['nes'].idxmax()

# Use the indices to filter the original DataFrame
sig_gsea_no_duplicates = significant_gsea_df.loc[idx].reset_index(drop=True)

sig_gsea_no_duplicates.sort_values(by='nes', key=abs, ascending = False).head(50)


# In[11]:


# Define cut-offs
lfc_cutoff = 0.584
fdr_cutoff = 0.25

    
plt.figure()
plt.scatter(x=all_GSEA_results_df['es'],y=all_GSEA_results_df['fdr'].apply(lambda x:-np.log10(x)),s=10, color='grey')
plt.scatter(x=significant_gsea_df['es'],y=significant_gsea_df['fdr'].apply(lambda x:-np.log10(x)),s=10)
#LFC and FDR lines
plt.axhline(y=-np.log10(fdr_cutoff), color='r', linestyle='--', linewidth=1)
plt.axvline(x=lfc_cutoff, color='g', linestyle='--', linewidth=1)
plt.axvline(x=-lfc_cutoff, color='g', linestyle='--', linewidth=1)
plt.xlabel('log2 Fold Change (ES)')
plt.ylabel('-log10(fdr)')
plt.ylim(0,20)
plt.title('Gene Set Enrichment Analysis')

#save figure
gsea_save_path = pathlib.Path("../1.data-exploration/figures/gsea.png")
plt.savefig(gsea_save_path, bbox_inches="tight", dpi=600)


plt.figure()
plt.scatter(x=neg_GSEA_results_df['es'],y=neg_GSEA_results_df['fdr'].apply(lambda x:-np.log10(x)), s=10, color='grey')
plt.scatter(x=significant_negs['es'],y=significant_negs['fdr'].apply(lambda x:-np.log10(x)),s=10)
#LFC and FDR lines
plt.axhline(y=-np.log10(fdr_cutoff), color='r', linestyle='--', linewidth=1)
plt.axvline(x=lfc_cutoff, color='g', linestyle='--', linewidth=1)
plt.axvline(x=-lfc_cutoff, color='g', linestyle='--', linewidth=1)
plt.xlabel('log2 Fold Change (ES)')
plt.ylabel('-log10(fdr)')
plt.ylim(0,20)
plt.title('Control Gene Set Enrichment Analysis')

#save figure
cgsea_save_path = pathlib.Path("../1.data-exploration/figures/controlgsea.png")
plt.savefig(cgsea_save_path, bbox_inches="tight", dpi=600)


# In[12]:


# Using VAE generated data

pdf_path = pathlib.Path("../1.data-exploration/figures/gsea_plots.pdf")
pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)

# Looping over each dataframe in all_signatures to generate gsea plots for the chosen geneset with data 
# from each latent dimension and saving the plots to a singular pdf
for df in all_signatures:
    col_titles = df.columns.tolist()
    dim = col_titles[1]
    z_result = results[int(dim)-1]

    geneset = "M Phase R-HSA-68886"

    text, ax = plt.subplots()
    ax.text(0.5, 0.5, 'The three following figures visualize the gene set enrichment analysis results for ' + geneset + ' in the latent dimension z=' + dim, fontsize=16, ha='center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    pdf.savefig(text, bbox_inches='tight')
    plt.close()

    fig = blitz.plot.running_sum(df, geneset, library, result=z_result, compact=False)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


    fig_compact = blitz.plot.running_sum(df, geneset, library, result=z_result, compact=True)
    pdf.savefig(fig_compact, bbox_inches='tight')
    plt.close()

    fig_table = blitz.plot.top_table(df, library, z_result, n=15)
    pdf.savefig(fig_table, bbox_inches='tight')
    plt.close()

pdf.close()


# Using negative control

ctrl_pdf_path = pathlib.Path("../1.data-exploration/figures/ctrl_gsea_plots.pdf")
ctrl_pdf = matplotlib.backends.backend_pdf.PdfPages(ctrl_pdf_path)

# Looping over each dataframe in negative_control to generate gsea plots for the chosen geneset with data 
# from each latent dimension and saving the plots to a singular pdf
for df in negative_control:
    col_titles = df.columns.tolist()
    dim = col_titles[1]
    z_result = results[int(dim)-1]

    geneset = "M Phase R-HSA-68886"

    text, ax = plt.subplots()
    ax.text(0.5, 0.5, 'The three following figures visualize the negative control gene set enrichment analysis results for ' + geneset + ' in the latent dimension z=' + dim, fontsize=16, ha='center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ctrl_pdf.savefig(text, bbox_inches='tight')
    plt.close()

    fig = blitz.plot.running_sum(df, geneset, library, result=z_result, compact=False)
    ctrl_pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    fig_compact = blitz.plot.running_sum(df, geneset, library, result=z_result, compact=True)
    ctrl_pdf.savefig(fig_compact, bbox_inches='tight')
    plt.close()

    fig_table = blitz.plot.top_table(df, library, z_result, n=15)
    ctrl_pdf.savefig(fig_table, bbox_inches='tight')
    plt.close()

ctrl_pdf.close()

