#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pathlib
from sklearn.preprocessing import StandardScaler


# In[2]:


data_directory = pathlib.Path("../6.collab-data/data/").resolve()
rnaseq_file = pathlib.Path(data_directory, "GSE231858_norm_counts_TPM_GRCh38.p13_NCBI.tsv.gz").resolve()
annot_file = pathlib.Path(data_directory, "Human.GRCh38.p13.annot.tsv.gz").resolve()
final_file = pathlib.Path(data_directory, "collaboration_rna_data.parquet")

rnaseq_df = pd.read_csv(rnaseq_file, sep="\t")
rnaseq_parquet = rnaseq_file.with_suffix('.parquet')
rnaseq_df.to_parquet(rnaseq_parquet, index=False)
print(rnaseq_df)

# Read the annotation file
annot_df = pd.read_csv(annot_file, sep="\t")
# Select only the necessary columns
annot_df = annot_df[['GeneID', 'Symbol', 'EnsemblGeneID']]
annot_parquet = annot_file.with_suffix('.parquet')
annot_df.to_parquet(annot_parquet, index=False)

print(annot_df)


# In[3]:


# Assuming you have the two dataframes: df1 (with GeneID, Symbol, EnsemblGeneID) and df2 (gene expression data)

# Step 1: Create the new 'Symbol (GeneID)' column in df1
annot_df['Symbol (GeneID)'] = annot_df['Symbol'] + ' (' + annot_df['GeneID'].astype(str) + ')'

# Step 2: Merge this new column with df2
# Assuming df2 has GeneID as one of its columns, replace GeneID with Symbol (GeneID)
rnaseq_df = rnaseq_df.set_index('GeneID')  # Set the GeneID column as index in df2
rnaseq_df.index.name = 'Symbol (GeneID)'  # Rename index to match the new column name

# Step 3: Map the Symbol (GeneID) values from df1 to df2 index
rnaseq_df.index = rnaseq_df.index.map(annot_df.set_index('GeneID')['Symbol (GeneID)'])

# Step 4: Transpose the dataframe so that Symbol (GeneID) are the column names and SampleIDs are the rows
rna_transposed = rnaseq_df.T

rna_transposed.columns.name = 'SampleID'

# Now df2_transposed will have Symbol (GeneID) as column names and SampleIDs as rows
print(rna_transposed.head())


# In[4]:


# Z-score normalization
scaler = StandardScaler()
zscored_data = pd.DataFrame(
    scaler.fit_transform(rna_transposed),
    columns=rna_transposed.columns,
    index=rna_transposed.index
)
zscored_data.head()


# In[5]:


zscored_data.to_parquet(final_file, index=False)

