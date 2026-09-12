#!/usr/bin/env python
# coding: utf-8

# ## Download RNAseq Data
# 
# Source: [Cancer Dependency Map resource](https://depmap.org/portal/download/).
# 
# - `CRISPRGeneDependency.csv`: The data in this document describes the probability that a gene knockdown has an effect on cell-inhibition or death. These probabilities are derived from the data contained in CRISPRGeneEffect.csv using methods described [here](https://doi.org/10.1101/720243)
# - `
# 
# >Tsherniak A, Vazquez F, Montgomery PG, Weir BA, Kryukov G, Cowley GS, Gill S, Harrington WF, Pantel S, Krill-Burger JM, Meyers RM, Ali L, Goodale A, Lee Y, Jiang G, Hsiao J, Gerath WFJ, Howell S, Merkel E, Ghandi M, Garraway LA, Root DE, Golub TR, Boehm JS, Hahn WC. Defining a Cancer Dependency Map. Cell. 2017 Jul 27;170(3):564-576.

# In[1]:


import pathlib
import urllib.request
import pandas as pd
from pathlib import Path
import pyarrow as pa


# In[2]:


def download_dependency_data(figshare_id, figshare_url, output_file):
    """
    Download the provided figshare resource
    """
    urllib.request.urlretrieve(f"{figshare_url}/{figshare_id}", output_file)


# In[3]:


# Set download constants
output_dir = pathlib.Path("data")
figshare_url = "https://ndownloader.figshare.com/files/"

download_dict = {
    "46493242": "RNASeq.csv",

     # DepMap, Broad (2024). DepMap 24Q2 Public. Figshare+. Dataset. https://doi.org/10.25452/figshare.plus.25880521.v1
}


# In[4]:


# Make sure directory exists
output_dir.mkdir(exist_ok=True)


# In[5]:


for figshare_id in download_dict:
    # Set output file
    output_file = pathlib.Path(output_dir, download_dict[figshare_id])

    # Download the dependency data
    print(f"Downloading {output_file}...")

    download_dependency_data(
        figshare_id=figshare_id, figshare_url=figshare_url, output_file=output_file
    )


# In[6]:


#Convert to parquet
# List of CSV files

data_directory = "../5.RNAseq/data/"
csv_file = pathlib.Path(data_directory, "RNASeq.csv").resolve()


df = pd.read_csv(csv_file)
    
# Define the output Parquet file name
parquet_file = csv_file.with_suffix('.parquet')
    
# Save the DataFrame as a Parquet file
df.to_parquet(parquet_file, index=False)


# In[7]:


print(df)

