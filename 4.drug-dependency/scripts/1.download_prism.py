#!/usr/bin/env python
# coding: utf-8

# ## Download PRISM Repurposing Data
# 
# https://depmap.org/repurposing/
# 
# - Primary screen: 578 cell lines treated with 4,518 compounds
# - Secondary screen: 489 cell lines treated with 1,448 compounds in 8 doses

# In[1]:


import pathlib
import sys

import pandas as pd

sys.path.append("../")
from utils import download_utils


# In[2]:


# For downloading figshare IDs
figshare_url = "https://ndownloader.figshare.com/files/"


# In[3]:


base_output_dir = "data"

download_dict = {
    # Primary screen
    "20237718": {  # Cell line details
        "output_file": "primary-screen-cell-line-info.csv",
        "output_data_dir": "primary_screen",
    },
    "20237715": {  # Compound details
        "output_file": "primary-screen-replicate-collapsed-treatment-info.csv",
        "output_data_dir": "primary_screen",
    },
    "20237709": {  # PRISM readouts replicate collapsed
        "output_file": "primary-screen-replicate-collapsed-logfold-change.csv",
        "output_data_dir": "primary_screen",
    },
    # Secondary screen
    "20237769": {  # Cell line details
        "output_file": "secondary-screen-cell-line-info.csv",
        "output_data_dir": "secondary_screen",
    },
    "20237763": {  # Compound details
        "output_file": "secondary-screen-replicate-collapsed-treatment-info.csv",
        "output_data_dir": "secondary_screen",
    },
    "20237757": {  # PRISM readouts replicate collapsed
        "output_file": "secondary-screen-replicate-collapsed-logfold-change.csv",
        "output_data_dir": "secondary_screen",
    },
}


# In[4]:


for figshare_id in download_dict:
    # Create directories and paths for outputing data
    output_dir = pathlib.Path(
        base_output_dir, download_dict[figshare_id]["output_data_dir"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = pathlib.Path(output_dir, download_dict[figshare_id]["output_file"])

    # Download data from figshare
    print(f"Downloading {output_file}...")

    download_utils.download_figshare(
        figshare_id=figshare_id, output_file=output_file, figshare_url=figshare_url
    )

