#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import pathlib
import sys
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random
import colorsys


script_directory = pathlib.Path("../utils/").resolve()
sys.path.insert(0, str(script_directory))
from data_loader import load_model_data, load_train_test_data


# In[ ]:


def generate_random_palette(num_colors, seed=12):
    # Generate random colors with varied lightness and saturation
    random.seed(seed)
    
    colors = []
    
    for _ in range(num_colors):
        h = random.random()  # Random hue between 0 and 1
        l = random.uniform(0.2, 0.8)  # Lightness between 0.3 and 0.9 for contrast
        s = random.uniform(0.5, 1.0)  # Saturation between 0.6 and 1.0 for vivid colors
        color = colorsys.hls_to_rgb(h, l, s)  # Convert HLS to RGB color
        colors.append(color)
    
    return colors


# In[ ]:





# In[2]:


# Load dependency data
data_directory = pathlib.Path("../0.data-download/data").resolve()
dependency_file = pathlib.Path(f"{data_directory}/CRISPRGeneEffect.parquet").resolve()
gene_dict_file = pathlib.Path(f"{data_directory}/CRISPR_gene_dictionary.parquet").resolve()
dependency_df, gene_dict_df= load_model_data(dependency_file, gene_dict_file)
dependency_df = dependency_df.set_index("ModelID")


# In[3]:


cancer_type_input_file = pathlib.Path("../0.data-download/data/Model.parquet")
cancer_type_df = pd.read_parquet(cancer_type_input_file)


# In[4]:


combined_df = dependency_df.merge(
    cancer_type_df[["ModelID", "OncotreePrimaryDisease"]], 
    on="ModelID", 
    how="left"
)


# In[5]:


combined_df.head()


# In[6]:


gene_cols = combined_df.columns.drop("ModelID")
gene_cols = gene_cols.drop("OncotreePrimaryDisease")


# In[7]:


features = combined_df[gene_cols]


# In[8]:


features


# In[ ]:


# Prepare PCA input (excluding non-numeric columns)
pca_input = combined_df[gene_cols].apply(pd.to_numeric, errors="coerce")
pca = PCA(n_components=2, random_state=0)
pca_embedding = pca.fit_transform(pca_input)

# Add PCA components to the dataframe
combined_df["PCA1"] = pca_embedding[:, 0]
combined_df["PCA2"] = pca_embedding[:, 1]

# Prepare color map for each cancer type
cancer_types = combined_df["OncotreePrimaryDisease"].unique()
color_map = px.colors.qualitative.Plotly + px.colors.qualitative.Light24 + px.colors.qualitative.Dark24
highlight_color_map = {cancer: color_map[i % len(color_map)] for i, cancer in enumerate(cancer_types)}

# Create one trace per cancer type
traces = []
for cancer in cancer_types:
    df_subset = combined_df[combined_df["OncotreePrimaryDisease"] == cancer]
    trace = go.Scatter(
        x=df_subset["PCA1"],
        y=df_subset["PCA2"],
        mode='markers',
        name=cancer,
        marker=dict(size=7),
        text=[f"{cancer} | {model_id}" for model_id in df_subset["ModelID"]],
        hoverinfo='text',
    )
    traces.append(trace)

# Create dropdown buttons
dropdown_buttons = []

for i, cancer in enumerate(cancer_types):
    # Make all grey except the selected one
    visibility = [True] * len(cancer_types)
    colors = ['lightgrey'] * len(cancer_types)
    colors[i] = highlight_color_map[cancer]

    button = dict(
        method="update",
        label=cancer,
        args=[
            {"visible": visibility,
             "marker": [{'color': colors[j]} for j in range(len(cancer_types))]},
            {"title": f"PCA Highlighted: {cancer}"}
        ]
    )
    dropdown_buttons.append(button)

# Add a button to show all with normal colors
default_colors = [highlight_color_map[cancer] for cancer in cancer_types]
dropdown_buttons.insert(0, dict(
    method="update",
    label="Show All",
    args=[
        {"visible": [True]*len(cancer_types),
         "marker": [{'color': default_colors[j]} for j in range(len(cancer_types))]},
        {"title": "PCA of DepMap Single Gene Data"}
    ]
))

# Create the figure
fig = go.Figure(data=traces)

# Set default colors
for i, trace in enumerate(fig.data):
    trace.marker.color = default_colors[i]

fig.update_layout(
    title="PCA of DepMap Single Gene Data",
    xaxis_title="PCA1",
    yaxis_title="PCA2",
    updatemenus=[{
        "buttons": dropdown_buttons,
        "direction": "down",
        "showactive": True,
        "x": 1.15,
        "xanchor": "left",
        "y": 1.15,
        "yanchor": "top"
    }],
    width=1500,
    height=800
)

fig.show()
fig.write_html("achilles_pca.html")

