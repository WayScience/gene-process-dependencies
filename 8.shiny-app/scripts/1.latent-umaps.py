#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pathlib
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import ipywidgets as widgets
from sklearn.decomposition import PCA
import random
import colorsys


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


def make_dropdown_pca_with_selection(df, title="PCA Interactive Plot"):
    out = widgets.Output()  # Create a fresh output for this plot!

    # Prepare PCA input
    pca_input = df.drop(columns=["OncotreePrimaryDisease", "source", "ModelID"], errors="ignore").fillna(0)
    pca = PCA(n_components=2, random_state=0)
    embedding = pca.fit_transform(pca_input)

    # Add PCA coordinates to the dataframe
    df["PCA1"] = embedding[:, 0]
    df["PCA2"] = embedding[:, 1]

    # Prepare color map for each cancer type
    cancer_types = df["OncotreePrimaryDisease"].unique()
    color_map = px.colors.qualitative.Plotly + px.colors.qualitative.Light24 + px.colors.qualitative.Dark24
    highlight_color_map = {cancer: color_map[i % len(color_map)] for i, cancer in enumerate(cancer_types)}

    # Create one trace per cancer type
    traces = []
    for cancer in cancer_types:
        df_subset = df[df["OncotreePrimaryDisease"] == cancer]
        trace = go.Scattergl(
            x=df_subset["PCA1"],
            y=df_subset["PCA2"],
            mode='markers',
            name=cancer,
            marker=dict(size=7, color=highlight_color_map[cancer]),
            text=[f"{cancer} | {model_id}" for model_id in df_subset["ModelID"]],
            customdata=df_subset["ModelID"],  # Attach ModelID to customdata
            hoverinfo='text'
        )
        traces.append(trace)

    # Dropdown buttons to highlight cancer types
    dropdown_buttons = []
    for i, cancer in enumerate(cancer_types):
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

    # Default color button
    default_colors = [highlight_color_map[cancer] for cancer in cancer_types]
    dropdown_buttons.insert(0, dict(
        method="update",
        label="Show All",
        args=[
            {"visible": [True] * len(cancer_types),
             "marker": [{'color': default_colors[j]} for j in range(len(cancer_types))]},
            {"title": title}
        ]
    ))

    fig = go.Figure(data=traces)

    for i, trace in enumerate(fig.data):
        trace.marker.color = default_colors[i]

    fig.update_layout(
        title=title,
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
        dragmode='lasso',  # Ensure Lasso is enabled
        width=1500,
        height=800
    )


    return fig, out  # Return both the figure and the output widget for selection


# In[2]:


cancer_type_input_file = pathlib.Path("../0.data-download/data/Model.parquet")
cancer_type_df = pd.read_parquet(cancer_type_input_file)

reactome_dims = pathlib.Path("../5.drug-dependency/results/all_reactome_results.parquet")
reactome_df = pd.read_parquet(reactome_dims)

corum_dims = pathlib.Path("../5.drug-dependency/results/all_corum_results.parquet")
corum_df = pd.read_parquet(corum_dims)

drug_dims = pathlib.Path("../5.drug-dependency/results/all_drug_results.parquet")
drug_df = pd.read_parquet(drug_dims)


# In[3]:


reactome_df.head()


# In[4]:


# Step 2: Subset based on matching keys
subset_keys = ["model", "latent_dim_total", "init", "z"]

# Replace latent identifiers with pathway
reactome_df["feature"] = reactome_df["reactome_pathway"]
reactome_df = reactome_df.drop(columns=["model", "latent_dim_total", "init", "seed", "z", "reactome_pathway"])

corum_df["feature"] = corum_df["reactome_pathway"]
corum_df = corum_df.drop(columns=["model", "latent_dim_total", "init", "seed", "z", "reactome_pathway"])

drug_df["feature"] = drug_df["name"]
drug_df = drug_df.drop(columns=["model", "latent_dim_total", "init", "seed", "z", "name"])


# In[5]:


drug_df_sorted = drug_df.sort_values(by=["OncotreePrimaryDisease", "ModelID", "feature"], ascending=True)
drug_df_sorted.head()


# In[ ]:


# Create a unique column name by combining feature and source
meta_cols = ["ModelID", "OncotreePrimaryDisease"]
reactome_meta = reactome_df[meta_cols].drop_duplicates()
corum_meta = corum_df[meta_cols].drop_duplicates()
drug_meta = drug_df[meta_cols].drop_duplicates()

# Pivot to wide format
reactome_matrix = reactome_df.pivot(index="ModelID", columns="feature", values="latent_score")
corum_matrix = corum_df.pivot(index="ModelID", columns="feature", values="latent_score")
drug_matrix = drug_df.pivot(index="ModelID", columns="feature", values="latent_score")
display(corum_matrix.isna().sum().sum())

# Join metadata back to each matrix
reactome_matrix = reactome_matrix.merge(reactome_meta, on="ModelID", how="left")
corum_matrix = corum_matrix.merge(corum_meta, on="ModelID", how="left")
drug_matrix = drug_matrix.merge(drug_meta, on="ModelID", how="left")


# In[12]:


all_pairs = pd.MultiIndex.from_product(
    [
        corum_df["ModelID"].unique(),
        corum_df["feature"].unique()
    ],
    names=["ModelID", "feature"]
)

observed_pairs = pd.MultiIndex.from_frame(
    corum_df[["ModelID", "feature"]]
)

missing = all_pairs.difference(observed_pairs)

print(len(missing))
missing[:10]


# In[7]:


print("Pathway rows with NA latent scores:")
display(reactome_df[reactome_df["latent_score"].isna()])

print("Drug rows with NA latent scores:")
display(drug_df[drug_df["latent_score"].isna()])

print("CORUM rows with NA latent scores:")
display(corum_df[corum_df["latent_score"].isna()])


# In[10]:


# Assuming combined_df is your full dataset
reactome_fig, reactome_out = make_dropdown_pca_with_selection(reactome_matrix, "PCA: Reactome Subset")
corum_fig, corum_out = make_dropdown_pca_with_selection(corum_matrix, "PCA: CORUM Subset")
drug_fig, drug_out = make_dropdown_pca_with_selection(drug_matrix, "PCA: Drug Subset")

# Show plots
reactome_fig.show()
display(reactome_out)

corum_fig.show()
display(corum_out)

drug_fig.show()
display(drug_out)

# Save HTMLs
reactome_fig.write_html("pca_reactome.html")
corum_fig.write_html("pca_corum.html")
drug_fig.write_html("pca_drug.html")

