#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pathlib 
import sys
import joblib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import seaborn as sns
import matplotlib.pyplot as plt

script_directory = pathlib.Path("../utils/").resolve()
sys.path.insert(0, str(script_directory))
from data_loader import load_model_data
from model_utils import extract_latent_dims


# In[2]:


# Define the location of the saved models and output directory for results
model_save_dir = pathlib.Path("../4.gene-expression-signatures/saved_models")
output_dir = pathlib.Path("results")
output_dir.mkdir(parents=True, exist_ok=True)


# In[3]:


data_directory = pathlib.Path("../0.data-download/data").resolve()
dependency_file = pathlib.Path(f"{data_directory}/CRISPRGeneEffect.parquet").resolve()
gene_dict_file = pathlib.Path(f"{data_directory}/CRISPR_gene_dictionary.parquet").resolve()


# In[4]:


# Load metadata
metadata_df_dir = pathlib.Path("../0.data-download/data/metadata_df.parquet")
metadata = pd.read_parquet(metadata_df_dir)
print(metadata.shape)

#Load dependency data
dependency_df, gene_dict_df = load_model_data(dependency_file, gene_dict_file)
dependency_df.head()


# In[5]:


# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Apply the scaler to the numeric columns
dependency_df[dependency_df.select_dtypes(include=['float64', 'int']).columns] = scaler.fit_transform(
    dependency_df.select_dtypes(include=['float64', 'int'])
)


# In[6]:


train_and_test_subbed_dir = pathlib.Path("../0.data-download/data/train_and_test_subbed.parquet")
train_and_test_subbed = pd.read_parquet(train_and_test_subbed_dir)

train_and_test_subbed[train_and_test_subbed.select_dtypes(include=['float64', 'int']).columns] = scaler.fit_transform(
    train_and_test_subbed.select_dtypes(include=['float64', 'int'])
)

# Convert DataFrame to NumPy and then Tensor
train_test_array = train_and_test_subbed.to_numpy()
train_test_tensor = torch.tensor(train_test_array, dtype=torch.float32)

#Create TensorDataset and DataLoader
tensor_dataset = TensorDataset(train_test_tensor)
train_and_test_subbed_loader = DataLoader(tensor_dataset, batch_size=32, shuffle=False)


# In[7]:


results = []

for model_file in model_save_dir.glob("*.joblib"):
    model_file_name = model_file.stem
    try:
        parts = model_file_name.split("_")
        model_name = parts[0]
        num_components = int(parts[3])  # total latent dimensions
        init = int(parts[7])  # initialization value
        seed = int(parts[9])  # seed value
    except (IndexError, ValueError):
        print(f"Skipping file {model_file} due to unexpected filename format.")
        continue

    # Load the model
    print(f"Loading model from {model_file}")
    try:
        model = joblib.load(model_file)
    except Exception as e:
        print(f"Failed to load model from {model_file}: {e}")
        continue
        
    # Extract z, original input, and reconstruction
    latent_df, original_data, reconstructed_data = extract_latent_dims(
        model_name, model, dependency_df, train_and_test_subbed_loader, metadata
    )
    print(original_data)
    print(reconstructed_data)

    # Convert to torch tensors
    original_tensor = torch.tensor(original_data, dtype=torch.float32)
    reconstructed_tensor = torch.tensor(reconstructed_data, dtype=torch.float32)

    # Clamp reconstructions to avoid log(0)
    reconstructed_tensor = torch.clamp(reconstructed_tensor, min=1e-7, max=1 - 1e-7)

    # Compute BCE loss across all elements
    mse = F.mse_loss(reconstructed_tensor, original_tensor, reduction='mean')

    results.append({
        "model": model_name,
        "latent_dim": num_components,
        "init": init,
        "mse": mse.item()
    })
    print("Original min/max:", original_data.min(), original_data.max())
    print("Reconstructed min/max:", reconstructed_data.min(), reconstructed_data.max())

# Convert results to DataFrame
recon_df = pd.DataFrame(results)
print(recon_df)


# In[8]:


# Set global font sizes
plt.rcParams.update({
    "font.size": 16,          # Base font size
    "axes.titlesize": 18,     # Facet title
    "axes.labelsize": 16,     # Axis labels
    "xtick.labelsize": 7,    # X tick labels
    "ytick.labelsize": 14,    # Y tick labels
    "legend.fontsize": 14,    # Legend text
    "legend.title_fontsize": 16  # Legend title
})

# Convert latent_dim to a categorical type for equal spacing
recon_df['latent_dim'] = recon_df['latent_dim'].astype(str)

# Convert latent_dim to ordered categories for within_model_df
dimension_order = sorted(recon_df["latent_dim"].unique(), key=int)

recon_df["latent_dim"] = pd.Categorical(
    recon_df["latent_dim"], categories=dimension_order, ordered=True
)

# FacetGrid
g = sns.FacetGrid(
    recon_df,
    col="model",
    col_wrap=3,
    height=4,
    sharey=True
)

g.map_dataframe(
    sns.scatterplot,
    x="latent_dim", 
    y="mse",
    hue="init",
    style="init"
)

# Rotate x-axis labels
for ax in g.axes.flatten():
    for label in ax.get_xticklabels():
        label.set_rotation(90)

# Format
g.set_titles(col_template="{col_name}")
g.set_axis_labels("Latent Dimension", "Reconstruction MSE")
g.add_legend(title="Init")
g.tight_layout()
plt.subplots_adjust(top=0.9, wspace=0.3)

