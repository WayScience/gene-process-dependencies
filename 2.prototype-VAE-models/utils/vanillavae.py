import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import pathlib

class VanillaVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VanillaVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.Sigmoid(),
        )
        self.latent_dim = latent_dim

    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = h.chunk(2, dim=-1)  # Split into mean and logvar
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def loss_function(self, recon_x, x, mean, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return recon_loss + kl_div
    

def train_model(model, train_loader, optimizer):
    """
    Definition of the VAE training model for use in both training and compiling
    Args:
        model (VAE): VAE model to be trained.
        train_loader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for the model.
    Returns:
        The average training loss for the current epoch
    """
    model.train()
    train_loss = 0
    for batch in train_loader:
        data = batch[0]
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss = model.loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    avg_train_loss = train_loss / len(train_loader.dataset)
    return avg_train_loss

def train_vvae(model, train_loader, optimizer, epochs):
    """
    Train the VAE model.

    Args:
        model (VAE): VAE model to be trained.
        train_loader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for the model.
        epochs (int, optional): Number of training epochs. Defaults to 5.
    Returns:
        Training history (loss)
    """
    train_loss_history = []
    model.train()
    for epoch in range(epochs):
        avg_train_loss = train_model(model, train_loader, optimizer)
        train_loss_history.append(avg_train_loss)

    return train_loss_history



def evaluate_vvae(model, val_loader):
    """
    Evaluate the VAE model.

    Args:
        model (VAE): VAE model to be evaluated.
        test_loader (DataLoader): DataLoader for the test data.

    Returns:
        Average loss over the test dataset.
    """
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            data = batch[0]
            recon, mu, log_var = model(data)
            val_loss += model.loss_function(recon, data, mu, log_var).item()
    return val_loss / len(val_loader.dataset)

def vvae_extract_latent_dimensions(model: VanillaVAE, data_loader: DataLoader):
    """
    Extracts the latent dimensions (mean values) from the Vanilla VAE's encoder.

    Args:
        model (VAE): Trained VAE model.
        data_loader (DataLoader): DataLoader for the dataset.

    Returns:
        DataFrame: A DataFrame with the latent dimensions (mean values) for each sample.
    """
    model.eval()
    latent_means = []
    with torch.no_grad():
        for batch in data_loader:
            data = batch[0]
            mean, _ = model.encode(data)
            latent_means.append(mean.cpu().numpy())

    latent_means = torch.cat(latent_means, dim=0).numpy()
    latent_df = pd.DataFrame(latent_means)
    
    return latent_df


def vanilla_weights(model, subset_train_df, path=None):
    """
    Extract weight from the VAE model and save them with Model IDs.

    Args:
        model (VAE): Trained VAE model.
        subset_train_df: the qc training dataframe

    Returns:
        Gene weight dataframe
    """
    weight_matrix = model.encoder[0].weight.detach().cpu().numpy().T  # Transpose the weight matrix
    latent_dim = weight_matrix.shape[1] // 2
    weight_matrix = weight_matrix[:, :latent_dim]  # Only take the first half
    weight_df = pd.DataFrame(weight_matrix)

    if path:
        # Save as parquet to use for heatmap
        weight_df_dir = pathlib.Path("./results/vanilla_weight_matrix_encoder.parquet")
        weight_df.to_parquet(weight_df_dir, index=False)

    # Transpose, add gene names back in, transpose again, reset the index, renumber the columns 
    weight_df_T_df = weight_df.T
    gene_weight_df = pd.DataFrame(data=weight_df_T_df.values, columns=subset_train_df.columns)
    gene_weight_T_df = gene_weight_df.T

    gw_reindex_df = gene_weight_T_df.reset_index()
    gw_renumber_df = gw_reindex_df.rename(columns={x: y for x, y in zip(gw_reindex_df.columns, range(0, len(gw_reindex_df.columns)))})

    # Remove numbers from gene name column

    split_data_df = gw_renumber_df[0].str.split(" ", expand=True)
    gene_name_df = split_data_df.iloc[:, :1]
    trimmed_gene_weight_df = gw_renumber_df.iloc[:, 1:]

    final_gene_weights_df = gene_name_df.join(trimmed_gene_weight_df)

    # Save as parquet to use for GSEA
    if path:
        gene_weight_dir = pathlib.Path(path)
        final_gene_weights_df.to_parquet(gene_weight_dir, index=False)

    return final_gene_weights_df
