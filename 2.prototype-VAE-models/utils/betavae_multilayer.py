# Code based on https://github.com/1Konny/Beta-VAE/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 
import numpy as np
import pathlib


class BetaVAEMulti(nn.Module):
    """
    Beta Variational Autoencoder (VAE) class with multiple layers.

    Args:
        input_dim (int): Dimension of the input data.
        latent_dim (int): Dimension of the latent space.
        beta (float): Weight for the Kullback-Leibler divergence term in the loss function.
        hidden_dim (int): Dimension of the hidden layer. (utilized for multiple layers)
        num_layers (int): Number of layers
    """

    def __init__(self, input_dim: int, latent_dim:int, beta:float, hidden_dim:int, num_layers:int):
        super(BetaVAEMulti, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        # Create encoder with variable number of layers
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        # The final layer outputs mu and log_var
        layers.append(nn.Linear(in_dim, latent_dim * 2))
        self.encoder = nn.Sequential(*layers)

        # Create decoder with variable number of layers
        layers = []
        in_dim = latent_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        # The final layer outputs the reconstructed input
        layers.append(nn.Linear(in_dim, input_dim))
        layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*layers)
        self.latent_dim = latent_dim
        self.beta = beta

    def reparameterize(self, mu: float, log_var: float):
        """
        Reparameterize from N(mu, var) to N(0,1)
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass through the VAE.

        Args:

            x (Tensor): Input data.

        Returns:
            Reconstructed data, mean, and log variance of the latent space.
        """
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var
    
    def encode(self, x):
        """
        Encode the input data to the latent space.

        Args:
            x (Tensor): Input data.

        Returns:
            Mean and log variance of the latent space.
        """
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)
        return mu, log_var

    def loss_function(self, recon_x, x, mu, log_var):
        """
        Compute the VAE loss function (Mean squared error).

        Args:
            recon_x (Tensor): Reconstructed data.
            x (Tensor): Original input data.
            mu (Tensor): Mean of the latent Gaussian.
            log_var (Tensor): Log variance of the latent Gaussian.

        Returns:
            Computed loss.
        """
        MSE = F.mse_loss(recon_x, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return MSE + self.beta * KLD

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

def train_vae(model, train_loader, optimizer, epochs):
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
        print(f"Epoch {epoch}, Loss: {avg_train_loss}")

    return train_loss_history


def evaluate_vae(model, val_loader):
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


def compile_vae(model, train_loader, val_loader, test_loader, optimizer, epochs):
    model.train()
    train_loss_history = []
    val_loss_history = []
    test_loss_history = []

    for epoch in range(epochs):

        avg_train_loss = train_model(model, train_loader, optimizer)
        train_loss_history.append(avg_train_loss)

        avg_val_loss = evaluate_vae(model, val_loader)
        val_loss_history.append(avg_val_loss)

        avg_test_loss = evaluate_vae(model, test_loader)
        test_loss_history.append(avg_test_loss)

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}, Test Loss: {avg_test_loss}")

    return train_loss_history, val_loss_history, test_loss_history



def extract_latent_dimensions(model, data_loader, metadata):
    """
    Extract latent dimensions from the VAE model and save them with Model IDs.

    Args:
        model (VAE): Trained VAE model.
        data_loader (DataLoader): DataLoader for the data.
        metadata (DataFrame): Metadata containing Model IDs.

    Returns:
        DataFrame with latent dimensions and Model IDs.
    """
    model.eval()
    latent_space = []
    with torch.no_grad():
        for batch in data_loader:
            data = batch[0]
            mu, _ = model.encode(data)
            latent_space.append(mu.cpu().numpy())

    latent_space = np.concatenate(latent_space, axis=0)
    latent_df = pd.DataFrame(latent_space)
    latent_df.insert(0, 'ModelID', metadata['ModelID'])

    latent_df_dir = pathlib.Path("./results/latent_df_multi.parquet")
    latent_df.to_parquet(latent_df_dir, index=False)

    return latent_df


def weights(model, subset_train_df):
    """
    Extract weight from the VAE model and save them with Model IDs.

    Args:
        model (VAE): Trained VAE model.
        subset_train_df: the qc training dataframe

    Returns:
        Gene weight dataframe
    """
    weight_matrix = model.encoder[0].weight.detach().cpu().numpy().T  # Transpose the weight matrix
    weight_df = pd.DataFrame(weight_matrix)

    # Save as parquet to use for heatmap
    weight_df_dir = pathlib.Path("./results/weight_matrix_encoder.parquet")
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
    gene_weight_dir = pathlib.Path("./results/weight_matrix_gsea_multi.parquet")
    final_gene_weights_df.to_parquet(gene_weight_dir, index=False)

    return final_gene_weights_df