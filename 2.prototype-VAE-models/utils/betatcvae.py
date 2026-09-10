# betatcvae.py

import math
import pathlib
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class BetaTCVAE(nn.Module):
    """
    Beta Total Correlation Variational Autoencoder (Beta-TCVAE) implementation.

    Attributes:
    -----------
    latent_dim : int
        Dimensionality of the latent space.
    anneal_steps : int
        Number of steps to anneal the KL divergence term.
    alpha : float
        Weight of the mutual information term in the loss function.
    beta : float
        Weight of the total correlation term in the loss function.
    gamma : float
        Weight of the KL divergence term in the loss function.
    encoder : nn.Sequential
        Encoder network.
    fc1 : nn.Linear
        Fully connected layer 1 in the encoder.
    fc_mu : nn.Linear
        Fully connected layer to output the mean of the latent space.
    fc_logvar : nn.Linear
        Fully connected layer to output the log variance of the latent space.
    fc2 : nn.Linear
        Fully connected layer 2 in the decoder.
    fc3 : nn.Linear
        Fully connected layer 3 in the decoder.
    decoder_input : nn.Linear
        Input layer for the decoder.
    decoder : nn.Sequential
        Decoder network.
    final_layer : nn.Linear
        Final layer of the decoder.
    
    Methods:
    --------
    encode(input: torch.Tensor) -> List[torch.Tensor]
        Encodes the input into the latent space.
    decode(z: torch.Tensor) -> torch.Tensor
        Decodes the latent variable into the reconstructed input.
    reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor
        Reparameterization trick to sample from the latent space.
    forward(x: torch.Tensor)
        Forward pass through the network.
    log_density_gaussian(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor)
        Computes the log density of a Gaussian.
    loss_function(*args, **kwargs) -> dict
        Computes the loss for the Beta-TCVAE.
    """
    num_iter=0
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 beta: float,
                 hidden_dims: List = [256, 256, 256],
                 alpha: float = 1.0,
                 gamma: float = 1.0,
                 anneal_steps: int = 200,
                 **kwargs) -> None:
        super(BetaTCVAE, self).__init__()

        # Initialize class attributes
        self.latent_dim = latent_dim
        self.anneal_steps = anneal_steps
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Build Encoder
        modules = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            modules.append(nn.Linear(in_dim, h_dim))
            modules.append(nn.ReLU())
            in_dim = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc1 = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        # Build Decoder
        modules = []
        hidden_dims.reverse() 

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0])
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            modules.append(nn.ReLU())

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Linear(hidden_dims[-1], input_dim)

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        result = self.encoder(input)
        result = self.fc1(result)  # Apply fc1 to match dimensions
        mu = self.fc_mu(result)  # Latent space mean
        log_var = self.fc_logvar(result)  # Latent space log-variance
        return [mu, log_var]

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = self.decoder(result)
        return self.final_layer(result)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed_x = self.decode(z)
        return [reconstructed_x, mu, log_var, z]


    def log_density_gaussian(self, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        normalization = -0.5 * (math.log(2 * math.pi) + logvar) 
        inv_var = torch.exp(-logvar)  
        log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)  
        return log_density

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]  
        input = args[1]  
        mu = args[2]  
        log_var = args[3]  
        z = args[4] 
        m_n = kwargs.get('m_n', 1.0)  # Minibatch size

        weight = 1  # kwargs['m_n']  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input, reduction='sum')  

        log_q_zx = self.log_density_gaussian(z, mu, log_var).sum(dim=1)  # Log q(z|x)

        zeros = torch.zeros_like(z)
        log_p_z = self.log_density_gaussian(z, zeros, zeros).sum(dim=1)  # Log p(z)

        batch_size, latent_dim = z.shape
        mat_log_q_z = self.log_density_gaussian(z.view(batch_size, 1, latent_dim),
                                                mu.view(1, batch_size, latent_dim),
                                                log_var.view(1, batch_size, latent_dim))  # Log q(z)

        dataset_size = (1 / m_n) * batch_size  # Dataset size
        strat_weight = (dataset_size - batch_size + 1) / (dataset_size * (batch_size - 1))  # Stratified weight
        importance_weights = torch.Tensor(batch_size, batch_size).fill_(1 / (batch_size - 1)).to(input.device)
        importance_weights.view(-1)[::batch_size] = 1 / dataset_size  # Importance weights
        importance_weights.view(-1)[1::batch_size] = strat_weight
        importance_weights[batch_size - 2, 0] = strat_weight
        log_importance_weights = importance_weights.log()  # Log importance weights

        mat_log_q_z += log_importance_weights.view(batch_size, batch_size, 1)  # Adjusted log q(z)

        log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)  # Log q(z)
        log_prod_q_z = torch.logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1)  # Log product of q(z)

        mi_loss = (log_q_zx - log_q_z).mean()  # Mutual information loss
        tc_loss = (log_q_z - log_prod_q_z).mean()  # Total correlation loss
        kld_loss = (log_prod_q_z - log_p_z).mean()  # KL divergence loss

        if self.training:
            self.num_iter += 1  # Increment iteration count if training
            anneal_rate = min(0 + 1 * self.num_iter / self.anneal_steps, 1)  # Anneal rate
        else:
            anneal_rate = 1.  # No annealing if not training

        # Total loss with weighted components
        loss = recons_loss + (self.alpha * mi_loss +  
            self.beta * tc_loss +
            anneal_rate * self.gamma * kld_loss)
        
        return loss




    
def train_tc_model(model, train_loader, optimizer):
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
        recon_batch, mu, log_var, z = model(data)
        loss = model.loss_function(recon_batch, data, mu, log_var, z)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    avg_train_loss = train_loss / len(train_loader.dataset)
    return avg_train_loss

def train_tc_vae(model, train_loader, optimizer, epochs):
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
        avg_train_loss = train_tc_model(model, train_loader, optimizer)
        train_loss_history.append(avg_train_loss)
    return train_loss_history

def evaluate_tc_vae(model, val_loader):
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
            recon, mu, log_var, z = model(data)
            z = model.reparameterize(mu, log_var)
            val_loss += model.loss_function(recon, data, mu, log_var, z).item()
    return val_loss / len(val_loader.dataset)

def compile_tc_vae(model, train_loader, val_loader, test_loader, optimizer, epochs):
    """
    Compile the VAE model.

    Args:
        model (VAE): VAE model to be trained.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        test_loader (DataLoader): DataLoader for the testing data.
        optimizer (Optimizer): Optimizer for the model.
        epochs (int, optional): Number of training epochs. Defaults to 5.
    Returns:
        Training history (loss)
        Validation history (loss)
        Testing history (loss)
    """
    model.train()
    train_loss_history = []
    val_loss_history = []
    test_loss_history = []

    for epoch in range(epochs):
        avg_train_loss = train_tc_model(model, train_loader, optimizer)
        train_loss_history.append(avg_train_loss)

        avg_val_loss = evaluate_tc_vae(model, val_loader)
        val_loss_history.append(avg_val_loss)

        avg_test_loss = evaluate_tc_vae(model, test_loader)
        test_loss_history.append(avg_test_loss)

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}, Test Loss: {avg_test_loss}")

    return train_loss_history, val_loss_history, test_loss_history

def tc_extract_latent_dimensions(model: BetaTCVAE, data_loader: DataLoader, metadata: pd.DataFrame, path: pathlib.Path=None) -> pd.DataFrame:
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
            mu, log_var = model.encode(data)  # Extract the mean and log variance
            latent_space.append(mu.cpu().numpy())  # Append the mean to the latent space
    latent_space = np.concatenate(latent_space, axis=0)
    latent_df = pd.DataFrame(latent_space)
    latent_df.insert(0, 'ModelID', metadata['ModelID'])
    if path:
        latent_df_dir = pathlib.Path(path)
        latent_df.to_parquet(latent_df_dir, index=False)
    return latent_df

def tc_weights(model, subset_train_df, path=None):
    """
    Extract weight from the TC VAE model and save them with Model IDs.
    """
    # Ensure model is in evaluation mode
    model.eval()

    # Access the weight matrix corresponding to the latent dimensions
    latent_dim = model.latent_dim  # Assuming model has latent_dim attribute
    weight_matrix = model.encoder[0].weight.detach().cpu().numpy().T  # Extract weights

    # Split weight matrix if it's for both mu and logvar
    weight_matrix = weight_matrix[:, :latent_dim]  # Take only the first half (mu part)

    # Create a dataframe from the weight matrix
    weight_df = pd.DataFrame(weight_matrix)

    if path:
        # Save as parquet to use for heatmap
        weight_df_dir = pathlib.Path("./results/tc_weight_matrix_encoder.parquet")
        weight_df.to_parquet(weight_df_dir)

    # Transpose, reformat, and save the weights
    weight_df_T_df = weight_df.T
    gene_weight_df = pd.DataFrame(data=weight_df_T_df.values, columns=subset_train_df.columns)
    gene_weight_T_df = gene_weight_df.T

    # Reformat and save
    gw_reindex_df = gene_weight_T_df.reset_index()
    gw_renumber_df = gw_reindex_df.rename(columns={x: y for x, y in zip(gw_reindex_df.columns, range(0, len(gw_reindex_df.columns)))})
    
    # Remove numbers from gene name column
    split_data_df = gw_renumber_df[0].str.split(" ", expand=True)
    gene_name_df = split_data_df.iloc[:, :1]
    trimmed_gene_weight_df = gw_renumber_df.iloc[:, 1:]

    final_gene_weights_df = gene_name_df.join(trimmed_gene_weight_df)

    # Save as parquet for GSEA
    if path:
        gene_weight_dir = pathlib.Path(path)
        final_gene_weights_df.to_parquet(gene_weight_dir, index=False)

    return final_gene_weights_df
