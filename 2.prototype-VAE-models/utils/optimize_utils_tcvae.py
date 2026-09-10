import argparse

import torch
from betatcvae import BetaTCVAE, train_tc_vae, evaluate_tc_vae
from torch.utils.data import DataLoader, TensorDataset


def get_optimize_args_tc():
    """
    Get arguments for the hyperparameter optimization procedure
    """

    parser = argparse.ArgumentParser()
    # dummy arguments for ipykernel error
    parser.add_argument("-f", "--fff", help="dummy argument", default="1")
    parser.add_argument("--project_name", type=str, help="The name of the project")
    parser.add_argument(
        "--directory",
        default="hyperparameter",
        type=str,
        help="The name of the directory to save results",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="decision to overwrite already started hyperparameter search",
    )
    parser.add_argument(
        "--min_latent_dim",
        default=0,
        type=int,
        help="Minimum size of the internal latent dimensions",
    )
    parser.add_argument(
        "--max_latent_dim",
        default=200,
        type=int,
        help="Maximum size of the internal latent dimensions",
    )
    parser.add_argument(
        "--min_beta",
        default=1,
        type=int,
        help="Minimum beta penalty applied to TC",
    )
    parser.add_argument(
        "--max_beta",
        default=2,
        type=int,
        help="Maximum beta penalty applied to TC",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        nargs="+",
        help="learning rates to use in hyperparameter sweep",
    )
    parser.add_argument(
        "--min_epochs",
        default=5,
        type=int,
        help="Minimum epochs",
    )
    parser.add_argument(
        "--max_epochs",
        default=1000,
        type=int,
        help="Maximum epochs",
    )
    parser.add_argument(
        "--epoch_step",
        default=100,
        type=int,
        help="Epoch step size",
    )
    parser.add_argument(
        "--min_batch_size",
        default=16,
        type=int,
        help="Minimum batch size",
    )
    parser.add_argument(
        "--max_batch_size",
        default=112,
        type=int,
        help="Maximum batch size",
    )
    parser.add_argument(
        "--batch_size_step",
        default=32,
        type=int,
        help="Batch size step size",
    )
    parser.add_argument(
        "--min_lr",
        default=1e-6,
        type=float,
        help="Minimum learning rate",
    )
    parser.add_argument(
        "--max_lr",
        default=1e-4,
        type=float,
        help="Maximum learning rate",
    )
    parser.add_argument(
        "--min_alpha",
        default=0,
        type=float,
        help="Minimum alpha",
    )
    parser.add_argument(
        "--max_alpha",
        default=5.0,
        type=float,
        help="Maximum alpha",
    )
    parser.add_argument(
        "--min_gamma",
        default=0,
        type=float,
        help="Minimum gamma",
    )
    parser.add_argument(
        "--max_gamma",
        default=5.0,
        type=float,
        help="Maximum gamma",
    )
    parser.add_argument(
        "--architecture", 
        default="onelayer", 
        help="VAE architecture")

    parser.add_argument(
        "--dataset", 
        default="cell-painting", 
        help="cell-painting or L1000 dataset"
    )

    args = parser.parse_args()
    return args


def objective_tc(trial, train_tensor, val_tensor, train_df, latent_dim=None):
    """
    Optuna objective function
    Args:
        trial: Optuna trial
        train_tensor: Training data tensor
        val_tensor: Validation data tensor
        train_df: Training dataframe
        latent_dim: Optional latent dimension provided externally

    Returns:
        Validation loss
    """
    val_loss = []
    args = get_optimize_args_tc()

    # Use provided latent dimension if available, otherwise suggest via Optuna
    if latent_dim is None:
        latent_dim = trial.suggest_int("latent_dim", args.min_latent_dim, args.max_latent_dim)

    """
    Optuna objective function: optimized by study
    """
    # Define hyperparameters
    print(latent_dim)
    beta = trial.suggest_float(
        "beta", args.min_beta, args.max_beta
    )
    learning_rate = trial.suggest_float(
        "learning_rate", args.min_lr, args.max_lr
    )
    batch_size = trial.suggest_int(
        "batch_size", args.min_batch_size, args.max_batch_size
    )
    epochs = trial.suggest_int(
        "epochs", args.min_epochs, args.max_epochs
    )
    optimizer_type = trial.suggest_categorical(
        "optimizer_type", ["adam","rmsprop"]
    )

    # Create DataLoader
    train_loader = DataLoader(
        TensorDataset(train_tensor), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor), batch_size=batch_size, shuffle=False
    )

    model = BetaTCVAE(input_dim=train_df.shape[1], latent_dim=latent_dim, beta=beta)
    optimizer = get_optimizer_tc(optimizer_type, model.parameters(), learning_rate)

    train_tc_vae(model, train_loader, optimizer, epochs=epochs)

    val_loss = evaluate_tc_vae(model, val_loader)


    return val_loss

def get_optimizer_tc(optimizer_type, model_parameters, learning_rate):
    if optimizer_type == 'adam':
        return torch.optim.Adam(model_parameters, lr=learning_rate)
    elif optimizer_type == 'rmsprop':
        return torch.optim.RMSprop(model_parameters, lr=learning_rate)