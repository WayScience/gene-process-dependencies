import argparse
import torch
from vanillavae import VanillaVAE, train_vvae, evaluate_vvae  # Import VanillaVAE 
from torch.utils.data import DataLoader, TensorDataset

def get_optimize_args_vvae():
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
        default=10,
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
        default=5e-3,
        type=float,
        help="Maximum learning rate",
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

def objective_vvae(trial, train_tensor, val_tensor, train_df, latent_dim=None):
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
    args = get_optimize_args_vvae()

    # Use provided latent dimension if available, otherwise suggest via Optuna
    if latent_dim is None:
        latent_dim = trial.suggest_int("latent_dim", args.min_latent_dim, args.max_latent_dim)

    # Define hyperparameters
    
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
        "optimizer_type", ["adam", "rmsprop"]
    )
    
    # Create DataLoader
    train_loader = DataLoader(
        TensorDataset(train_tensor), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor), batch_size=batch_size, shuffle=False
    )

    # Initialize VanillaVAE
    model = VanillaVAE(input_dim=train_df.shape[1], latent_dim=latent_dim)
    
    optimizer = get_optimizer_vvae(optimizer_type, model.parameters(), learning_rate)

    # Train and evaluate VanillaVAE
    train_vvae(model, train_loader, optimizer, epochs=epochs)
    val_loss = evaluate_vvae(model, val_loader)

    return val_loss

def get_optimizer_vvae(optimizer_type, model_parameters, learning_rate):
    if optimizer_type == 'adam':
        return torch.optim.Adam(model_parameters, lr=learning_rate)
    elif optimizer_type == 'rmsprop':
        return torch.optim.RMSprop(model_parameters, lr=learning_rate)
