import joblib
import torch
import pathlib
import numpy as np
import random
from pathlib import Path

def save_model(trial, model, directory, modelname, latent_dims, init, seed):
    """
    Save the model state_dict or the full model based on its type.
    
    Args:
        trial: Current Optuna trial object or string for non-Optuna cases
        model: The model being optimized
        directory: The directory where the models will be saved
        modelname: Name of the model (e.g., "pca", "ica", etc.)
        latent_dims: Number of latent dimensions
        init: Initialization number (e.g., 0 to 4)
        seed: Random seed used for the initialization
    """
    # Ensure the save directory exists
    directory = Path(directory)
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

    # Handle trial number for Optuna trials or use "non-optuna" for others
    trial_number = trial.number if hasattr(trial, "number") else "non-optuna"

    # Define the model save path
    model_save_path = directory / f"{modelname}_latent_dims_{latent_dims}_trial_{trial_number}_init_{init}_seed_{seed}.pt"
    
    # Save the model - assuming the model has a 'state_dict' attribute
    if hasattr(model, "state_dict"):
        import torch
        torch.save(model.state_dict(), model_save_path)
    else:
        import pickle
        with open(model_save_path, "wb") as f:
            pickle.dump(model, f)


def set_random_seed(seed):
    """
    Set the random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    