#!/usr/bin/env python
# coding: utf-8

# Run this script with `2.prototype-VAE-models/` as the working directory — its relative paths (`../2.prototype-VAE-models/utils/`, `../utils/`) only resolve correctly from there.

# In[1]:


import sys
import pathlib
import numpy as np
import pandas as pd
import optuna
import torch
import logging 

script_directory = pathlib.Path("../2.prototype-VAE-models/utils/").resolve()
sys.path.insert(0, str(script_directory))
from optimize_utils import get_optimize_args, objective
from optimize_utils_tcvae import get_optimize_args_tc, objective_tc

script_directory = pathlib.Path("../utils/").resolve()
sys.path.insert(0, str(script_directory))
from data_loader import load_train_test_data

import hiplot
from optuna.visualization import plot_param_importances
import plotly.io as pio


# In[2]:


# Load command line arguments
args = get_optimize_args()
tc_args = get_optimize_args_tc()

# Load data
data_directory = pathlib.Path("../1.data-exploration/data").resolve()

train_data, test_data, val_data, load_gene_stats = load_train_test_data(
    data_directory, train_or_test="all", load_gene_stats=True, zero_one_normalize=True
)


# In[3]:


# Convert dataframes to tensors
train_tensor = torch.tensor(train_data, dtype=torch.float32)
test_tensor = torch.tensor(test_data, dtype=torch.float32)
val_tensor = torch.tensor(val_data, dtype=torch.float32)


# In[4]:


# Run Optuna optimization and save study
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "BetaVAE-Optimization"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(study_name=study_name, storage=storage_name, direction="minimize", load_if_exists=True)
study.optimize(
    lambda trial: objective(trial, train_tensor, val_tensor, train_data), n_trials=500
)


# In[5]:


# Save best hyperparameters
best_trial = study.best_trial
print(best_trial)
print(f"Best trial: {best_trial.values}")
print(f"Best hyperparameters: {best_trial.params}")


# In[6]:


#Plot and save hyperparameter importance
save_path = pathlib.Path("../1.data-exploration/figures/param_importance.png").resolve()
figure = plot_param_importances(study)
pio.write_image(figure, save_path)


# In[7]:


#Create a hiplot for the parameters 
hiplot.Experiment.from_optuna(study).display()


# In[12]:


# Run Optuna optimization for Beta TC VAE and save study
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "BetaTCVAE-Optimization"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(study_name=study_name, storage=storage_name, direction="minimize", load_if_exists=True)
study.optimize(
    lambda trial: objective_tc(trial, train_tensor, val_tensor, train_data), n_trials=500
)


# In[13]:


# Save best hyperparameters
best_trial = study.best_trial
print(best_trial)
print(f"Best trial: {best_trial.values}")
print(f"Best hyperparameters: {best_trial.params}")


# In[14]:


#Plot and save hyperparameter importance
save_path = pathlib.Path("../1.data-exploration/figures/tc_param_importance.png").resolve()
figure = plot_param_importances(study)
pio.write_image(figure, save_path)


# In[15]:


#Create a hiplot for the parameters 
hiplot.Experiment.from_optuna(study).display()

