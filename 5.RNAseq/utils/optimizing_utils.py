from typing import Union
import optuna
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def optimize_elasticnet(
    trial: optuna.trial.Trial, 
    X_train: Union[pd.DataFrame, np.ndarray], 
    y_train: Union[pd.Series, np.ndarray], 
    X_val: Union[pd.DataFrame, np.ndarray], 
    y_val: Union[pd.Series, np.ndarray]
) -> float:
    """
    Optimize ElasticNet hyperparameters using Optuna and evaluate the model on validation data.
    
    Parameters:
    trial (optuna.trial.Trial): A single optimization trial instance.
    X_train (pd.DataFrame or np.array): Feature matrix for training.
    y_train (pd.Series or np.array): Target values for training.
    X_val (pd.DataFrame or np.array): Feature matrix for validation.
    y_val (pd.Series or np.array): Target values for validation.
    
    Returns:
    float: The R² score of the model on the validation set.
    """
    alpha = trial.suggest_float('alpha', 1e-5, 10.0)
    l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
    
    # Build and train the ElasticNet model
    model = make_pipeline(StandardScaler(), ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000))
    model.fit(X_train, y_train)
    
    # Evaluate on validation data
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    return r2

def model_training(x_train, x_val, x_test, y_train, y_val, y_test, model_name, latent_dim, init, seed):
    """
    Train and evaluate ElasticNet models for each target variable in y_train, optimizing hyperparameters
    using Optuna and saving the best models.
    
    Parameters:
    x_train (pd.DataFrame or np.array): Feature matrix for training.
    x_val (pd.DataFrame or np.array): Feature matrix for validation.
    x_test (pd.DataFrame or np.array): Feature matrix for testing.
    y_train (pd.DataFrame): Target values for training with multiple columns (one per target variable).
    y_val (pd.DataFrame): Target values for validation with multiple columns.
    y_test (pd.DataFrame): Target values for testing with multiple columns.
    model_name (str): Name of the model for saving purposes.
    latent_dim (int): Number of latent dimensions used.
    init (int): Initialization identifier.
    seed (int): Random seed for reproducibility.
    
    Returns:
    tuple: 
        - test_results_df (pd.DataFrame): DataFrame containing R² scores for each target variable on the test set.
        - final_df (pd.DataFrame): DataFrame containing predicted and actual values for each target variable.
    """
    os.makedirs("joblib", exist_ok=True)  # Ensure joblib directory exists
    results = {}  # Store optimization results
    final_rows = []  # Store data for final DataFrame

    # Train or load models for each column in y_train
    for col in y_train.columns:
        print(f"Processing latent dimension: {col}")
        model_path = os.path.join("joblib", f"elasticnet_{model_name}_dims_{latent_dim}_z_{col}_init_{init}_seed_{seed}.joblib")
        
        if os.path.exists(model_path):
            print(f"Loading pre-trained model for {col} from {model_path}")
            model = joblib.load(model_path)
        else:
            print(f"Training a new model for {col}")
            y_train_col, y_val_col = y_train[col], y_val[col]
            
            # Optimize hyperparameters using Optuna
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: optimize_elasticnet(trial, x_train, y_train_col, x_val, y_val_col), n_trials=100)
            
            results[col] = {'best_params': study.best_params, 'best_score': study.best_value}
            print(f"Best R² score for {col}: {study.best_value:.4f}")
            
            # Train final model with best hyperparameters
            best_params = study.best_params
            model = make_pipeline(StandardScaler(), ElasticNet(alpha=best_params['alpha'], l1_ratio=best_params['l1_ratio'], max_iter=10000))
            model.fit(x_train, y_train_col)
            dump(model, model_path)
            print(f"Model for {col} saved as {model_path}")
    
    # Save optimization results
    if results:
        pd.DataFrame.from_dict(results, orient='index').to_csv('elasticnet_optimization_results.csv')
    
    # Test models
    test_results, mse_results, predicted_values, test_results_list = {}, {}, {}, []
    for col in y_test.columns:
        print(f"Testing for latent dimension: {col}")
        model_path = os.path.join("joblib", f"elasticnet_{model_name}_dims_{latent_dim}_z_{col}_init_{init}_seed_{seed}.joblib")
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            print(f"No model found for {col}. Ensure training was completed successfully.")
            continue
        
        # Evaluate model
        y_test_col = y_test[col]
        y_test_pred = model.predict(x_test)
        r2_test, mse_test = r2_score(y_test_col, y_test_pred), mean_squared_error(y_test_col, y_test_pred)
        test_results[col], mse_results[col], predicted_values[col] = r2_test, mse_test, y_test_pred
        
        # Store test results
        test_results_list.append({'latent_dimension': latent_dim, 'z_dimension': f'z_{col}', 'R2_score': r2_test, 'model': model_name})
        print(f"Test R² score for {col}: {r2_test:.4f}")
        
        # Prepare predicted vs. actual data
        final_rows.append({**{'model': model_name, 'latent_dimension': latent_dim, 'z_dimension': f'z_{col}', 'type': 'predicted'}, **{f'{i}': pred for i, pred in enumerate(y_test_pred)}})
        final_rows.append({**{'model': model_name, 'latent_dimension': latent_dim, 'z_dimension': f'z_{col}', 'type': 'actual'}, **{f'{i}': actual for i, actual in enumerate(y_test_col)}})
    
    return pd.DataFrame.from_dict(test_results_list), pd.DataFrame(final_rows)
