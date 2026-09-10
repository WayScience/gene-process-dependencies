import pandas as pd    


def find_drugs(predictions_df, r2_df, drug_df):
    """
    Append top drug predictions to a DataFrame based on high R² scores and matching latent dimensions.

    Parameters:
    -----------
    predictions_df : pd.DataFrame
        DataFrame containing model predictions, actual values, and associated metadata (e.g., latent dimensions).
    r2_df : pd.DataFrame
        DataFrame with R² scores for each latent dimension and model combination.
    drug_df : pd.DataFrame
        DataFrame with Pearson correlations between drugs and latent features.

    Returns:
    --------
    pd.DataFrame
        The input predictions_df with additional rows representing top drug associations
        for each qualifying latent dimension (based on R² score > 0.2).
    """
    # Filter the test results for R² score over 0.2
    filtered_r2_df = r2_df[r2_df['R2_score'] > 0.2]

    # Copy the predictions DataFrame to avoid modifying the original
    pred_df = predictions_df.copy()

    # Initialize a results dictionary to store the top correlations for each case
    results = {}

    # Loop through each row in the filtered R² DataFrame
    for _, row in filtered_r2_df.iterrows():
        latent_dim = row['latent_dimension']
        z_dim = row['z_dimension']  # Use extracted numeric z
        z_numeric = int(z_dim.split('_')[1])  # Extract numeric part of 'z_' (e.g., 'z_5' -> 5)
        model = row['model']
        init = row['init']
        
        # Filter the correlation DataFrame based on the specified conditions
        filtered_df = drug_df[
            (drug_df['full_model_z'] == latent_dim) &
            (drug_df['z'] == z_numeric) &
            (drug_df['model'] == model) &
            (drug_df['init'] == init)
        ]

        if not filtered_df.empty:
            # Select the row with the highest absolute Pearson correlation
            top_drug_row = filtered_df.loc[filtered_df['pearson_correlation'].abs().idxmax()]
            
            # Create a new row with type 'drug' and the selected drug
            new_drug_row = pd.DataFrame([{
                'model': model,
                'latent_dimension': latent_dim,
                'z_dimension': f'z_{z_numeric}',  # Ensure 'z_' format
                'type': 'drug',
                'drug': top_drug_row['drug'],
                'pearson_correlation': top_drug_row['pearson_correlation'],
                'init': init,
            }])

            # Append the new row to the dataframe
            pred_df = pd.concat([pred_df, new_drug_row], ignore_index=True)

            # Set 'drug' column to 'n/a' for 'predicted' and 'actual' rows as there is no value attached
            pred_df.loc[
                (pred_df['model'] == model) &
                (pred_df['latent_dimension'] == latent_dim) &
                (pred_df['z_dimension'] == f'z_{z_numeric}') &
                (pred_df['init'] == init) &
                (pred_df['type'].isin(['predicted', 'actual'])),
                'drug'
            ] = 'n/a'

            # Store result for verification
            results[(latent_dim, z_dim, model, init)] = top_drug_row
        else:
            print(f"No drug found for latent_dimension: {latent_dim}, z_dimension: {z_dim}, model: {model}, init: {init}")

    # Print out top drug results
    for key, top_drug_row in results.items():
        latent_dim, z_dim, model, init = key
        print(f"Top drug for latent_dimension: {latent_dim}, z_dimension: {z_dim}, model: {model}, init: {init}")
        print(top_drug_row)
        print()

    return pred_df
