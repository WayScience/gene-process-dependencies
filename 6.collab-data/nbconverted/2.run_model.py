import pandas as pd
import pathlib
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# Path to ElasticNet models
model_path = pathlib.Path("../5.RNAseq/joblib").resolve()

latent_pred_file = collab_preds_dir / "phgg_latent_predictions.parquet"

# Define the location of the saved models and output directory for results
output_dir = pathlib.Path("../5.RNAseq/results")
output_dir.mkdir(parents=True, exist_ok=True)

# File for combined correlation results
final_test_results_file = output_dir / "test_r2.parquet"
final_test_predictions_file = output_dir /  "test_preds.parquet"

output_dir = pathlib.Path("../4.drug-dependency/results")
output_dir.mkdir(parents=True, exist_ok=True)
final_output_file = output_dir / "combined_latent_drug_correlations.parquet"

# Load new RNA-seq data
rna_path = pathlib.Path("./data").resolve()
rna_file = pathlib.Path(f"{rna_path}/shc_collaboration_rna_data.parquet").resolve()
train = pd.read_parquet("../5.RNAseq/data/RNASeq_train_zscored.parquet")

raw_rna_data = pd.read_parquet("../5.RNAseq/data/RNASeq.parquet")
raw_rna_data.rename(columns={'Unnamed: 0': 'SampleID'}, inplace=True)

new_rnaseq_data = pd.read_parquet(rna_file)
print("New RNA-seq data shape:", new_rnaseq_data.shape)

new_rnaseq_data.head()

# Get the list of columns in the train data
train_columns = train.columns

# Filter the new RNAseq data to include only the columns in the train data
filtered_rnaseq_data = new_rnaseq_data.loc[:, train_columns.intersection(new_rnaseq_data.columns)]

# Add any missing columns with default values
for col in train_columns:
    if col not in filtered_rnaseq_data and col != "SampleID":
        filtered_rnaseq_data[col] = 0

# Assuming shc_rnaseq_data is already a pandas dataframe
# Remove 'SampleID' column if it is not needed for the comparison
data = new_rnaseq_data.copy()

# Calculate the mean for each gene (column)
mean_values = data.mean()

# Calculate the Euclidean distance for each row from the mean values
new_rnaseq_data['Euclidean_Distance'] = np.linalg.norm(data - mean_values, axis=1)

# Scale data
scaler = StandardScaler()
rnaseq_data_scaled = scaler.fit_transform(filtered_rnaseq_data)

# Create an empty list to store DataFrames for each model
all_latent_dfs = []

# Iterate over all files in the saved_models directory
for model_file in model_path.glob("*.joblib"):
    # Extract model name and number of components from the filename
    model_file_name = model_file.stem
    try:
        parts = model_file_name.split("_")
        model_name = parts[1]  # First part is the model name
        dims = int(parts[3])
        init = int(parts[7])
        # Handle the case with multiple `z_` components by ensuring we grab the correct part
        z_value = int(parts[5])
    except (IndexError, ValueError):
        print(f"Skipping file {model_file} due to unexpected filename format.")
        continue

    # Load the model
    print(f"Loading model from {model_file}")
    
    model = joblib.load(model_file)
    # Predict latent variables using the best model
    latent_predictions = model.predict(rnaseq_data_scaled)

    latent_df = pd.DataFrame(latent_predictions, columns=["latent_score"])

    # Add the sample names (row indices) as a new column
    latent_df["ModelID"] = new_rnaseq_data.index

    # Add model name and z_value columns
    latent_df["model"] = model_name
    latent_df["z"] = z_value
    latent_df["latent_dim_total"] = dims
    latent_df["init"] = init

    # Append to the list of DataFrames
    all_latent_dfs.append(latent_df)

# Combine all DataFrames into one
final_latent_df = pd.concat(all_latent_dfs, ignore_index=True)
print("Latent DataFrame shape:", final_latent_df.shape)

collab_preds_dir = pathlib.Path("../6.collab-data/results").resolve()
collab_preds_dir.mkdir(parents=True, exist_ok=True)

final_latent_df.to_parquet(latent_pred_file)

final_latent_df.head(50)

final_test_results_df = pd.read_parquet(final_test_results_file)
final_test_predictions_df = pd.read_parquet(final_test_predictions_file)

combined_results_df = pd.read_parquet(final_output_file)

final_latent_df.sort_values(by='latent_score', ascending=False).head(50)

# Assuming filtered_df is the DataFrame
# Group by 'model', 'z', and 'latent_dim_total' and calculate the variation in 'Latent_0'
variation_df = (
    final_latent_df.groupby(['model', 'z', 'latent_dim_total', 'init'])
    .agg(
        max_latent_score=('latent_score', 'max'),
        min_latent_score=('latent_score', 'min'),
        std_latent_score=('latent_score', 'std'),
    )
    .reset_index()
)

# Add a column for the range of Latent_0
variation_df['range_latent_score'] = variation_df['max_latent_score'] - variation_df['min_latent_score']

# Sort by range_latent_0 or std_latent_0 to find the groups with the biggest variation
sorted_variation_df = variation_df.sort_values(by='range_latent_score', ascending=False)

# Display the top groups
sorted_variation_df.head(50)

#This code describes the variation for scores, checking to see if top range is v low so no interesting results are expected b/w samples