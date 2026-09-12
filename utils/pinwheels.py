import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_pinwheel(sample_id, final_scores, id, name, labels=None):
    save_dir = pathlib.Path("./visualize/pinwheels").resolve()
    
    # Get pathway names and scores
    ids = final_scores[id].values
    scores = final_scores["pathway_score"].values

    # Convert to radians for plotting
    num_pathways = len(ids)
    angles = np.linspace(0, 2 * np.pi, num_pathways, endpoint=False).tolist()

    # Create polar plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.fill(angles, scores, color="mediumpurple", alpha=0.8)


    # Remove angle labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if labels:
        label_indices = [i for i, val in enumerate(ids) if val in labels]
    else:
        label_indices = np.argsort(scores)[-5:]
    
    # Add labels for the top N scores
    for i in label_indices:
        ax.text(
            angles[i], scores[i], ids[i],
            horizontalalignment="center", 
            verticalalignment="center", 
            fontsize=14, 
            color="black", 
            rotation=0
        )

    plt.title(f"{name} Graph for {sample_id}", y=1.1, fontsize=22)
    
    # Save the plot to the specified directory
    plot_path = pathlib.Path(save_dir, f"pinwheel_{name}_{sample_id}.png")
    plt.savefig(plot_path, bbox_inches='tight')  # Save with tight bounding box
    plt.close()  # Close the plot to free memory

def compute_and_plot_latent_scores(model_id, latent_df, comp_df, id, score, name, make_plot=True):
    """
    Computes pathway scores based on latent dimensions and generates a pinwheel plot.

    Parameters:
        model_id (str): Identifier for the model.
        latent_df (DataFrame): DataFrame containing latent dimension scores.
        comp_df (DataFrame): DataFrame containing pathway or drug comparison scores.
        id (str): Column name for pathway or drug identifiers.
        score (str): Column name for the score used in calculations.
        name (str): Type of plot - e.g. pathway or drug.
        make_plot (bool): Whether to generate and save a pinwheel plot. Default True,
            matching prior behavior; pass False to compute scores without plotting.
    """
    # Filter by ModelID
    latent_filtered = latent_df[latent_df["ModelID"] == model_id].copy()
    
    # Melt latent dataframe
    latent_long = latent_filtered.melt(
        id_vars=["ModelID", "model", "latent_dim_total", "init", "seed"],
        var_name="z",  # Latent dimension names
        value_name="latent_score"
    )
    
    # Convert z from string to int
    latent_long["z"] = latent_long["z"].astype(str).str.replace(r"^z_", "", regex=True).astype(int)
        
    # Prepare GSEA/drug dataframe
    comp_long = comp_df.rename(columns={"full_model_z": "latent_dim_total"}).copy()
    comp_long = comp_long[[id, "latent_dim_total", "init", "model", "z", score]]
    comp_long["z"] = comp_long["z"].astype(str).str.replace(r"^z_", "", regex=True).astype(int)

    # Merge latent and GSEA data while keeping only one "model" and "init"
    merged_df = latent_long.merge(
        comp_long, 
        on=["z", "model", "latent_dim_total", "init"],
        how="inner"
    )

    # Compute final pathway scores. Using merged_df[score] here, not comp_long[score]:
    # merged_df is the result of an inner join and has its own fresh index, which
    # doesn't line up with comp_long's original index or length. Multiplying against
    # comp_long[score] directly silently misaligns rows (and can introduce NaNs) rather
    # than raising, so it must use the copy of the column that actually survived the merge.
    merged_df["pathway_score"] = abs(merged_df["latent_score"]) * merged_df[score]
    merged_df = merged_df.dropna(subset=["pathway_score"])

    print(model_id)

    if make_plot:
        plot_pinwheel(model_id, merged_df, id, name)

    return merged_df
    

def assign_unique_latent_dims(df, score_col, target_col, latent_col="z"):
    """
    Assigns each pathway/drug to the highest available latent dimension without replacement.

    Parameters:
    - df (pd.DataFrame): DataFrame containing pathways/drugs and their corresponding latent dimensions.
    - score_col (str): Column name for the enrichment score (e.g., "gsea_es_score", "drug_es_score").
    - target_col (str): Column name for the pathway or drug.
    - latent_col (str): Column name for the latent dimension (default: "z").

    Returns:
    - pd.DataFrame: A filtered DataFrame with unique latent dimensions assigned to each pathway/drug.
    """
    # Check to see if score is pearson correlation or not, so the ordering is the correct direction. GSEA results are non-directional, pearson correlation is
    if score_col != "pearson_correlation":
        df[score_col] = df[score_col].abs()
        is_drug = False
    else:
        is_drug = True

    # Sort so that the most important values are first (highest abs for GSEA, most negative for pearson correlation)
    df_sorted = df.sort_values(score_col, ascending=is_drug).copy()

    # Track used latent dimensions and their associated pathways
    used_latents = {}
    assigned_rows = []

    # Group by pathway or drug to process each one sequentially
    grouped = df_sorted.groupby(target_col)

    sorted_groups = sorted(grouped, key=lambda group: group[1][score_col].max(), reverse=True)

    for pathway, group in sorted_groups:
        print(f"Processing pathway: {pathway}")

        # Filter out rows with already used latent dimensions
        available_latents = group[~group.apply(
            lambda r: (r[latent_col], r['model'], r['full_model_z'], r['init']) in used_latents.keys(), axis=1)]

        if available_latents.empty:
            print(f"No unused latent dimensions left for pathway {pathway}.")
            continue  # Skip to the next pathway

        # Sort by GSEA score (descending) to pick the top available latent dimension
        best_latent = available_latents.sort_values(by=score_col, ascending=is_drug).head(1).iloc[0]

        # Record the assignment
        latent_key = (best_latent[latent_col], best_latent['model'], best_latent['full_model_z'], best_latent['init'])
        used_latents[latent_key] = pathway  # Assign the latent dimension to the pathway
        assigned_rows.append(best_latent)  # Add to assigned rows

        print(f"Assigned latent dimension {best_latent[latent_col]} for model {best_latent['model']}, {best_latent['full_model_z']}, init {best_latent['init']} to pathway {pathway}.")

    # Convert list of assigned rows back to DataFrame
    return pd.DataFrame(assigned_rows)
