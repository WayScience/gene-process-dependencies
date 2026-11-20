import pathlib
import matplotlib.pyplot as plt
import numpy as np


def plot_pinwheel(sample_id, final_scores, id, name, labels=None):
    save_dir = pathlib.Path("./visualize/pinwheels").resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get pathway names and scores
    ids = final_scores[id].values
    # pathway_score can be negative (see latent_annotation.py's
    # compute_and_plot_latent_scores, which no longer takes abs()) - take
    # the magnitude here specifically because this is the one place that
    # needs a non-negative polar radius, not because the score itself is
    # magnitude-only.
    scores = np.abs(final_scores["pathway_score"].values)

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
