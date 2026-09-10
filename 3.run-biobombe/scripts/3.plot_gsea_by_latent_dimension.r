
## Visualizing GSEA Results for Latent Dimensions Across Models
# 
# This script loads GSEA results for various latent dimension models, extracts the highest enrichment scores (ES) for 
# each model and dimension, and generates visualizations. The plots include the maximum ES score across all pathways as well 
# as a specific pathway. The results are saved as PNG files.

suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(cowplot))
suppressPackageStartupMessages(library(reticulate))




#Load data
data_dir <- file.path("../3.run-biobombe/gsea_results")

results_file <- file.path(data_dir, "combined_z_matrix_gsea_results.parquet")
gsea_results_df <- arrow::read_parquet(
    results_file,
)



# Source the themes.R file from the utils folder
source("../utils/themes.r")



# Prepare the data
# Filter for the single highest ES score for each model at each dimension
max_es_df <- gsea_results_df %>%
  group_by(model, full_model_z) %>%
  summarize(max_es = max(abs(`gsea_es_score`), na.rm = TRUE)) %>%
  ungroup()



# Prepare the data
# Filter for some specific pathway the single highest ES score for each model at each dimension
path_es_df <- gsea_results_df %>%
  filter(`reactome_pathway` == "Cardiac Conduction R-HSA-5576891") %>%
  group_by(model, full_model_z) %>%
  summarize(max_es = max(abs(`gsea_es_score`), na.rm = TRUE)) %>%
  ungroup()



# Plot the data
latent_plot <- ggplot(max_es_df, aes(x = factor(full_model_z), y = log(max_es), color = model, fill = model)) +
  geom_point(size = 3, shape = 21) +  # Points on the line
  geom_smooth(aes(group = model), method = "loess", se = TRUE, size = 1, alpha = 0.1) +  # Trend line with shading 
  scale_color_manual(name = "Algorithm", values = model_colors, labels = model_labels) +
  scale_fill_manual(name = "Algorithm", values = model_colors, labels = model_labels) +
  labs(x = "Latent Dimensions", y = "Highest ES Score", title = "Highest ES Score Across All Pathways by Latent Dimension for Each Model") +
  custom_theme()




# Save the plot with custom dimensions
ggsave("./visualize/latent_plot.png", plot = latent_plot, width = 10, height = 8, units = "in")




# Plot the data
path_plot <- ggplot(path_es_df, aes(x = factor(full_model_z), y = log(max_es), color = model, fill = model)) +
  geom_point(size = 3, shape = 21) +  # Points on the line
  geom_smooth(aes(group = model), method = "loess", se = TRUE, size = 1, alpha = 0.1) +  # Trend line with shading 
  scale_color_manual(name = "Algorithm", values = model_colors, labels = model_labels) +
  scale_fill_manual(name = "Algorithm", values = model_colors, labels = model_labels) +
  labs(x = "Latent Dimensions", y = "Highest ES Score", title = "Highest ES Score for Regulation of Cardiac Conduction Pathway by Latent Dimension for Each Model") +
  theme(legend.position = "right")




# Save the plot with custom dimensions
ggsave("./visualize/cardiac_conduction_latent_plot.png", plot = path_plot, width = 10, height = 8, units = "in")

