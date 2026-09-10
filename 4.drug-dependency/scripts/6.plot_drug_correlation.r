#!/usr/bin/env python
# coding: utf-8

# In[1]:


suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(cowplot))
suppressPackageStartupMessages(library(reticulate))
suppressPackageStartupMessages(library(arrow))


# In[2]:


#Load data
data_dir <- file.path("../4.drug-dependency/results")

results_file <- file.path(data_dir, "combined_latent_drug_correlations.parquet")
drug_results_df <- arrow::read_parquet(
    results_file,
)

glioma_file <- file.path(data_dir, "Diffuse_Glioma_Analysis.parquet")
glioma_df <- arrow::read_parquet(
    glioma_file,
)


# In[3]:


# Source the themes.R file from the utils folder
source("../utils/themes.r")


# In[4]:


drug_results_df$adjusted_p_value <- p.adjust(drug_results_df$p_value, method = "BH")

positive_df <- drug_results_df %>%
    filter(`shuffled` == FALSE)

control_df <- drug_results_df %>%
    filter(`shuffled` == TRUE)


# In[5]:


# Prepare the data
# Filter for the single highest correlation for each model at each dimension
max_corr_df <- drug_results_df %>%
  group_by(model, full_model_z) %>%
  summarize(max_corr = max(abs(`pearson_correlation`), na.rm = TRUE)) %>%
  ungroup()


# In[6]:


# Prepare the data
# Filter for PHGG drug the single highest correlation for each model at each dimension
phgg_corr_df <- drug_results_df %>%
  filter(`drug` == "BRD-K98572433-001-02-9::2.5::HTS") %>%
  group_by(model, full_model_z) %>%
  summarize(max_corr = max(abs(`pearson_correlation`), na.rm = TRUE)) %>%
  ungroup()


# In[7]:


# Prepare the data
# Filter for hepatoblastoma drug the single highest correlation for each model at each dimension
hepato_corr_df <- drug_results_df %>%
  filter(`drug` == "BRD-K11742128-003-23-4::2.5::HTS") %>%
  group_by(model, full_model_z) %>%
  summarize(max_corr = max(abs(`pearson_correlation`), na.rm = TRUE)) %>%
  ungroup()


# In[8]:


# Plot the data
latent_plot <- ggplot(max_corr_df, aes(x = factor(full_model_z), y = max_corr, color = model, fill = model)) +
  geom_point(size = 3, shape = 21) +  # Points on the line
  geom_smooth(aes(group = model), method = "loess", se = TRUE, size = 1, alpha = 0.1) +  # Trend line with shading 
  scale_color_manual(name = "Algorithm", values = model_colors, labels = model_labels) +
  scale_fill_manual(name = "Algorithm", values = model_colors, labels = model_labels) +
  labs(x = "Latent Dimension", y = "Highest Pearson Correlation\nfor Dimension", title = "All Drugs") +
  custom_theme() 


# In[9]:


# Save the plot with custom dimensions
ggsave("./visualize/drug_latent_plot.png", plot = latent_plot, width = 10, height = 8, units = "in")


# In[10]:


# Plot the hepatoblastoma data
hepato_plot <- ggplot(hepato_corr_df, aes(x = factor(full_model_z), y = max_corr, color = model, fill = model)) +
  geom_point(size = 3, shape = 21) +  # Points on the line
  geom_smooth(aes(group = model), method = "loess", se = TRUE, size = 1, alpha = 0.1) +  # Trend line with shading 
  scale_color_manual(name = "Algorithm", values = model_colors, labels = model_labels) +
  scale_fill_manual(name = "Algorithm", values = model_colors, labels = model_labels) +
  labs(x = "Latent Dimension", y = "Highest Pearson Correlation\nfor Dimension", title = "Amonafide") +
  custom_theme


# In[11]:


# Save the hepatoblastoma plot with custom dimensions
ggsave("./visualize/hepatoblastoma_latent_plot.png", plot = hepato_plot, width = 10, height = 8, units = "in")


# In[12]:


#Plot the PHGG data
phgg_plot <- ggplot(phgg_corr_df, aes(x = factor(full_model_z), y = max_corr, color = model, fill = model)) +
  geom_point(size = 3, shape = 21) +  # Points on the line
  geom_smooth(aes(group = model), method = "loess", se = TRUE, size = 1, alpha = 0.1) +  # Trend line with shading 
  scale_color_manual(name = "Algorithm", values = model_colors, labels = model_labels) +
  scale_fill_manual(name = "Algorithm", values = model_colors, labels = model_labels) +
  labs(x = "Latent Dimension", y = "Highest Pearson Correlation\nfor Dimension", title = "Idasanutlin") +
  custom_theme


# In[13]:


# Save the PHGG plot with custom dimensions
ggsave("./visualize/phgg_latent_plot.png", plot = phgg_plot, width = 10, height = 8, units = "in")


# In[14]:


#Normal volcano plot 
combined_df <- combined_df %>%
  mutate(
    passes_qc = abs(pearson_correlation) > 0.25 & -log10(adjusted_p_value) > 2.5
  )

volcano_plot <- ggplot(combined_df, aes(
  x = pearson_correlation,
  y = -log10(adjusted_p_value),
  color = passes_qc,
  fill = passes_qc
)) +
  geom_point(alpha = 0.9, size = 3, shape = 21) +
  geom_vline(xintercept = c(-0.25, 0.25), linetype = "dashed", color = "black", size = 1) +
  geom_hline(yintercept = 1.3, linetype = "dashed", color = "black", size = 1) +
  scale_color_manual(values = c(`TRUE` = "#b80668", `FALSE` = "grey70")) +
  scale_fill_manual(values = c(`TRUE` = "#b80668", `FALSE` = "grey70")) +
  ylim(0, 30) +
  labs(
    x = "Correlation",
    y = "-log10(p-value)",
    title = "Correlation Results\n(PRISM with Real Data)"
  ) +
  theme(legend.position = "none") +
  custom_theme


# In[15]:


# Save the plot with custom dimensions
ggsave("./visualize/drug_volcano_plot.png", plot = volcano_plot, width = 10, height = 8, units = "in")


# In[16]:


#Control volcano plot
control_df <- control_df %>%
  mutate(
    passes_qc = abs(pearson_correlation) > 0.25 & -log10(adjusted_p_value) > 2.5
  )

# Control volcano plot
control_plot <- ggplot(control_df, aes(
  x = pearson_correlation,
  y = -log10(adjusted_p_value),
  color = passes_qc,
  fill = passes_qc
)) +
  geom_point(alpha = 0.9, size = 3, shape = 21) +
  geom_vline(xintercept = c(-0.25, 0.25), linetype = "dashed", color = "black", size = 1) +
  geom_hline(yintercept = 1.3, linetype = "dashed", color = "black", size = 1) +
  scale_color_manual(values = c(`TRUE` = "#b80668", `FALSE` = "grey70")) +
  scale_fill_manual(values = c(`TRUE` = "#b80668", `FALSE` = "grey70")) +
  ylim(0, 30) +
  xlim(-0.6, 0.6) +
  labs(
    x = "Correlation",
    y = "-log10(p-value)",
    title = "Correlation Results\n(PRISM with Shuffled Data)"
  ) +
  theme(legend.position = "none") +
  custom_theme


# In[17]:


# Save the control plot with custom dimensions
ggsave("./visualize/drug_volcano_plot_control.png", plot = control_plot, width = 10, height = 8, units = "in")


# In[18]:


#Manually annotate groups of drugs by In clinical trials, In vitro testing, and No testing
glioma_df$group <- ifelse(glioma_df$name %in% c(
   "gefitinib", "AEE788", "BMS-599626", "osimertinib", 
  "lapatinib", "MEK162", "selumetinib",  "afatinib", "vandetanib", 
  "EGF816", "AZD8330", "trametinib", "CUDC-101", "PD-0325901", 
  "cobimetinib",  "OTX015", "ACY-1215", "OSI-027", 
  "linsitinib", "abemaciclib"), "In clinical trials", 
ifelse(glioma_df$name %in% c(
    "AZD8931", "BVD-523", "AS-703026", "refametinib", "XL388", "WYE-354",
    "ibrutinib", "OSI-420", "ARRY-334543", "tyrphostin-AG-1478", "neratinib", 
  "XL-647", "U-18666A", "BIBU-1361", "I-BET-762", "CH5132799", "dacomitinib", 
  "alpelisib", "SRC-kinase-inhibitor-I", "fenofibrate", "calcitriol", 
  "alfacalcidol", "I-BET151", "medroxyprogesterone-acetate", "mycophenolic-acid", 
  "bosutinib", "triciribine", "3-deazaneplanocin-A", "scriptaid", "tacalcitol", 
  "spironolactone", "tucatinib", "mercaptopurine"), "In vitro results", 
"No testing"))


# In[19]:


# Plot the glioma drug data
glioma_plot <- ggplot(glioma_df, aes(x = correlation, y = `F-statistic`, color = group)) +
  geom_point(size = 3, alpha = 0.7) +  # Adjust size and transparency of points
  theme_minimal() +  # Use a minimal theme for a clean look
  labs(
    title = "Diffuse Glioma Data: Correlation vs F-statistic",
    x = "Correlation",
    y = "F-statistic"
  ) +
  scale_color_manual(values = c("In clinical trials" = "#648FFF", 
                                "In vitro results" = "#FFC20A", 
                                "No testing" = "#D41159")) +  # Custom colors for each group
  theme(
    plot.title = element_text(hjust = 0.5),  # Center the plot title
    legend.title = element_blank(),  # Remove legend title
    legend.position = "top"          # Position legend at the top
  )


# In[20]:


#Save the Glioma drug plot 
ggsave("./visualize/glioma_plot.png", plot = glioma_plot, width = 10, height = 8, units = "in")

