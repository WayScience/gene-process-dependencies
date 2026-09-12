#!/usr/bin/env python
# coding: utf-8

# # Initial exploration of the CRISPR dependency data

# In[1]:


import pathlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import seaborn as sns

warnings.filterwarnings("ignore")


# In[2]:


# Set constants
adult_threshold = 18
liquid_tumors = ["Leukemia", "Lymphoma"]


# In[3]:


# Set i/o paths and files
data_dir = pathlib.Path("../0.data-download/data")
fig_dir = pathlib.Path("figures")
fig_dir.mkdir(exist_ok=True)

# Input files
model_input_file = pathlib.Path(f"{data_dir}/Model.csv")
crispr_input_file = pathlib.Path(f"{data_dir}/CRISPRGeneEffect.csv")
gene_input_file = pathlib.Path(f"{data_dir}/depmap_gene_meta.tsv")

# Output figures
cancer_type_output_figure = pathlib.Path(f"{fig_dir}/sample_cancer_types_bar_chart.png")
age_category_output_figure = pathlib.Path(f"{fig_dir}/age_categories_bar_chart.png")
age_distribution_output_figure = pathlib.Path(
    f"{fig_dir}/sample_age_distribution_plot.png"
)
sex_output_figure = pathlib.Path(f"{fig_dir}/sample_gender_bar_chart.png")
pediatric_cancer_type_output_figure = pathlib.Path(
    f"{fig_dir}/pediatric_sample_cancer_types_bar_chart.png"
)
adult_cancer_type_output_figure = pathlib.Path(
    f"{fig_dir}/adult_sample_cancer_types_bar_chart.png"
)


# In[4]:


# Load model data
model_df = pd.read_csv(model_input_file)

print(model_df.shape)
model_df.head(3)


# In[5]:


# Load gene effect data
gene_dependency_df = pd.read_csv(crispr_input_file)

print(gene_dependency_df.shape)
gene_dependency_df.head(3)


# In[6]:


# Load gene data for subsetting
gene_meta_df = pd.read_csv(gene_input_file, sep="\t")

print(gene_meta_df.shape)
gene_meta_df.head(3)


# ## Describe input data

# In[7]:


# Model.csv visualization
# How many samples from Model.csv?
n_samples_model = len(model_df["ModelID"].unique())
print(f"Number of samples documented in Model.csv: {n_samples_model} \n")

# How many samples from CRISPRGeneDependency.csv?
n_samples_gene = len(gene_dependency_df["ModelID"].unique())
print(f"Number of samples measured in CRISPRGeneDependency.csv: {n_samples_gene} \n")

# Identify which samples are included in both Model.csv and CRISPRGeneDependency.csv
sample_overlap = list(set(model_df["ModelID"]) & set(gene_dependency_df["ModelID"]))

# count the number of samples that overlap in both data sets
print(f"Samples measured in both: {len(sample_overlap)} \n")

# count the number of genes we will use to subset
print(f"Number of genes that pass QC: {gene_meta_df.symbol.nunique()} \n")

# How many different types of cancer?
n_cancer_types = model_df.query("ModelID in @sample_overlap")[
    "OncotreePrimaryDisease"
].nunique()
print(f"Number of Cancer Types: \n {n_cancer_types} \n")


# In[8]:


# Visualize cancer type distribution
cancer_types_bar = (
    gg.ggplot(model_df, gg.aes(x="OncotreePrimaryDisease"))
    + gg.geom_bar(width = 0.5, position = gg.position_dodge2(padding = 0.5))
    + gg.coord_flip()
    + gg.xlab("Primary Disease")
    + gg.ylab("Count")
    + gg.ggtitle("Distribution of cancer types")
    + gg.theme_bw()
    + gg.theme(figure_size = (16, 12))
)

cancer_types_bar.save(cancer_type_output_figure, dpi=500, height=11, width=12)
print(cancer_type_output_figure)

cancer_types_bar


# ## Visualize age categories and distribution

# In[9]:


age_categories_bar = (
    gg.ggplot(model_df, gg.aes(x="AgeCategory"))
    + gg.geom_bar()
    + gg.ggtitle(
        f"Age categories of derived cell lines"
    )
    + gg.theme_bw()
)

age_categories_bar.save(age_category_output_figure, dpi=500)

age_categories_bar


# In[10]:


age_distribution_plot = (
    gg.ggplot(model_df, gg.aes(x="Age"))
    + gg.geom_density()
    + gg.geom_vline(xintercept=adult_threshold, linetype="dashed", color="red")
    + gg.annotate(geom="text", x=30, y = 0.0045, label="Adult Threshold", color="red")    
    + gg.ggtitle(
        f"Age distribution of derived cell lines"
    )
    + gg.theme_bw()
)

age_distribution_plot.save(age_distribution_output_figure, dpi=500)

age_distribution_plot


# In[11]:


model_df['AgeCategory'].value_counts()


# In[12]:


gendersamp_plot = (
    gg.ggplot(model_df, gg.aes(x="Sex"))
    + gg.geom_bar()
    + gg.ggtitle(f"Sex categories of derived cell lines")
    + gg.theme_bw()
)

gendersamp_plot.save(sex_output_figure)

gendersamp_plot


# ## What cell lines are pediatric cancer?

# In[13]:


pediatric_model_df = (
    model_df.query("AgeCategory == 'Pediatric'")
    .query("ModelID in @sample_overlap")
    .reset_index(drop=True)
)

print(pediatric_model_df.shape)
pediatric_model_df.head(3)


# In[14]:


# What are the neuroblastoma models?
pediatric_model_df.query(
    "OncotreeSubtype == 'Neuroblastoma'"
).StrippedCellLineName


# In[15]:


# What is the distribution of pediatric tumor types
pediatric_cancer_counts = pediatric_model_df.OncotreePrimaryDisease.value_counts()
pediatric_cancer_counts


# In[16]:


# What is the distribution of pediatric tumor types
pediatric_cancer_counts = (
    pediatric_model_df
    .OncotreePrimaryDisease
    .value_counts()
    .reset_index()
    .rename(columns={"count": "cancer_type_count"})
)
pediatric_cancer_counts


# In[17]:


# Visualize pediatric cancer type distribution
ped_cancer_types_bar = (
    gg.ggplot(
        pediatric_cancer_counts.reset_index(), gg.aes(x="OncotreePrimaryDisease", y="cancer_type_count")
    )
    + gg.geom_bar(stat="identity")
    + gg.coord_flip()
    + gg.ggtitle("Distribution of pediatric cancer types")
    + gg.ylab("count")
    + gg.xlab("cancer type")
    + gg.theme_bw()
    + gg.theme(figure_size = (14, 6))
)

ped_cancer_types_bar.save(pediatric_cancer_type_output_figure, dpi=500, height=6, width=12)

ped_cancer_types_bar


# In[18]:


# Pediatric solid vs liquid tumors
cancer_types = pediatric_model_df['OncotreePrimaryDisease'].tolist()

ped_liquid = []
ped_non_liquid = []

for cancer_type in cancer_types:
    if liquid_tumors[0] in cancer_type or liquid_tumors[1] in cancer_type:
        ped_liquid.append(cancer_type)
    else:
        ped_non_liquid.append(cancer_type)

print("The number of pediatric solid tumors:")
print(len(ped_non_liquid))

print("The number of pediatric liquid tumors:")
print(len(ped_liquid))


# ## What cell lines are adult cancer?

# In[19]:


adult_model_df = (
    model_df.query("AgeCategory == 'Adult'")
    .query("ModelID in @sample_overlap")
    .reset_index(drop=True)
)

print(adult_model_df.shape)
adult_model_df.head(3)


# In[20]:


# What is the distribution of adult tumor types
adult_cancer_counts = adult_model_df.OncotreePrimaryDisease.value_counts()
adult_cancer_counts


# In[21]:


# What is the distribution of adult tumor types
adult_cancer_counts = (
    adult_model_df
    .OncotreePrimaryDisease
    .value_counts()
    .reset_index()
    .rename(columns={"count": "cancer_type_count"})
)

adult_cancer_counts


# In[22]:


# Visualize adult cancer type distribution
adult_cancer_types_bar = (
    gg.ggplot(
        adult_cancer_counts.reset_index(), gg.aes(x="OncotreePrimaryDisease", y="cancer_type_count")
    )
    + gg.geom_bar(stat="identity")
    + gg.coord_flip()
    + gg.ggtitle("Distribution of adult cancer types")
    + gg.ylab("count")
    + gg.xlab("cancer type")
    + gg.theme_bw()
    + gg.theme(figure_size = (14, 10))
)

adult_cancer_types_bar.save(adult_cancer_type_output_figure, dpi=500, height=10, width=12)

adult_cancer_types_bar


# In[23]:


# Adult solid vs liquid tumors
cancer_types = adult_model_df['OncotreePrimaryDisease'].tolist()

adult_liquid = []
adult_non_liquid = []

for cancer_type in cancer_types:
    if liquid_tumors[0] in cancer_type or liquid_tumors[1] in cancer_type:
        adult_liquid.append(cancer_type)
    else:
        adult_non_liquid.append(cancer_type)

print("The number of adult solid tumors:")
print(len(adult_non_liquid))

print("The number of adult liquid tumors:")
print(len(adult_liquid))

