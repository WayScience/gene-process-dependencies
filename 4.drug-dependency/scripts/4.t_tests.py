#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import ttest_ind, f_oneway

sys.path.append("../")
from utils import load_utils


# In[2]:


# Load PRISM data
top_dir = "../4.drug-dependency"
data_dir = "data"

prism_df, prism_cell_df, prism_trt_df = load_utils.load_prism(
    top_dir=top_dir,
    data_dir=data_dir,
    secondary_screen=False,
    load_cell_info=True,
    load_treatment_info=True,
)

print(prism_df.shape)
prism_df.head(3)


# In[3]:


#Load Model data
data_dir = pathlib.Path("../0.data-download/data")
model_input_file = pathlib.Path(f"{data_dir}/Model.csv")
model_df = pd.read_csv(model_input_file)

print(model_df.shape)
model_df.head(3)


# In[4]:


#Load correlation data
#Load Model data
correlation_data_dir = pathlib.Path("../4.drug-dependency/results/")
correlation_input_file = pathlib.Path(f"{correlation_data_dir}/drug_correlation.parquet")
correlation_df = pd.read_parquet(correlation_input_file)

print(correlation_df.shape)
correlation_df.head(3)


# In[5]:


# Merge drug_df with model_df on ModelID to add the OncotreePrimaryDisease column
drug_df_with_disease = prism_df.merge(model_df[['ModelID', 'OncotreePrimaryDisease']], left_index=True, right_on='ModelID')


# In[6]:


# Prepare a DataFrame to store t-test results
ttest_results = []

# Loop through each OncotreePrimaryDisease type
for disease in drug_df_with_disease['OncotreePrimaryDisease'].unique():
    print(f"Processing {disease}...")

    # Filter the drug matrix for the current disease
    disease_drug_df = drug_df_with_disease[drug_df_with_disease['OncotreePrimaryDisease'] == disease].drop(columns=['ModelID', 'OncotreePrimaryDisease'])

    # Filter the drug matrix for the rest of the dataset (all other diseases)
    other_drug_df = drug_df_with_disease[drug_df_with_disease['OncotreePrimaryDisease'] != disease].drop(columns=['ModelID', 'OncotreePrimaryDisease'])

    # Perform t-tests comparing current disease vs the rest for each drug
    t_test_results = ttest_ind(disease_drug_df, other_drug_df, axis=0, nan_policy='omit')
    t_test_results_df = pd.DataFrame({
        "drug": prism_df.columns,
        "t_stat": t_test_results.statistic,
        "p_value": t_test_results.pvalue
    })

    # Filter significant drugs based on p-value < 0.05
    significant_drugs = t_test_results_df[t_test_results_df['p_value'] < 0.05]

    # Perform ANOVA on these drugs
    for drug in significant_drugs['drug'].unique():
        # Extract drug responses for the current drug
        disease_drug_responses = disease_drug_df[drug].dropna()
        other_drug_responses = other_drug_df[drug].dropna()

        # Perform ANOVA if both groups have sufficient data
        if len(disease_drug_responses) > 1 and len(other_drug_responses) > 1:
            f_statistic, p_value = f_oneway(disease_drug_responses, other_drug_responses)
            higher_group = disease if disease_drug_responses.mean() > other_drug_responses.mean() else "Other Types"

            # Store the results
            ttest_results.append({
                'OncotreePrimaryDisease': disease,
                'Drug': drug,
                'F-statistic': f_statistic,
                'p-value': p_value,
                'Higher in': higher_group
            })

# Convert results to DataFrame
ttest_results_df = pd.DataFrame(ttest_results)

# Apply a significance threshold (e.g., p < 0.05)
significant_ttest_results_df = ttest_results_df[ttest_results_df['p-value'] < 0.05]

# Display the top 50 significant results based on F-statistic
significant_ttest_results_df.sort_values(by='F-statistic', key=abs, ascending=False).head(50)


# In[7]:


# Assuming 'drug_column_name' is the column in prism_trt_df that matches the 'drug' column in correlation_df
prism_trt_df_filtered = prism_trt_df[['column_name', 'name', 'moa', 'target']]

# Merge correlation_df with prism_trt_df based on the 'drug' column in correlation_df and the matching column in prism_trt_df
merged_df = pd.merge(significant_ttest_results_df, prism_trt_df_filtered, how='left', left_on='Drug', right_on='column_name')

# Drop the redundant drug_column_name column after the merge if needed
merged_df = merged_df.drop(columns=['column_name'])

# Save results to a parquet file
merged_df.to_parquet("../4.drug-dependency/results/drug_diff_results.parquet", index=False)


# In[8]:


# Assuming merged_df and correlation_df are already defined
correlation_df_filtered = correlation_df[['drug', 'latent_dimension', 'correlation', 'indication','phase']]

# Get unique values from "Higher in" column and exclude "Other types"
higher_in_values = merged_df["Higher in"].unique()
higher_in_values = [value for value in higher_in_values if value != "Other Types"]

# Filter for 'Diffuse Glioma' in the 'Higher in' column
diffuse_glioma_df = merged_df[merged_df["Higher in"] == "Diffuse Glioma"]

# Merge with correlation_df on the "Drug" column for Diffuse Glioma
diffuse_glioma_final_df = pd.merge(diffuse_glioma_df, correlation_df_filtered, left_on="Drug", right_on="drug", how="inner")

# Save the Diffuse Glioma DataFrame to a Parquet file
diffuse_glioma_parquet_path = pathlib.Path('../4.drug-dependency/results/Diffuse_Glioma_Analysis.parquet').resolve()
diffuse_glioma_final_df.to_parquet(diffuse_glioma_parquet_path)

#Path for PDF file
pdf_path = pathlib.Path('../4.drug-dependency/results/Individual_Drug_Analysis.pdf').resolve()

# Create a PDF file to save the plots
with PdfPages(pdf_path) as pdf:
    for value in higher_in_values:
        # Filter rows for the current "Higher in" value
        filtered_df = merged_df[merged_df["Higher in"] == value]

        # Merge with correlation_df on the "Drug" column
        final_df = pd.merge(filtered_df, correlation_df_filtered, left_on="Drug", right_on="drug", how="inner")

        # Apply significance threshold for F-statistic
        final_df = final_df[final_df['F-statistic'] > 7]
        final_df = final_df[final_df['correlation'].abs() > 0.13]

        # Sort by absolute correlation
        sorted_df = final_df.loc[final_df["correlation"].abs().sort_values(ascending=False).index]

        # If sorted_df is not empty, create a page with the dataframe
        if not sorted_df.empty:
            fig, ax = plt.subplots(figsize=(10, sorted_df.shape[0] * 0.5))  # Adjust the height based on the number of rows
            ax.axis('tight')
            ax.axis('off')

            # Create a table from the dataframe
            table = ax.table(cellText=sorted_df.values, colLabels=sorted_df.columns, cellLoc='center', loc='center')

            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(5, 5)  # Adjust the scale of the table for better readability

            # Set the title of the page
            plt.title(f'Analysis for {value}', fontsize=10)
            print(f'Analysis for {value}')

            # Save the page to the PDF
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

             # Create a new page with the list of unique drug names
            unique_drugs = sorted_df['name'].unique()

            fig, ax = plt.subplots(figsize=(10, len(unique_drugs) * 0.25))
            ax.axis('off')  # No axes

            # Display the unique drugs as text
            drug_list_text = '\n'.join(unique_drugs)
            ax.text(0.5, 0.95, drug_list_text, va='top', ha='center', fontsize=15)

            # Add title for unique drugs list
            plt.title(f'Unique Drugs for {value}', fontsize=25)

            # Save this page to the PDF
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

