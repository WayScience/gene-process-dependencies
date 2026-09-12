#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pathlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, "../utils/")
from data_loader import load_data
from sklearn.model_selection import train_test_split
import random


# In[2]:


def scale_dataframe(df: pd.DataFrame):
    """
    Scales the gene effect data columns of a DataFrame to a 0-1 range.
    The first column (ID) and the last two columns (age and sex) are not scaled.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The scaled DataFrame.
    """
    col_num = df.shape[1]
    df_to_scale = df.iloc[:, 1:col_num-1]
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_df = scaler.fit_transform(df_to_scale)
    
    scaled_df = pd.DataFrame(scaled_df)
    scaled_df.insert(0, df.columns[0], df[df.columns[0]])
    scaled_df.insert(col_num-1, df.columns[col_num-1], df[df.columns[col_num-1]])
    scaled_df.columns = df.columns
    
    return scaled_df


# In[3]:


def save_dataframe(df, file_path: pathlib.Path):
    """
    Saves a DataFrame to a specified file path.

    Parameters:
    df (pd.DataFrame): The DataFrame to save.
    file_path (str): The file path to save the DataFrame.
    """
    df = df.reset_index(drop=True)
    df.to_parquet(file_path, index=False)
    print(f"DataFrame saved to {file_path}. Shape: {df.shape}")
    print(df.head(3))


# In[4]:


random.seed(18)
print(random.random())


# In[5]:


# load all of the data
data_directory = "../0.data-download/data/"
model_df, effect_df = load_data(data_directory, adult_or_pediatric="all")


# In[6]:


# verifying that the ModelIDs in model_df and effect_df are alligned
model_df["ID_allignment_verify"] = np.where(
    effect_df["ModelID"] == model_df["ModelID"], "True", "False"
)
verrify = len(model_df["ID_allignment_verify"].unique())
print(model_df["ID_allignment_verify"])
print(
    f"There is {verrify} output object contained in the ID_allignment_verify column \n"
)


# In[7]:


# assign 'AgeCategory' and 'Sex' columns to the effect dataframe as a single column
presplit_effect_df = effect_df.assign(
    age_and_sex=model_df.AgeCategory.astype(str) + "_" + model_df.Sex.astype(str)
)
presplit_effect_df


# In[8]:


groups = model_df.groupby("AgeCategory")
df_list = []
for name, df in groups:

    # only looking for samples that contain Adult or Pediatric information
    if name == "Adult" or name == "Pediatric":
        df_list.append(df)

# merge sample dataframes through concatentation and reorganize so that ModelIDs are in alphabetical order
new_df = pd.concat(df_list, axis=0)
new_df = new_df.set_index("ModelID")
new_df = new_df.sort_index(ascending=True)
new_df = new_df.reset_index()


# In[9]:


# creating a list of ModelIDs that correlate to pediatric and adult samples
PA_effect_IDs = new_df["ModelID"].tolist()

PA_IDs = set(PA_effect_IDs) & set(presplit_effect_df["ModelID"].tolist())

# creating a new gene effect data frame containing correlating ModelIDs to the filtered sample info IDs
PA_effect_df = presplit_effect_df.loc[
    presplit_effect_df["ModelID"].isin(PA_IDs)
].reset_index(drop=True)


# In[10]:


# split the data based on age category and sex
train_df, testandvalidation_df = train_test_split(
    PA_effect_df, test_size=0.3, stratify=PA_effect_df.age_and_sex
)
train_df.reset_index(drop=True,inplace=True)
testandvalidation_df.reset_index(drop=True,inplace=True)
test_df, val_df = train_test_split(
    testandvalidation_df, test_size=0.5
)
test_df.reset_index(drop=True,inplace=True)
val_df.reset_index(drop=True,inplace=True)


# In[11]:


#save each dataframe
save_dataframe(train_df, pathlib.Path("data/VAE_train_df.parquet").resolve())
save_dataframe(test_df, pathlib.Path("data/VAE_test_df.parquet").resolve())
save_dataframe(val_df, pathlib.Path("data/VAE_val_df.parquet").resolve())


# In[16]:


# create a data frame of both test and train gene effect data with sex, AgeCategory, and ModelID for use in later t-tests
# load in the data

# create dataframe containing the genes that passed an initial QC (see Pan et al. 2022) and a saturated signal qc, then extracting their corresponding gene label
gene_dict_df = pd.read_parquet("../0.data-download/data/CRISPR_gene_dictionary.parquet")
gene_list_passed_qc = gene_dict_df.loc[gene_dict_df["qc_pass"], 'dependency_column'].tolist()
concat_frames = [train_df, test_df, val_df]
train_and_test = pd.concat(concat_frames).reset_index(drop=True)
train_and_test[["AgeCategory", "Sex"]] = train_and_test.age_and_sex.str.split(
    pat="_", expand=True
)
train_and_test_subbed = train_and_test.filter(gene_list_passed_qc, axis=1)
metadata_holder = pd.DataFrame()
metadata = metadata_holder.assign(
    ModelID=train_and_test.ModelID.astype(str),
    AgeCategory=train_and_test.AgeCategory.astype(str),
    Sex=train_and_test.Sex.astype(str),
)

metadata_df_dir = pathlib.Path("data/metadata_df.parquet").resolve()
metadata.to_parquet(metadata_df_dir, index=False)

train_and_test_subbed_dir = pathlib.Path("data/train_and_test_subbed.parquet").resolve()
train_and_test_subbed.to_parquet(train_and_test_subbed_dir, index=False)

