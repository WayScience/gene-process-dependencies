"""
Data processing utils
"""

import pandas as pd
from typing import Dict


def process_tissue_ccle_column(
    input_df: pd.DataFrame,
    recode_col_name: str = "CCLE Cell Line Name",
    nan_col_to_fill: str = "cell_line_name",
    missing_tissue_dict: Dict = {},
) -> pd.DataFrame:
    """
    Split a cell line id column into tissue and cell line name, thus cleaning

    Arguments:
    ----------
    input_df : pandas.DataFrame
        The data frame containing the column to clean
    recode_col_name : str, default "CCLE Cell Line Name"
        The specific column to clean
    nan_col_to_fill : str, default "cell_line_name"
        Which column to fill missing values, if CCLE name does not exist
    missing_tissue_dict : dict, default {}
        Optional argument to fill in missing details

    Returns:
    --------
        The same input dataframe with a cleaned tissue and cell line column
    """
    cell_line_part = input_df.loc[:, recode_col_name].apply(
        lambda x: str(x).split("_")[0]
    )
    tissue_part = input_df.loc[:, recode_col_name].apply(
        lambda x: "_".join(str(x).split("_")[1:])
    )

    input_df = input_df.assign(cell_line_clean=cell_line_part, tissue=tissue_part)

    # Sometimes the cleaned column doesn't have a corresponding CCLE name
    # In these cases, fill with a column of choice.
    input_df.loc[input_df.cell_line_clean == "nan", "cell_line_clean"] = input_df.loc[
        input_df.cell_line_clean == "nan", nan_col_to_fill
    ]

    if len(missing_tissue_dict) >= 1:
        input_df = process_missing_tissues(
            input_df=input_df,
            recode_col_name=recode_col_name,
            missing_tissue_dict=missing_tissue_dict,
        )

    return input_df


def process_missing_tissues(
    input_df: pd.DataFrame,
    recode_col_name: str = "CCLE Cell Line Name",
    missing_tissue_dict: Dict = {},
) -> pd.DataFrame:
    """
    Fill in missing tissue information

    Arguments:
    ----------
    input_df : pandas.DataFrame
        The data frame containing the column to clean
    recode_col_name : str, default "CCLE Cell Line Name"
        The specific column to subset cell line model from
    missing_tissue_dict : dict, default {}
        Optional argument to fill in missing details

    Returns:
    --------
        The same input dataframe with a cleaned tissue and cell line column
    """
    # Adjust for missing tissue values if provided
    for cell_line in missing_tissue_dict:
        input_df.loc[
            input_df.loc[:, recode_col_name] == cell_line, "tissue"
        ] = missing_tissue_dict[cell_line]

    return input_df
