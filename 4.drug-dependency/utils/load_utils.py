"""
Loading data helper functions
"""

import pathlib
import pandas as pd

from typing import List, Union

from .data_utils import process_tissue_ccle_column, process_missing_tissues

# Set constants
# Note the dictionary key is both the dataset and also the directory
RESOURCE_FILENAMES = {
    "ccle": {"viability": "CCLE_NP24.2009_Drug_data_2015.02.24.csv"},
    "depmap": {
        "viability": "CRISPRGeneDependency.csv",
        "celllines": "Model.csv",
        "genes": "depmap_gene_dictionary.tsv",
    },
    "prism": {
        "primary_screen": {
            "viability": "primary-screen-replicate-collapsed-logfold-change.csv",
            "celllines": "primary-screen-cell-line-info.csv",
            "treatments": "primary-screen-replicate-collapsed-treatment-info.csv",
        },
        "secondary_screen": {
            "viability": "secondary-screen-replicate-collapsed-logfold-change.csv",
            "celllines": "secondary-screen-cell-line-info.csv",
            "treatments": "secondary-screen-replicate-collapsed-treatment-info.csv",
        },
    },
    "nci60": {"viability": "DOSERESP.csv", "treatments": "chemnames_Aug2013.zip"},
}

CCLE_MISSING_TISSUE_NAMES = {
    "M059J": "GLIAL",
    "SF8657": "FIBROBLAST",
    "SNUC2B": "LARGE_INTESTINE",
}

DEPMAP_MISSING_TISSUE_NAMES = {
    "NCC-LMS1-C1": "BONE",
    "NCC-MPNST2-C1": "MPNST",
    "A375_RPMI": "SKIN",
    "RVH421_RPMI": "SKIN",
}


def load_ccle(
    top_dir: str,
    data_dir: str = "ccle/data",
    process_tissues: bool = False,
    only_get_cells: bool = False,
) -> pd.DataFrame:
    """
    Load Cancer Cell Line Encyclopedia (CCLE) viability data

    Arguments:
    ----------
    top_dir : str
        Where the repository top level directory is in relation to your current wd
    data_dir : str, default "ccle/data"
        The intermediate level folder where data are stored
    process_tissues : boolean, default False
        Whether or not to extract and clean tissue from cell line identifiers
    only_get_cells : boolean, default False
        Decision to load only columns (without duplicates) of cell line info

    Returns:
    --------
        The CCLE viability information per cell line : pandas.DataFrame
    """
    # Only get cells works if process_tissues=True
    if only_get_cells and not process_tissues:
        raise Exception("process_tissues must be True if only_get_cells is True")

    # Create path to CCLE data
    ccle_dir = pathlib.Path(top_dir, data_dir)
    ccle_filename = RESOURCE_FILENAMES["ccle"]["viability"]

    ccle_file = pathlib.Path(ccle_dir, ccle_filename)

    # Load data
    ccle_df = pd.read_csv(ccle_file)

    if process_tissues:
        ccle_df = process_tissue_ccle_column(
            input_df=ccle_df,
            recode_col_name="CCLE Cell Line Name",
            nan_col_to_fill="Primary Cell Line Name",
            missing_tissue_dict=CCLE_MISSING_TISSUE_NAMES,
        )

    if only_get_cells:
        cell_line_cols = [
            "CCLE Cell Line Name",
            "Primary Cell Line Name",
            "cell_line_clean",
            "tissue",
        ]
        ccle_df = (
            ccle_df.loc[:, cell_line_cols].drop_duplicates().reset_index(drop=True)
        )

    return ccle_df


def load_depmap(
    top_dir: str,
    data_dir: str = "depmap/data",
    load_cell_info: bool = False,
    load_gene_info: bool = False,
    only_get_cells: bool = False,
) -> Union[List[pd.DataFrame], pd.DataFrame]:
    """
    Load Cancer Dependency Map viability data

    Arguments:
    ----------
    top_dir : str
        Where the repository top level directory is in relation to your current wd
    data_dir : str, default "depmap/data"
        The intermediate level folder where data are stored
    load_cell_info : boolean, default False
        Whether or not to load cell line map
    load_gene_info : boolean, default False
        Whether or not to load gene map
    only_get_cells : boolean, default False
        Decision to load only columns (without duplicates) of cell line info

    Returns:
    --------
        The DepMap viability information per cell line : list of pandas.DataFrame
    """
    # Certain conditions must be met for only_get_cells
    if only_get_cells:
        if not load_cell_info:
            raise ValueError("load_cell_info must be True if only_get_cells is True")
        if load_gene_info:
            raise ValueError("load_gene_info must be False if only_get_cells is True")

    # Create path to DepMap data
    depmap_dir = pathlib.Path(top_dir, data_dir)
    depmap_filename = RESOURCE_FILENAMES["depmap"]["viability"]

    depmap_file = pathlib.Path(depmap_dir, depmap_filename)

    # Load data
    depmap_return_packet = []
    depmap_return_packet.append(pd.read_csv(depmap_file))

    # Append cell info dataframe if specified
    if load_cell_info:
        depmap_cell_filename = RESOURCE_FILENAMES["depmap"]["celllines"]
        depmap_cell_file = pathlib.Path(depmap_dir, depmap_cell_filename)

        depmap_cell_df = pd.read_csv(depmap_cell_file)

        # Clean the CCLE column to facilitate dataset alignment
        depmap_cell_df = process_tissue_ccle_column(
            input_df=depmap_cell_df,
            recode_col_name="CCLE_Name",
            nan_col_to_fill="cell_line_name",
        )

        # Clean entries with missing tissue, according to hand curated info
        depmap_cell_df = process_missing_tissues(
            input_df=depmap_cell_df,
            recode_col_name="cell_line_name",
            missing_tissue_dict=DEPMAP_MISSING_TISSUE_NAMES,
        )

        depmap_return_packet.append(depmap_cell_df)

    # Append gene info dataframe if specified
    if load_gene_info:
        depmap_gene_filename = RESOURCE_FILENAMES["depmap"]["genes"]
        depmap_gene_file = pathlib.Path(depmap_dir, depmap_gene_filename)

        depmap_return_packet.append(pd.read_csv(depmap_gene_file, sep="\t"))

    # Return a pandas dataframe and not a list if asking for only one item
    if len(depmap_return_packet) == 1:
        depmap_return_packet = depmap_return_packet[0]

    # Return only targeted cell line info
    if only_get_cells:
        cell_line_cols = [
            "DepMap_ID",
            "cell_line_name",
            "stripped_cell_line_name",
            "CCLE_Name",
            "cell_line_clean",
            "tissue",
        ]
        depmap_return_packet = (
            depmap_cell_df.loc[:, cell_line_cols]
            .drop_duplicates()
            .reset_index(drop=True)
        )

    return depmap_return_packet


def load_prism(
    top_dir: str,
    data_dir: str = "prism/data",
    secondary_screen: bool = False,
    load_cell_info: bool = False,
    load_treatment_info: bool = False,
    only_get_cells: bool = False,
) -> Union[List[pd.DataFrame], pd.DataFrame]:
    """
    Load PRISM data

    Arguments:
    ----------
    top_dir : str
        Where the repository top level directory is in relation to your current wd
    data_dir : str, default "prism/data"
        The intermediate level folder where data are stored
    secondary_screen : boolean, default False
        Whether or not to load the secondary screen
    load_cell_info : boolean, default False
        Whether or not to load cell line map
    load_treatment_info : boolean, default False
        Whether or not to load treatment map
    only_get_cells : boolean, default False
        Decision to load only columns (without duplicates) of cell line info

    Returns:
    --------
        The PRISM viability information per cell line : list of pandas.DataFrame
    """
    # Certain conditions must be met for only_get_cells
    if only_get_cells:
        if not load_cell_info:
            raise ValueError("load_cell_info must be True if only_get_cells is True")
        if load_treatment_info:
            raise ValueError("load_treatment_info must be False if only_get_cells is True")

    # Create path to DepMap data
    prism_dir = pathlib.Path(top_dir, data_dir)
    prism_dict = RESOURCE_FILENAMES["prism"]

    if secondary_screen:
        screen_key = "secondary_screen"
    else:
        screen_key = "primary_screen"

    prism_dict = prism_dict[screen_key]
    prism_viability_file = prism_dict["viability"]
    prism_viability_file = pathlib.Path(prism_dir, screen_key, prism_viability_file)

    # Load data
    prism_return_packet = []
    prism_return_packet.append(pd.read_csv(prism_viability_file, index_col=0))

    if load_cell_info:
        prism_cell_filename = prism_dict["celllines"]
        prism_cell_file = pathlib.Path(prism_dir, screen_key, prism_cell_filename)

        prism_cell_df = pd.read_csv(prism_cell_file)

        # Clean tissue column based on ccle file name
        prism_cell_df = process_tissue_ccle_column(
            input_df=prism_cell_df,
            recode_col_name="ccle_name",
            nan_col_to_fill="depmap_id",
        )

        # Fix tissue column if missing, control, or fail STR profiling
        prism_cell_df.loc[prism_cell_df.ccle_name.isna(), "tissue"] = prism_cell_df.loc[
            prism_cell_df.ccle_name.isna(), "primary_tissue"
        ].str.upper()
        prism_cell_df.loc[
            prism_cell_df.row_name.str.contains("FAILED_STR"), "tissue"
        ] = "FAILED_STR"
        prism_cell_df.loc[
            prism_cell_df.row_name.str.contains("CONTROL_BARCODE"), "tissue"
        ] = "CONTROL"

        prism_return_packet.append(prism_cell_df)

    if load_treatment_info:
        prism_treatment_filename = prism_dict["treatments"]
        prism_treatment_filename = pathlib.Path(
            prism_dir, screen_key, prism_treatment_filename
        )

        prism_return_packet.append(pd.read_csv(prism_treatment_filename))

    # Return a pandas dataframe and not a list if asking for only one item
    if len(prism_return_packet) == 1:
        prism_return_packet = prism_return_packet[0]

    # Return only targeted cell line info
    if only_get_cells:
        # Remove controls and cell lines that failed STR
        prism_cell_df = prism_cell_df.loc[
            ~prism_cell_df.row_name.str.contains("CONTROL"), :
        ]
        prism_cell_df = prism_cell_df.loc[
            ~prism_cell_df.row_name.str.contains("FAILED"), :
        ]

        # Subset to only cell line related columns, drop duplicates, and reset index
        cell_line_cols = [
            "row_name",
            "depmap_id",
            "ccle_name",
            "cell_line_clean",
            "tissue",
        ]
        prism_return_packet = (
            prism_cell_df.loc[:, cell_line_cols]
            .drop_duplicates()
            .reset_index(drop=True)
        )

    return prism_return_packet


def load_nci60(
    top_dir: str,
    data_dir: str = "nci60/data",
    load_treatment_info: bool = False,
    only_get_cells: bool = False,
) -> Union[List[pd.DataFrame], pd.DataFrame]:
    """
    Load NCI-60 viability data

    Arguments:
    ----------
    top_dir : str
        Where the repository top level directory is in relation to your current wd
    data_dir : str, default "nci60/data"
        The intermediate level folder where data are stored
    load_treatment_info : boolean, default False
        Whether or not to load treatment map
    only_get_cells : boolean, default False
        Decision to load only columns (without duplicates) of cell line info

    Returns:
    --------
        The NCI60 viability information per cell line : pandas.DataFrame or list
    """
    # Create path to CCLE data
    nci60_dir = pathlib.Path(top_dir, data_dir)
    nci60_filename = RESOURCE_FILENAMES["nci60"]["viability"]

    nci60_filename = pathlib.Path(nci60_dir, nci60_filename)

    # Load data
    nci_60_return_packet = []

    nci60_df = pd.read_csv(nci60_filename)

    # Only select cell columns and remove everything else
    if only_get_cells:
        cell_line_cols = [
            "PANEL_NUMBER",
            "CELL_NUMBER",
            "PANEL_NAME",
            "CELL_NAME",
            "PANEL_CODE",
        ]
        nci60_df = (
            nci60_df.loc[:, cell_line_cols].drop_duplicates().reset_index(drop=True)
        )

    nci_60_return_packet.append(nci60_df)

    if load_treatment_info:
        nci60_treatment_filename = RESOURCE_FILENAMES["nci60"]["treatments"]
        nci60_treatment_filename = pathlib.Path(nci60_dir, nci60_treatment_filename)

        nci_60_return_packet.append(
            pd.read_csv(
                nci60_treatment_filename,
                sep="|",
                header=None,
                names=["nsc_number", "cpd_name", "cpd_name_type"],
            )
        )

    # Return a pandas dataframe and not a list if asking for only one item
    if len(nci_60_return_packet) == 1:
        nci_60_return_packet = nci_60_return_packet[0]

    return nci_60_return_packet
