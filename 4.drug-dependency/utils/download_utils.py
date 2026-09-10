"""
Functions to facilitate downloading of specific publicly-available resources
"""

import requests
from zipfile import ZipFile
import pathlib


def download_figshare(
    figshare_id: str,
    output_file: pathlib.Path,
    figshare_url: str = "https://ndownloader.figshare.com/files/",
    chunk_size: int = 1024,
) -> pathlib.Path:
    """
    Download the provided figshare resource

    Attributes
    ----------
    figshare_id : str
        string of numbers that corresponds to the figshare identifier
    output_file : pathlib.Path
        the location and file name of where to save the downloaded data
    figshare_url: str, default "https://ndownloader.figshare.com/files/"
        the location of where the figshare id is stored
    chunk_size : int, default 1024
        How many bytes to write at a given time to the output file, for requests stream

    Returns:
    --------
        The output file name
    """
    download_url = f"{figshare_url}/{figshare_id}"

    request_streamer = requests.get(download_url, stream=True)
    with open(output_file, "wb") as output_file_writer:
        for chunk in request_streamer.iter_content(chunk_size=chunk_size):
            output_file_writer.write(chunk)

    return output_file


def download_depmap_bucket(
    file_name: str,
    output_dir: pathlib.Path,
    bucket: str = "depmap-external-downloads",
    resource: str = "pharmacological_profiling",
    chunk_size: int = 1024,
) -> pathlib.Path:
    """
    Download a legacy depmap file not stored on figshare

    Attributes
    ----------
    file_name : str
        name of the file to download, this will also be the name of the downloaded file
    output_dir : pathlib.Path
        the directory of where to save the file (directory only)
    bucket: str, default "depmap-external-downloads"
        the name of the bucket where the DepMap data are stored
    resource: str, default "pharmacological_profiling"
        the category of DepMap resource
    chunk_size : int, default 1024
        How many bytes to write at a given time to the output file, for requests stream

    Returns:
    --------
        The output file name
    """

    # Build the url to retrieve
    base_url = "https://depmap.org/portal/download/api/download"
    file_url = f"?file_name=ccle_legacy_data%2F{resource}%2F{file_name}&bucket={bucket}"

    download_url = f"{base_url}{file_url}"
    output_file = pathlib.Path(output_dir, file_name)

    request_streamer = requests.get(download_url, stream=True)
    with open(output_file, "wb") as output_file_writer:
        for chunk in request_streamer.iter_content(chunk_size=chunk_size):
            output_file_writer.write(chunk)

    return output_file


def download_nci60(
    output_file: pathlib.Path,
    base_url: str,
    attachment_id: str,
    attachment_name: str,
    data_version: str = "6",
    modification_date: str = "1672801037000",
    api_version: str = "v2",
    extract_zip: bool = False,
    chunk_size: int = 1024,
) -> pathlib.Path:
    """
    Download the given nci-60 resource

    Attributes
    ----------
    output_file : pathlib.Path()
        the location to save the downloaded file
    base_url : str
        url to download from
    attachment_id : str
        the unique file identifier
    attachment_name: str
        the name of the file attachment
    data_version: str, default = "6"
        the nci60 data version
    modification_date: str, default "1672801037000"
        the identifier indicating last modification time for version control
    api_version: str, default = "v2"
        the version of the API used to download the file
    extract_zip, bool, default = False
        whether or not to extract the zip file
    chunk_size : int, default 1024
        How many bytes to write at a given time to the output file, for requests stream

    Returns:
    --------
        The output file name

    """
    request_string = f"{base_url}/{attachment_id}/{attachment_name}?version={data_version}&modificationDate={modification_date}&api={api_version}"

    request_streamer = requests.get(request_string, stream=True)
    with open(output_file, "wb") as output_file_writer:
        for chunk in request_streamer.iter_content(chunk_size=chunk_size):
            output_file_writer.write(chunk)

    if extract_zip:
        with ZipFile(output_file) as z:
            z.extractall(path=output_file.parent)

    return output_file
