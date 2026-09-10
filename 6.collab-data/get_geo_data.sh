#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p data

# Define full file URLs
URLS=(
    "https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc=GSE231858&format=file&file=GSE231858_norm_counts_TPM_GRCh38.p13_NCBI.tsv.gz"
    "https://www.ncbi.nlm.nih.gov/geo/download/?format=file&type=rnaseq_counts&file=Human.GRCh38.p13.annot.tsv.gz"
)

# Corresponding filenames for saving
FILENAMES=(
    "GSE231858_norm_counts_TPM_GRCh38.p13_NCBI.tsv.gz"
    "Human.GRCh38.p13.annot.tsv.gz"
)

# Loop through URLs and download the files
for i in "${!URLS[@]}"; do
    FILE_URL="${URLS[$i]}"
    FILE_NAME="${FILENAMES[$i]}"
    echo "Downloading $FILE_NAME..."
    wget -O "data/$FILE_NAME" "$FILE_URL"
    if [ $? -eq 0 ]; then
        echo "$FILE_NAME downloaded successfully!"
    else
        echo "Failed to download $FILE_NAME"
        exit 1
    fi
done

echo "All files downloaded to the 'data' folder."
