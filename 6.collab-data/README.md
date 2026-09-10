# Downloading collaborator data

In this module, we download and test bulk RNA-seq data from pediatric tumors provided by a collaborator on the project (Dr. Adam Green) to extend and test real-world applications. The 'get_geo_data.sh' script will download this data from the NCBI database. 


## Files Downloaded
| **File**                                   | **Description**                                           |
|-------------------------------------------|-----------------------------------------------------------|
| `GSE231858_norm_counts_TPM_GRCh38.p13_NCBI.tsv.gz` | Normalized RNA-seq counts in TPM format.                 |
| `Human.GRCh38.p13.annot.tsv.gz`            | Human genome annotation file for GRCh38.p13 reference.    |

Source: [NCBI GEO Accession](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE231858)

---

## Setup and Usage

Execute the following command to update permissions and run the script:

```bash
chmod +x 6.collab-data/get_geo_data.sh && ./6.collab-data/get_geo_data.sh
