#!/usr/bin/env python
# coding: utf-8

# ## Load new data and assess gene overlap with DepMap
# 
# New gene dependency data isn't guaranteed to cover the same genes, use the
# same gene identifiers, or list them in the same order as the DepMap data
# BioBombe was trained on.
# This notebook loads the new dataset and reports exactly how it lines up
# against DepMap and against the trained ensemble's gene set, before anything
# downstream trusts a join between them.
# 
# Run this script with `8.apply-biobombe-new-data/` as the working directory.

# In[ ]:


import pathlib
import sys

import pandas as pd

sys.path.insert(0, "utils")
import data_prep as dp


# In[ ]:


data_directory = pathlib.Path("../0.data-download/data").resolve()
model_save_dir = pathlib.Path("../3.run-biobombe/saved_models").resolve()

# Point this at the new dataset once it's available; see the module README.
NF1_data_path = pathlib.Path("data/largaespada/NF1_data.parquet")

# Everything derived from the new dataset itself, not just DepMap's public
# data, stays under data/ (gitignored) rather than the committed results/.
results_dir = pathlib.Path("data/largaespada/results")
results_dir.mkdir(parents=True, exist_ok=True)


# In[ ]:


depmap_df, gene_dict_df = dp.load_depmap_reference(data_directory)
trained_gene_order = dp.get_trained_gene_order(model_save_dir)

print(f"DepMap reference: {depmap_df.shape[0]} models, {depmap_df.shape[1] - 1} genes")
print(f"Trained ensemble gene set: {len(trained_gene_order)} genes")


# In[ ]:


new_df = dp.load_new_dependency_data(NF1_data_path)
print(f"New data: {new_df.shape[0]} samples, {new_df.shape[1] - 1} genes")
new_df.head()


# In[ ]:


overlap_report = dp.assess_gene_overlap(new_df, depmap_df, gene_dict_df, trained_gene_order)

print(f"Resolved {overlap_report['n_resolved']} / {overlap_report['n_new_genes']} new-data gene columns to an entrez id")
print(f"Shared with DepMap's full gene set: {overlap_report['n_shared_with_depmap']} / {len(overlap_report['depmap_entrez_ids'])}")
print(f"Shared with the trained ensemble's gene set: {overlap_report['n_shared_with_trained']} / {len(overlap_report['trained_entrez_ids'])}")
print(f"Trained genes with no match in the new data: {len(overlap_report['trained_genes_missing_in_new'])}")
print(f"New data already in DepMap gene order: {overlap_report['is_order_aligned']}")

if overlap_report["unresolved_columns"]:
    print(f"\n{len(overlap_report['unresolved_columns'])} new-data columns couldn't be resolved to an entrez id, "
          "first 10:")
    print(overlap_report["unresolved_columns"][:10])

if overlap_report["trained_genes_missing_in_new"]:
    coverage = 1 - len(overlap_report["trained_genes_missing_in_new"]) / len(trained_gene_order)
    print(f"\nTrained-gene coverage: {coverage:.1%}. "
          "Missing genes get imputed with the training mean in step 2, not dropped.")


# In[ ]:


# Save the full report for the later steps and for manual review.
# entrez_map and the two gene-name lists are the parts worth keeping on disk;
# the entrez id sets are large and easily recomputed, so they're left out.
report_to_save = pd.DataFrame({
    "NF1_data_column": list(overlap_report["entrez_map"].keys()),
    "resolved_entrez_id": list(overlap_report["entrez_map"].values()),
})
report_to_save.to_parquet(results_dir / "gene_overlap_report.parquet", index=False)

missing_genes_df = pd.DataFrame({"trained_gene_missing_in_new": overlap_report["trained_genes_missing_in_new"]})
missing_genes_df.to_parquet(results_dir / "trained_genes_missing_in_new.parquet", index=False)

print(f"Saved gene overlap report to {results_dir / 'gene_overlap_report.parquet'}")

