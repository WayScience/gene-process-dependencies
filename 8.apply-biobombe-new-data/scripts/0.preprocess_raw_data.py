#!/usr/bin/env python
# coding: utf-8

# ## Prepare the raw Largaespada lab MPNST data
#
# The delivered files, under `data/largaespada/raw_data/`, don't match what
# `1.load_and_assess_gene_overlap.py`, `2.format_and_scale_data.py`, and
# `4.apply_biobombe_ensemble.py` expect:
# - `Achilles_scaled_gene_effects_MPNST.csv` has genes as rows and samples as
#   columns (the transpose of DepMap's convention, which every downstream
#   function assumes: samples as rows, genes as columns).
# - `largaespada_CRISPR_samples.csv`'s sample identifier column is named
#   "sample", not "ModelID".
# - Some gene columns use gene symbols HGNC has since renamed (e.g. "AARS"
#   is now "AARS1", the old aminoacyl-tRNA synthetase names, the 2019 ATP
#   synthase subunit renames, the 2020-21 histone nomenclature overhaul).
#   BioBombe's own gene set (`3.run-biobombe/data/trained_gene_order.parquet`)
#   is left exactly as it was trained on, out-of-date symbols and all; this
#   only harmonizes the new data's symbols to match it.
#
#   Resolution works backward from the trained set, not forward from the
#   raw data: for each trained gene missing under its current symbol, ask
#   mygene.info for that gene's own former names/aliases, then check for an
#   exact match among the raw column names. Going the other way (resolving
#   each raw column's symbol independently) hits genuinely ambiguous
#   aliases shared by unrelated genes (e.g. "STRA13" names both CENPX and
#   the unrelated BHLHE40) with no principled way to disambiguate; asking
#   "what were CENPX's former names" instead has exactly one right answer.
#
# This writes `NF1_data.parquet` and `NF1_metadata.parquet` directly under
# `data/largaespada/`, the exact filenames those scripts already hardcode,
# so nothing downstream needs to change. It also writes
# `data/largaespada/results/genes_unresolved_after_alias_lookup.parquet`,
# recording which trained genes have no match in this data at all (as of
# the last time this ran), for documentation and future reference.
#
# Run this script with `8.apply-biobombe-new-data/` as the working
# directory, once, whenever the raw files are replaced.

# In[1]:


import pathlib

import pandas as pd
import requests

largaespada_dir = pathlib.Path("data/largaespada")
raw_dir = largaespada_dir / "raw_data"
results_dir = largaespada_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)


# In[2]:


gene_effects = pd.read_csv(raw_dir / "Achilles_scaled_gene_effects_MPNST.csv", index_col=0)
gene_effects = gene_effects.T.reset_index(names="ModelID")
print(f"Loaded raw data: {gene_effects.shape[0]} samples, {gene_effects.shape[1] - 1} genes")


# ## Harmonize gene symbols to BioBombe's trained convention
#
# Only the trained gene set matters here: nothing outside it is used by the
# ensemble regardless of what it's named, so resolution targets exactly the
# genes BioBombe was trained on, not DepMap's full gene set.

# In[3]:


trained_gene_dict = pd.read_parquet(pathlib.Path("../3.run-biobombe/data/trained_gene_order.parquet").resolve())
trained_symbols = set(trained_gene_dict["symbol_id"])

raw_gene_columns = set(gene_effects.columns) - {"ModelID"}
missing_trained_symbols = sorted(trained_symbols - raw_gene_columns)
print(f"{len(trained_symbols) - len(missing_trained_symbols)} / {len(trained_symbols)} trained genes "
      f"already present under their current symbol; looking up aliases for the other {len(missing_trained_symbols)}")


# In[4]:


def fetch_aliases(symbols, batch_size=500):
    """
    Fetch each gene's own known aliases/former symbols from mygene.info.

    Returns
    -------
    dict
        Maps each input (current) symbol to its list of aliases (possibly empty).
    """
    aliases_by_symbol = {}
    for start in range(0, len(symbols), batch_size):
        batch = symbols[start:start + batch_size]
        response = requests.post(
            "https://mygene.info/v3/query",
            data={
                "q": ",".join(batch),
                "scopes": "symbol",
                "species": "human",
                "fields": "alias",
            },
            timeout=60,
        )
        response.raise_for_status()
        for hit in response.json():
            symbol = hit["query"]
            alias = hit.get("alias", [])
            aliases_by_symbol.setdefault(symbol, []).extend(
                [alias] if isinstance(alias, str) else alias
            )
    return aliases_by_symbol


# In[5]:


aliases_by_symbol = fetch_aliases(missing_trained_symbols) if missing_trained_symbols else {}

rename_map = {}
still_missing = []
for symbol in missing_trained_symbols:
    match = next((a for a in aliases_by_symbol.get(symbol, []) if a in raw_gene_columns), None)
    if match is not None:
        rename_map[match] = symbol
    else:
        still_missing.append(symbol)

print(f"Resolved {len(rename_map)} / {len(missing_trained_symbols)} missing trained genes via a former alias")
if still_missing:
    print(f"Still missing after alias lookup: {still_missing}")


# In[6]:


gene_effects = gene_effects.rename(columns=rename_map)

gene_effects.to_parquet(largaespada_dir / "NF1_data.parquet", index=False)
print(f"Wrote NF1_data.parquet: {gene_effects.shape[0]} samples, {gene_effects.shape[1] - 1} genes")

sample_metadata = pd.read_csv(raw_dir / "largaespada_CRISPR_samples.csv")
sample_metadata = sample_metadata.rename(columns={"sample": "ModelID"})
sample_metadata.to_parquet(largaespada_dir / "NF1_metadata.parquet", index=False)
print(f"Wrote NF1_metadata.parquet: {sample_metadata.shape[0]} samples, "
      f"columns {list(sample_metadata.columns)}")

pd.DataFrame({"trained_gene_missing_from_raw_data": still_missing}).to_parquet(
    results_dir / "genes_unresolved_after_alias_lookup.parquet", index=False
)
print(f"Wrote genes_unresolved_after_alias_lookup.parquet: {len(still_missing)} trained genes "
      "have no match anywhere in the raw data, under any known name")
