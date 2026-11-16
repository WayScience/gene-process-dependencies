#!/usr/bin/env python
# coding: utf-8

# ## Investigate per-sample hits and test for differences by mutation status
#
# Four exploratory passes over the full (every-pathway, every-drug)
# per-sample tables `5.annotate_latent_scores.py` persists
# (`NF1_data_all_pathways.parquet` / `NF1_data_all_drugs.parquet`):
#
# 1. Each sample's own top 10 CORUM pathways and top 10 drugs.
# 2. One consolidated pinwheel-style figure per feature family, all 8
#    samples overlaid on shared angular positions, colored by cell type and
#    marked by SUZ12 status - meant for exploration, not a publication
#    figure, so it's one combined PDF rather than one file per sample.
# 3. A stratified differential test: within each of the 3 cell-type strata,
#    do pathway/drug scores differ by SUZ12/NF1 mutation status? Sample
#    sizes are as small as n=1 per group, so this uses a moderated t-test
#    (empirical-Bayes variance shrinkage, `utils/stats.py`) rather than an
#    ordinary t-test, which is undefined for an n=1 group.
# 4. A calibration check (QQ plot) on that test's own p-values, since an
#    earlier run of this test produced implausibly large |t| values across
#    every comparison, not just the one with zero within-comparison
#    replication - see Section D and `utils/stats.py`'s fit_shared_prior().
#
# All of the actual logic (the shrinkage test, comparison derivation,
# pivoting, plotting) lives in tested functions in `utils/` - this script
# is orchestration only. See `utils/tests/test_stats.py` and
# `utils/tests/test_pinwheel_overlay.py`.
#
# Run this script with `8.apply-biobombe-new-data/` as the working directory.

# In[ ]:


import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, "utils")
from pinwheel_overlay import plot_overlay_pinwheel, top_feature_union
from qq_plot import qq_plot_pvalues
from stats import collect_replicate_groups, derive_comparisons, differential_test


# In[ ]:


data_results_dir = pathlib.Path("data/largaespada/results")
results_dir = data_results_dir / "06_investigate_hits"
results_dir.mkdir(parents=True, exist_ok=True)
visualize_dir = pathlib.Path("visualize/06_investigate_hits")
visualize_dir.mkdir(parents=True, exist_ok=True)

metadata_df = pd.read_parquet("data/largaespada/NF1_metadata.parquet")
all_pathways_df = pd.read_parquet(data_results_dir / "NF1_data_all_pathways.parquet")
all_drugs_df = pd.read_parquet(data_results_dir / "NF1_data_all_drugs.parquet")
print(f"Pathways: {all_pathways_df.shape} ({all_pathways_df['reactome_pathway'].nunique()} unique)")
print(f"Drugs: {all_drugs_df.shape} ({all_drugs_df['name'].nunique()} unique)")


# ### Section A - per-sample top 10 hits

# In[ ]:


top10_pathways_df = (
    all_pathways_df.sort_values(["ModelID", "pathway_score"], ascending=[True, False])
    .groupby("ModelID")
    .head(10)
)
top10_pathways_df.to_parquet(results_dir / "NF1_data_top10_pathways_per_sample.parquet", index=False)
print(top10_pathways_df[["ModelID", "reactome_pathway", "gene_set_library", "nes_score", "pathway_score"]].to_string(index=False))


# In[ ]:


top10_drugs_df = (
    all_drugs_df.sort_values(["ModelID", "pathway_score"], ascending=[True, False])
    .groupby("ModelID")
    .head(10)
)
top10_drugs_df.to_parquet(results_dir / "NF1_data_top10_drugs_per_sample.parquet", index=False)
print(top10_drugs_df[["ModelID", "name", "pearson_correlation", "pathway_score"]].to_string(index=False))


# ### Section B - consolidated pinwheel visualization
#
# One polar overlay per feature family: fixed angular positions shared by
# all 8 samples (the union of each sample's own top 10, capped at 30
# features total), one line per sample colored by cell type with a filled
# marker for SUZ12-positive samples and an open marker for SUZ12-negative.

# In[ ]:


pathway_features = top_feature_union(all_pathways_df, metadata_df, "reactome_pathway", "pathway_score")
drug_features = top_feature_union(all_drugs_df, metadata_df, "name", "pathway_score")
print(f"Pinwheel feature counts - pathways: {len(pathway_features)}, drugs: {len(drug_features)}")

pdf_path = visualize_dir / "NF1_data_pinwheel_overlay_all_samples.pdf"
with PdfPages(pdf_path) as pdf:
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    plot_overlay_pinwheel(all_pathways_df, metadata_df, "reactome_pathway", "pathway_score", pathway_features, "CORUM pathways - all 8 samples", ax)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    plot_overlay_pinwheel(all_drugs_df, metadata_df, "name", "pathway_score", drug_features, "Drugs - all 8 samples", ax)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print(f"Saved combined pinwheel figure to {pdf_path}")


# ### Section C - stratified differential analysis
#
# Within each cell-type stratum, does the pathway/drug score differ by
# SUZ12 or NF1 mutation status? 4 testable comparisons (tumor_derived's
# SUZ12 and NF1 splits are identical, so it's tested once, not twice;
# schwann-cell x NF1 has no variation and is skipped).
#
# Every feature in a given comparison shares the same group sizes, so the
# pooled within-group degrees of freedom is a single number per comparison
# - 0 for the schwann-cell comparison (n=1 vs n=1), meaning there is no
# empirical variance information in that comparison at all. The shared
# prior below is fit once per feature family from every other n>=2 sample
# group in the dataset (not per-comparison) specifically so that
# comparison still has a prior to shrink toward. d0 (the prior's assumed
# "pseudo-replicate" weight) is a small fixed constant, not method-of-moment
# fit - see `utils/stats.py`'s module docstring for why.

# In[ ]:


comparisons = derive_comparisons(metadata_df)
print(f"Testable comparisons ({len(comparisons)}):")
for comparison in comparisons:
    print(f"  {comparison['stratum']} x {comparison['marker']}: "
          f"pos={comparison['pos_ids']} vs neg={comparison['neg_ids']}")

replicate_groups = collect_replicate_groups(comparisons)
print(f"Replicate (n>=2) groups used to fit the shared prior: {replicate_groups}")


# In[ ]:


differential_pathways_df = differential_test(all_pathways_df, "reactome_pathway", "pathway_score", comparisons, replicate_groups)
differential_pathways_df.to_parquet(results_dir / "NF1_data_differential_pathways_by_mutation_status.parquet", index=False)
print(f"Differential pathway results: {differential_pathways_df.shape}")

for (stratum, marker), group in differential_pathways_df.groupby(["stratum", "marker"], sort=False):
    print(f"\nTop pathways, {stratum} x {marker} (d0=4):")
    print(group.sort_values("q_d0_4").head(10)[["feature", "mean_pos", "mean_neg", "mean_diff", "t_d0_4", "p_d0_4", "q_d0_4"]].to_string(index=False))


# In[ ]:


differential_drugs_df = differential_test(all_drugs_df, "name", "pathway_score", comparisons, replicate_groups)
differential_drugs_df.to_parquet(results_dir / "NF1_data_differential_drugs_by_mutation_status.parquet", index=False)
print(f"Differential drug results: {differential_drugs_df.shape}")

for (stratum, marker), group in differential_drugs_df.groupby(["stratum", "marker"], sort=False):
    print(f"\nTop drugs, {stratum} x {marker} (d0=4):")
    print(group.sort_values("q_d0_4").head(10)[["feature", "mean_pos", "mean_neg", "mean_diff", "t_d0_4", "p_d0_4", "q_d0_4"]].to_string(index=False))


# ### Section D - p-value calibration check (QQ plot)
#
# A uniform QQ plot of each comparison's own p-values, one page per
# feature family. With ~150 pathways or ~3,900 drugs tested per
# comparison, most features aren't truly differential, so most p-values
# should land close to the y=x diagonal; points sitting well above it mean
# the test is reporting far more "significant" hits than a uniform null
# would - i.e. it's anti-conservative. This is the direct diagnostic for
# the inflated |t| values noted when this test was first run: that
# inflation traced to a real bias in how the shared prior's variance was
# estimated (a low-df variance estimate's MEDIAN sits well below its true
# value - see `fit_shared_prior()`'s docstring in `utils/stats.py`), which
# is now corrected there. This plot checks whether that correction (or any
# inflation beyond it) shows up here, in the actual p-values.

# In[ ]:


pathway_p_by_comparison = {
    f"{stratum} x {marker}": group["p_d0_4"]
    for (stratum, marker), group in differential_pathways_df.groupby(["stratum", "marker"], sort=False)
}
drug_p_by_comparison = {
    f"{stratum} x {marker}": group["p_d0_4"]
    for (stratum, marker), group in differential_drugs_df.groupby(["stratum", "marker"], sort=False)
}

qq_pdf_path = visualize_dir / "NF1_data_qq_plot_differential_pvalues.pdf"
with PdfPages(qq_pdf_path) as pdf:
    fig, ax = plt.subplots(figsize=(6, 6))
    qq_plot_pvalues(pathway_p_by_comparison, ax)
    ax.set_title("CORUM pathways - p-value calibration (d0=4)")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    qq_plot_pvalues(drug_p_by_comparison, ax)
    ax.set_title("Drugs - p-value calibration (d0=4)")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print(f"Saved p-value calibration QQ plots to {qq_pdf_path}")


# ### Caveats
#
# - `pathway_score` for a fixed feature varies across samples only through
#   `abs(latent_score)` (the feature's own weight is fixed ensemble-wide) -
#   that's right-skewed and bounded at zero, not symmetric, so treat the
#   mean differences above as directional signals rather than as if drawn
#   from a textbook Gaussian null. A second-order concern given the
#   already-very-low power here, not a blocker.
# - Permutation/rank tests (SAM, Wilcoxon) were considered and rejected:
#   with n=1 vs n=1 or n=1 vs n=2, there are only 2-3 possible label
#   permutations - nowhere near enough for a meaningful null distribution.
# - `d0` above is a documented, fixed choice (reported at 2 and 4), not a
#   fitted one - see `utils/stats.py`'s module docstring for why a full
#   method-of-moments eBayes fit wasn't used here. The shared prior's
#   *variance* (s0_sq), unlike d0, is corrected for a real low-df bias
#   (Section D above) - that correction is closed-form, not fitted, so it
#   doesn't carry the same risk as hand-rolling d0's trigamma-based fit.
# - Watch out for the schwann-cell comparison specifically (true n=1 vs
#   n=1, df_i=0): even after the variance-bias correction, it still
#   borrows 100% of its variance denominator from the shared prior rather
#   than any variance of its own, so it remains the least trustworthy of
#   the four comparisons - zero within-comparison replication to check the
#   borrowed prior against - regardless of where its |t| values rank.
# - The QQ plot above is itself an approximation: it assumes most features
#   are non-differential, which is reasonable for thousands of drugs but
#   weaker for the ~150 CORUM pathways curated for this ensemble (a higher
#   fraction may be genuinely differential), so expect that panel to sit
#   further above the diagonal even in a well-calibrated test.

print("\nDone.")
