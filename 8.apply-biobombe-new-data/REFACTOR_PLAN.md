# Module 8 refactor plan: separating generic pipeline from Largaespada specifics

Status: **proposal only — not yet implemented.** This document exists so that if/when a second dataset gets run through this module, there's already a considered plan rather than a rushed one done under the pressure of "I just need this new dataset working." Nothing here should be acted on without a separate explicit decision to do so.

## Why this is worth planning now

The module's own README already states the intent: "The pipeline itself (0 through 5) is generic... The Largaespada lab MPNST screen is the one example dataset currently configured." In practice that's only half true. The *shared library* (`utils/data_prep.py`, `utils/pinwheels.py`) really is generic — confirmed by grep, there is not one Largaespada- or NF1-specific reference anywhere in `utils/`. But every numbered script (`0` through `5`) hardcodes `data/largaespada/` paths and `NF1_*` filenames directly in its body, in ~25 separate places across 6 files. Running a second dataset through this module today means hand-editing every one of those lines, in every script, and hoping none got missed — exactly the kind of error-prone, easy-to-get-subtly-wrong process a config-driven design exists to prevent.

Separately: `3.sanity_check_distribution.py` is **not** part of this genericization effort at all, by explicit decision. It's a one-off sanity check written for onboarding this specific dataset — not a generic, reusable notebook meant to run unchanged against a different dataset later. Its MPNST-specific biological-plausibility analysis (DepMap's own MPNST lines, the permutation test, the name-matched identity check) only makes sense because *this* dataset is MPNST/NF1-derived; a future dataset would warrant its own, differently-shaped one-off sanity check, written fresh rather than by generalizing this one. Scripts `0` through `2`, `4`, and `5` are the actual reusable pipeline this plan is about; `3` stays exactly as domain-specific and non-reusable as `0` already is, just without `0`'s "every new dataset needs one of these" expectation — a future dataset might get an analogous sanity notebook, or might not, as a separate judgment call each time.

## Audit: current state, file by file

| File | Generic? | Largaespada/NF1-specific parts |
|---|---|---|
| `utils/data_prep.py` | **Yes, fully.** | None. |
| `utils/pinwheels.py` | **Yes, fully** (no Largaespada-specific content) — but see the note below: as of this session it's being split into `utils/pinwheels.py` (plotting only) and a new `utils/latent_annotation.py` (the statistical/assignment core), since the name had drifted far from what the file actually does. | None. |
| `utils/model_utils.py` (project-wide, not module 8's own) | Yes. | None — the NMF clipping fix added this session is dataset-agnostic and a no-op for in-range data. |
| `0.preprocess_raw_data.py` | **No, and by design** — every dataset needs its own version of this step; the README already says so. | The raw-format-specific loading (`gene_effects.csv`, `largaespada_CRISPR_samples.csv`, the `cell_line_name`→`ModelID` rename) is inherently one-off. But the mygene.info alias-resolution logic (`fetch_aliases`, the backward-resolution loop, the all-NaN-column check) is *not* dataset-specific — it only needs `trained_gene_order` and a raw dataframe, and would be identical code in any future `0.preprocess_raw_data.py`. Currently it's copy-paste-and-modify, not shared. |
| `1.load_and_assess_gene_overlap.py` | Logic: yes. | Two hardcoded paths (`NF1_data_path`, `results_dir`) and one hardcoded output column name (`"NF1_data_column"`). |
| `2.format_and_scale_data.py` | Logic: yes. | `NF1_data_path`, `data_results_dir`, and two hardcoded output filenames. |
| `3.sanity_check_distribution.py` | **No, and by design — out of scope for this plan.** A one-off sanity check written for this dataset's onboarding, not a reusable notebook; see above. | Out of scope: the whole file, not just the MPNST-specific analysis within it. |
| `4.apply_biobombe_ensemble.py` | Logic: yes. | `data_results_dir`, one hardcoded input filename, one hardcoded metadata path, one hardcoded output filename. |
| `5.annotate_latent_scores.py` | Logic: yes. | `data_results_dir`, one hardcoded input filename, two hardcoded output filenames. |

Net: the shared library layer needs no changes. Scripts `1`, `2`, `4`, and `5` need their hardcoded dataset identity factored out; `0` needs its reusable alias-resolution logic extracted into the shared library; `3` needs nothing done to it under this plan.

## Proposed architecture

### 1. One per-dataset config, instead of ~25 scattered hardcoded strings

Introduce a tiny config step at the top of each numbered script — a single line that reads something like:

```python
DATASET = "largaespada"
dataset_dir = pathlib.Path(f"data/{DATASET}")
data_results_dir = dataset_dir / "results"
```

and a shared helper in `utils/data_prep.py` (or a new small `utils/dataset_config.py`) that derives every path/filename from `DATASET`, e.g.:

```python
def dataset_paths(dataset_name, module_dir=pathlib.Path(".")):
    base = module_dir / "data" / dataset_name
    return {
        "raw_dir": base / "raw_data",
        "results_dir": base / "results",
        "new_data_path": base / "new_data.parquet",
        "metadata_path": base / "new_metadata.parquet",
    }
```

Adding a second dataset then means setting `DATASET = "<new_name>"` in each script (or, better, one shared constant imported everywhere) rather than finding and editing every hardcoded path by hand.

### 2. Drop the `NF1_` filename prefix in favor of a dataset-derived one

`NF1_data.parquet`, `NF1_metadata.parquet`, `NF1_data_latent_scores.parquet`, etc. bake one specific study's nomenclature into what's supposed to be generic infrastructure. Rename to a dataset-agnostic pattern, e.g. `new_data.parquet` / `new_metadata.parquet` / `new_data_latent_scores.parquet` — or derive the prefix from `DATASET` itself if per-dataset output filenames are preferred for clarity when multiple datasets' results sit side by side later. Either way, the name shouldn't be the one place a new dataset's own identity leaks into shared code.

### 3. Move the generic half of `0.preprocess_raw_data.py`'s harmonization into `utils/data_prep.py`

Extract `fetch_aliases()` and the backward-resolution loop (missing-symbol detection, alias lookup, rename-map construction, the all-NaN-column check added this session) into a function like:

```python
def harmonize_gene_symbols(raw_df, trained_gene_order, sample_id_column="ModelID"):
    """Returns (harmonized_df, missing_genes_df) — see 0.preprocess_raw_data.py's
    original docstring for the backward-resolution rationale."""
```

A new dataset's own `0.preprocess_raw_data.py` then only needs to handle its own raw format (whatever shape *this* delivery arrives in) and call the shared function for the symbol harmonization — the part that's identical regardless of dataset. This doesn't change `0`'s status as "the one genuinely dataset-specific script" (the README's framing stays correct), it just stops that script from re-implementing shared logic from scratch each time.

### 4. Leave `3.sanity_check_distribution.py` alone entirely

Not a step — a deliberate non-step. It's a one-off analysis notebook, not infrastructure; nothing in this plan touches it. A future dataset's onboarding would produce its own differently-shaped sanity notebook if one is wanted, written with that dataset's own relevant comparisons in mind (module 3's structure — cheap-and-definitive checks before expensive-and-suggestive ones, ending in a stated bottom line — is a reasonable template to reuse by eye, but not a script to import or generalize).

### 5. Leave everything else alone

`utils/data_prep.py`, `utils/pinwheels.py`, the overall `data/<dataset>/raw_data/` + `data/<dataset>/results/` directory convention, the persisted scaler's role, and the core per-script logic (gene overlap assessment, scaling, model application, pathway/drug annotation) are already correctly designed to be dataset-agnostic and need no structural change — only the hardcoded strings layered on top of them do.

## Suggested order of work, if/when this gets picked up

1. `utils/data_prep.py`: add the `dataset_paths()`-style helper and `harmonize_gene_symbols()` extraction. Low risk — purely additive, nothing existing calls these yet.
2. Scripts `1`, `2`, `4`, `5`: swap hardcoded paths/filenames for the new helper + `DATASET` constant. Mechanical, low risk, easy to verify (output should be byte-identical to today's run for the Largaespada data, since only variable sourcing changes, not logic).
3. `0.preprocess_raw_data.py`: switch to calling `harmonize_gene_symbols()`; verify the printed resolution counts (151/163 resolved historically, 0/17 for the current `gene_effects.csv`) are unchanged.
4. Update the README's per-script sections and the "Data" section to describe the config mechanism. No change needed to the README's existing framing of `3.sanity_check_distribution` as a one-off.
5. Update the **top-level** repository README too: its free-text "Goal" and "Data" sections (2026-11 rewrite) deliberately dropped the mention of applying BioBombe to new, external datasets, since this module wasn't generic yet and the only configured example (Largaespada/MPNST) wasn't worth describing in prose as "the" new-data story. Once this plan lands and a second dataset has actually been run through the module, add a paragraph back describing the generic new-data-application capability (not the dataset-agnostic refactor task). The top-level README's Repository structure table already names this module accurately and needs no change.
5. Only after all of the above: rename `NF1_*` outputs to the dataset-agnostic convention, since this is the one step that changes on-disk filenames anything downstream (R figures, saved pinwheel plots) might already reference. `3.sanity_check_distribution.py` reads several of these filenames too (`NF1_data_model_ready_scaled.parquet`, `NF1_data_depmap_aligned.parquet`); since it's staying as a one-off script rather than being refactored alongside the rest, it would need its own hardcoded filenames updated by hand at this step, same as any other one-time edit.

Each step is independently revertible and independently testable against the existing Largaespada run, which is the main reason to sequence it this way rather than as one large rewrite.

## Observation (2026-11-16): broader repo-wide testing/naming debt, out of scope here

While stress-testing and fixing `utils/pinwheels.py`'s assignment/scoring
logic (shared, top-level - not module 8's own), it became clear the same
pattern - a module whose name undersells what it actually does, with no
tests - likely exists elsewhere in the repo too (e.g. `4.drug-dependency`
and `6.collab-data` have their own `utils/` directories not audited
here). The user explicitly chose to keep this session's effort scoped to
the scoring/assignment fix rather than going hunting for more of this
now. Worth a deliberate look later, as its own effort - not something to
back into incidentally while doing something else.

## What this plan deliberately does not cover

- Whether to generalize `0.preprocess_raw_data.py` itself into a single parameterized script with per-dataset config (rather than a new file per dataset) — that's a bigger design question (how much do raw deliveries actually vary in practice?) best answered once a second real dataset exists to generalize *from*, not speculatively now.
- Any change to the actual scientific methodology (scaling, NMF clipping, the permutation test design, etc.) — this is purely about where dataset identity lives in the code, not what the code computes.
- A timeline or priority call on *when* to do this. That's a judgment call for whoever is about to onboard the next dataset, weighed against whatever else is in flight at the time.
