"""
Tests for utils/latent_annotation.py. Plain assert-based (no pytest in
this project's conda env), runnable directly:
    python3 utils/tests/test_latent_annotation.py
"""

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from latent_annotation import (
    assign_unique_latent_dims,
    benjamini_hochberg,
    compute_and_plot_latent_scores,
    filter_significant_candidates,
    flag_shared_latent_slots,
)


def test_benjamini_hochberg_hand_computed():
    p = np.array([0.01, 0.02, 0.03, 0.5])
    q = benjamini_hochberg(p)
    expected = np.array([0.04, 0.04, 0.04, 0.5])
    assert np.allclose(q, expected), f"expected {expected}, got {q}"


def test_filter_significant_candidates_bh_family_and_effect_gate():
    # Family for correction is the 5 non-shuffled rows (extra_mask excludes
    # the 6th, shuffled row - which has the smallest p-value of all, 0.001 -
    # confirming it's excluded from the family entirely, not just ranked).
    df = pd.DataFrame({
        "p_value": [0.001, 0.01, 0.02, 0.5, 0.9, 0.001],
        "effect": [0.5, 0.2, 0.05, 0.3, 0.1, 10.0],
        "shuffled": [False, False, False, False, False, True],
    })
    result = filter_significant_candidates(
        df, extra_mask=~df["shuffled"], effect_col="effect", min_abs_effect=0.1,
    )
    # Hand-computed: BH over p=[0.001,0.01,0.02,0.5,0.9] gives q<=0.05 for the
    # first three (p=0.001,0.01,0.02); the effect-size gate (|effect|>=0.1)
    # then drops the third (effect=0.05), leaving two.
    assert set(result["p_value"]) == {0.001, 0.01}, f"got {result['p_value'].tolist()}"
    assert 10.0 not in result["effect"].values, "the shuffled row's tiny p-value must not pass despite excluding it"


def test_assign_unique_latent_dims_robustness_ranking_changes_outcome():
    # Pathway A has one huge single hit (100) plus one weak one (1) - its
    # single-best score beats B's, but its candidates don't corroborate each
    # other (median of top-3 collapses to 50.5). Pathway B has three
    # consistently strong hits (90, 85, 80) - a lower single best than A, but
    # a higher robustness-adjusted score (85). A's best slot (z=1) is also
    # one of B's candidates, forcing real competition for it.
    df = pd.DataFrame([
        {"pathway": "A", "score": 100.0, "z": 1, "model": "m", "full_model_z": 10, "init": 0},
        {"pathway": "A", "score": 1.0, "z": 2, "model": "m", "full_model_z": 10, "init": 0},
        {"pathway": "B", "score": 90.0, "z": 1, "model": "m", "full_model_z": 10, "init": 0},
        {"pathway": "B", "score": 85.0, "z": 3, "model": "m", "full_model_z": 10, "init": 0},
        {"pathway": "B", "score": 80.0, "z": 4, "model": "m", "full_model_z": 10, "init": 0},
    ])
    result = assign_unique_latent_dims(df, score_col="score", target_col="pathway").set_index("pathway")

    # B is ranked first under the new robustness score (85 > 50.5) and wins
    # the contested z=1 slot with its own best (90) - under the old
    # max-only ranking, A (max=100) would have gone first and claimed z=1,
    # pushing B to z=3 (85) instead. This is the behavior change.
    assert result.loc["B", "z"] == 1, "B should win the contested slot under robustness ranking"
    assert result.loc["B", "score"] == 90.0
    assert result.loc["B", "n_candidates"] == 3
    assert np.isclose(result.loc["B", "assignment_score"], 85.0)

    assert result.loc["A", "z"] == 2, "A should be pushed to its remaining slot"
    assert result.loc["A", "score"] == 1.0
    assert result.loc["A", "n_candidates"] == 2
    assert np.isclose(result.loc["A", "assignment_score"], 50.5)


def test_flag_shared_latent_slots_detects_collision():
    corum_df = pd.DataFrame([
        {"reactome_pathway": "PathwayX", "model": "m", "full_model_z": 10, "init": 0, "z": 1},
        {"reactome_pathway": "PathwayY", "model": "m", "full_model_z": 10, "init": 0, "z": 2},
    ])
    drug_df = pd.DataFrame([
        {"name": "DrugZ", "model": "m", "full_model_z": 10, "init": 0, "z": 1},  # collides with PathwayX
        {"name": "DrugW", "model": "m", "full_model_z": 20, "init": 0, "z": 5},  # no collision
    ])

    corum_out, drug_out = flag_shared_latent_slots([
        ("CORUM", corum_df, "reactome_pathway"),
        ("Drug", drug_df, "name"),
    ])

    corum_out = corum_out.set_index("reactome_pathway")
    drug_out = drug_out.set_index("name")
    assert corum_out.loc["PathwayX", "shared_slot_with"] == ["Drug: DrugZ"]
    assert corum_out.loc["PathwayY", "shared_slot_with"] == []
    assert drug_out.loc["DrugZ", "shared_slot_with"] == ["CORUM: PathwayX"]
    assert drug_out.loc["DrugW", "shared_slot_with"] == []


def test_compute_and_plot_latent_scores_is_signed_not_abs():
    # latent_score=-2, fixed score=3: abs()'d (old) would give +6;
    # signed (new) must give -6.
    latent_df = pd.DataFrame([{
        "ModelID": "S1", "model": "pca", "latent_dim_total": 5, "init": 0, "seed": 0, "0": -2.0,
    }])
    comp_df = pd.DataFrame([{
        "reactome_pathway": "PathwayX", "full_model_z": 5, "init": 0, "model": "pca", "z": 0, "nes_score": 3.0,
    }])
    result = compute_and_plot_latent_scores(
        "S1", latent_df, comp_df, "reactome_pathway", "nes_score", "Pathway", make_plot=False,
    )
    assert np.isclose(result["pathway_score"].iloc[0], -6.0), f"expected -6.0, got {result['pathway_score'].iloc[0]}"


def test_compute_and_plot_latent_scores_extra_cols():
    latent_df = pd.DataFrame([{
        "ModelID": "S1", "model": "pca", "latent_dim_total": 5, "init": 0, "seed": 0, "0": -2.0,
    }])
    comp_df = pd.DataFrame([{
        "reactome_pathway": "PathwayX", "full_model_z": 5, "init": 0, "model": "pca", "z": 0,
        "nes_score": 3.0, "n_candidates": 7,
    }])
    with_extra = compute_and_plot_latent_scores(
        "S1", latent_df, comp_df, "reactome_pathway", "nes_score", "Pathway",
        make_plot=False, extra_cols=["n_candidates"],
    )
    assert with_extra["n_candidates"].iloc[0] == 7

    without_extra = compute_and_plot_latent_scores(
        "S1", latent_df, comp_df, "reactome_pathway", "nes_score", "Pathway", make_plot=False,
    )
    assert "n_candidates" not in without_extra.columns


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failures = []
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as e:
            failures.append(test.__name__)
            print(f"FAIL {test.__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    run_all()
