"""
Tests for utils/pinwheel_overlay.py. Plain assert-based, runnable directly:
    python3 utils/tests/test_pinwheel_overlay.py
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")  # no display needed for a test
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from pinwheel_overlay import plot_overlay_pinwheel, top_feature_union


def _synthetic_data():
    metadata_df = pd.DataFrame({
        "ModelID": ["A", "B", "C"],
        "SUZ12": ["pos", "neg", "pos"],
        "NF1": ["neg", "neg", "pos"],
        "type": ["tumor_derived", "tumor_derived", "immortalized_peripheral_nerve"],
    })
    scores_df = pd.DataFrame({
        "ModelID": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "feature": ["f1", "f2", "f3"] * 3,
        "score": [3.0, 1.0, 0.5, 2.5, 1.2, 0.4, 0.1, 2.0, 1.5],
    })
    return metadata_df, scores_df


def test_top_feature_union_returns_expected_features():
    metadata_df, scores_df = _synthetic_data()
    features = top_feature_union(scores_df, metadata_df, "feature", "score", top_n_per_sample=2, max_features=30)
    assert set(features) == {"f1", "f2", "f3"}, f"expected all 3 features with top_n=2, got {features}"


def test_top_feature_union_respects_max_features_cap():
    metadata_df, scores_df = _synthetic_data()
    features = top_feature_union(scores_df, metadata_df, "feature", "score", top_n_per_sample=3, max_features=2)
    assert len(features) == 2, f"expected cap at 2 features, got {len(features)}: {features}"
    # f1 has the highest max-across-samples score (3.0), f3 the lowest (1.5 max) - f3 should be dropped
    assert "f1" in features
    assert "f3" not in features


def test_plot_overlay_pinwheel_runs_without_error():
    metadata_df, scores_df = _synthetic_data()
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    plot_overlay_pinwheel(scores_df, metadata_df, "feature", "score", ["f1", "f2", "f3"], "test title", ax)
    lines = ax.get_lines()
    assert len(lines) == 3, f"expected one line per sample (3), got {len(lines)}"
    plt.close(fig)


def test_top_feature_union_selects_by_magnitude_not_raw_value():
    # A strongly negative score must be just as selectable as a strongly
    # positive one - score_col is signed, so sorting by raw value alone
    # would silently exclude large-magnitude negative hits.
    metadata_df, scores_df = _synthetic_data()
    scores_df = scores_df.copy()
    scores_df.loc[scores_df["feature"] == "f3", "score"] = -10.0  # now the most extreme, despite being negative
    features = top_feature_union(scores_df, metadata_df, "feature", "score", top_n_per_sample=1, max_features=30)
    assert "f3" in features, f"expected the large-magnitude negative feature to be selected, got {features}"


def test_plot_overlay_pinwheel_plots_magnitude_of_signed_scores():
    metadata_df, scores_df = _synthetic_data()
    scores_df = scores_df.copy()
    scores_df.loc[(scores_df["ModelID"] == "A") & (scores_df["feature"] == "f1"), "score"] = -3.0
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    plot_overlay_pinwheel(scores_df, metadata_df, "feature", "score", ["f1", "f2", "f3"], "test title", ax)
    radii = ax.get_lines()[0].get_ydata()
    assert radii[0] == 3.0, f"expected the negative score to plot as a positive radius, got {radii[0]}"
    plt.close(fig)


def test_plot_overlay_pinwheel_handles_missing_feature():
    # A sample missing one of the shared features should fill with 0, not crash.
    metadata_df, scores_df = _synthetic_data()
    scores_df = scores_df[~((scores_df["ModelID"] == "B") & (scores_df["feature"] == "f3"))]
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    plot_overlay_pinwheel(scores_df, metadata_df, "feature", "score", ["f1", "f2", "f3"], "test title", ax)
    assert len(ax.get_lines()) == 3
    plt.close(fig)


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
