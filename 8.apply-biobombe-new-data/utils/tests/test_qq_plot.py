"""
Tests for utils/qq_plot.py. Plain assert-based, runnable directly:
    python3 utils/tests/test_qq_plot.py
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")  # no display needed for a test
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from qq_plot import qq_plot_pvalues


def test_qq_plot_pvalues_runs_without_error():
    fig, ax = plt.subplots()
    p_values_by_label = {"a": [0.1, 0.5, 0.9], "b": [0.01, 0.3, 0.7, 0.99]}
    qq_plot_pvalues(p_values_by_label, ax)
    # One scatter collection per label.
    assert len(ax.collections) == 2
    plt.close(fig)


def test_qq_plot_pvalues_uniform_lands_near_diagonal():
    # Evenly-spaced p-values are exactly the uniform null's expected order
    # statistics, so observed should equal expected point-for-point.
    fig, ax = plt.subplots()
    n = 1000
    uniform_p = (np.arange(1, n + 1) - 0.5) / n
    qq_plot_pvalues({"uniform": uniform_p}, ax)
    offsets = ax.collections[0].get_offsets()
    expected, observed = offsets[:, 0], offsets[:, 1]
    assert np.allclose(expected, observed, atol=1e-6)
    plt.close(fig)


def test_qq_plot_pvalues_inflated_lands_above_diagonal():
    # All-tiny p-values should sit well above the y=x diagonal - the
    # visual signature of an anti-conservative (inflated) test.
    fig, ax = plt.subplots()
    tiny_p = np.full(100, 1e-6)
    qq_plot_pvalues({"inflated": tiny_p}, ax)
    offsets = ax.collections[0].get_offsets()
    expected, observed = offsets[:, 0], offsets[:, 1]
    assert np.all(observed > expected)
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
