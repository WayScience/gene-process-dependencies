"""
Tests for utils/pinwheels.py. Plain assert-based (no pytest in this
project's conda env), runnable directly:
    python3 utils/tests/test_pinwheels.py
"""

import pathlib
import shutil
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")  # no display needed for a test
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from pinwheels import plot_pinwheel


def test_plot_pinwheel_runs_and_saves_a_file():
    # plot_pinwheel saves to ./visualize/pinwheels relative to the cwd -
    # run from a scratch temp directory so this test never touches the
    # real repo tree, and to catch a regression of the missing mkdir this
    # function needed (make_plot=True is the default, so a caller who
    # doesn't pass make_plot=False would hit this).
    scores_df = pd.DataFrame({
        "pathway": ["a", "b", "c"],
        "pathway_score": [1.0, -2.0, 0.5],  # signed, as compute_and_plot_latent_scores now produces
    })
    tmp_dir = tempfile.mkdtemp()
    original_cwd = pathlib.Path.cwd()
    try:
        import os
        os.chdir(tmp_dir)
        plot_pinwheel("sample1", scores_df, "pathway", "Pathway")
        expected_path = pathlib.Path(tmp_dir, "visualize", "pinwheels", "pinwheel_Pathway_sample1.png")
        assert expected_path.exists(), f"expected a saved plot at {expected_path}"
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(tmp_dir, ignore_errors=True)


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
