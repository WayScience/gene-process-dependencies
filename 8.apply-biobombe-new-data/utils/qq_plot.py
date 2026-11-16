"""
A uniform p-value QQ plot: the standard diagnostic for whether a batch of
hypothesis tests is calibrated. Kept separate from utils/stats.py (pure
computation, no plotting) and utils/pinwheel_overlay.py (a different,
unrelated plot), so each module stays about one thing.

Under the null (most features here are not truly differential - with
3,891 drugs and 153 pathways tested per comparison, the great majority
shouldn't be), p-values should be approximately uniform on [0, 1], so
sorted -log10(p) should fall near the y=x line against its own expected
quantiles. Points systematically above the line mean the test is
anti-conservative (p-values too small, more "significant" hits than a
uniform null would produce) - exactly the symptom this plot is meant to
surface for utils/stats.py's moderated_t_test()/differential_test().
"""

import numpy as np


def qq_plot_pvalues(p_values_by_label, ax):
    """
    Draws one uniform QQ plot onto ax: for each (label, p_values) pair in
    p_values_by_label, sorts p_values ascending and plots observed
    -log10(p) against its expected value under a Uniform(0, 1) null
    (Order statistic i of n has expectation (i - 0.5) / n), plus a shared
    y=x reference line. Takes p-value arrays directly - no file I/O - so
    it's callable from a test with a tiny synthetic array.
    """
    max_val = 1.0
    for label, p_values in p_values_by_label.items():
        p_values = np.sort(np.asarray(p_values, dtype=float))
        n = len(p_values)
        expected = -np.log10((np.arange(1, n + 1) - 0.5) / n)
        observed = -np.log10(np.clip(p_values, np.finfo(float).tiny, 1.0))
        max_val = max(max_val, float(np.max(expected)), float(np.max(observed)))
        ax.scatter(expected, observed, s=8, alpha=0.5, label=f"{label} (n={n})")

    ax.plot([0, max_val], [0, max_val], color="black", linewidth=1, linestyle="--", label="expected (uniform null)")
    ax.set_xlabel("Expected -log10(p)")
    ax.set_ylabel("Observed -log10(p)")
    ax.legend(fontsize=7, loc="upper left")
