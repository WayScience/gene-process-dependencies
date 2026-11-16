"""
Small, hand-rolled statistics helpers for module 8's own analyses - kept
here rather than in the shared top-level utils/ since they're specific to
this module's one-off Largaespada analyses, not generic infrastructure.
"""

import numpy as np


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg FDR-adjusted q-values for an array of p-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(q_sorted, 0, 1)
    return q
