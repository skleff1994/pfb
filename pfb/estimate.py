"""Point estimates of the normal force measure from taxel data (the baselines that the bounds are compared with)."""

import numpy as np

from .measure import Measure


def plugin_uniform(taxels, F):
    """Plug-in estimate: each taxel force F_i spread uniformly (w.r.t. area) over its taxel.

    The estimate reproduces the taxel forces exactly (taxels.downsample gives back F). Its support
    function `support(plugin_uniform(taxels, F), q, mu)` (same twist q as the bounds) is the plug-in capacity.
    """
    patch = taxels.patch
    area = np.bincount(taxels.index, weights=patch.quad, minlength=taxels.r)
    w = np.asarray(F, float)[taxels.index] * patch.quad / area[taxels.index]
    return Measure(patch, w)
