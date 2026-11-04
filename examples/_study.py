"""Shared setup of the population study used by examples 08 (resolution) and 09 (plug-in).

Both scripts call `get_study()`, so they always use the same fields, resolutions and directions.
The first call computes the study (about 6-8 min) and caches it in examples/data/; later calls
reload it. Command-line flags of both scripts:
    --quick      1 field per family and k <= 16 (about 1 min, separate cache file)
    --recompute  ignore the cache
"""

import argparse
from pathlib import Path

import numpy as np

import pfb

DATA = Path(__file__).parent / "data"
MU = 1.0
LEVER = 0.2  # lever arm (in pad sides) that balances force and torque in the mixed direction
DIRECTIONS = {
    "support in direction q = (0, 0, 1)": np.array([0.0, 0.0, 1.0]),
    f"mixed  q ∝ (1, 0, 1/{LEVER})": np.array([1.0, 0.0, 1.0 / LEVER]) / np.hypot(1.0, 1.0 / LEVER),
}


def get_study(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--quick", action="store_true", help="1 field per family, k <= 16")
    parser.add_argument("--recompute", action="store_true", help="ignore the cached study")
    args = parser.parse_args()

    patch = pfb.Patch(1.0, 1.0, N=64)  # unit pad; all lengths in pad sides
    fields = pfb.population(patch, n_per_family=1 if args.quick else 3, seed=0)
    ks = [1, 2, 4, 8, 16] if args.quick else [1, 2, 4, 8, 16, 32]
    cache = DATA / ("study_quick.npz" if args.quick else "study.npz")
    if args.recompute and cache.exists():
        cache.unlink()
    Q = np.array(list(DIRECTIONS.values()))
    S = pfb.resolution_study(fields, ks, Q, MU, cache=cache)
    return S, list(DIRECTIONS)


def ratio(S, key, iq):
    """S[key] / truth for direction iq, shape (n_fields, n_k) (or (n_fields,) for the M0 keys)."""
    t = S["truth"][:, iq]
    return S[key][..., iq] / (t[:, None] if S[key].ndim == 3 else t)
