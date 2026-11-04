"""All the bounds over a population of fields and a range of taxel resolutions (examples 08 and 09).

`resolution_study` evaluates, for every field, every k x k taxel grid and every twist q:
    truth            h_{X(nu)}(q), the support of the true wrench set
    M0_lo, M0_up     coarse bounds (closed form, patch-vertex LP), independent of k
    MT_lo, MT_up     taxel forces and CoP (SOCP, corner LP)
    MF_lo, MF_up     taxel forces only (closed forms)
    plugin           support of the plug-in estimate `plugin_uniform` (not a bound)
The twists Q are physical twists given about a reference point `ref`: "cop" (the true CoP of each field, e.g. q =
(0, 0, 1) is the unit rotation about the CoP), a fixed point (2,), or None (the origin). They are moved to the origin
with `shift_twist` before evaluation. The results can be cached in an .npz file, which is reused only if the study
parameters match.
"""

import time
from pathlib import Path

import numpy as np

from .bounds import measure_M0
from .estimate import plugin_uniform
from .geometry import TaxelGrid, shift_twist
from .measure import support
from .reduced import reduced_M0, reduced_MF, reduced_MT

PER_K = ("MT_lo", "MT_up", "MF_lo", "MF_up", "plugin")


def _params(fields, ks, Q, mu, ref):
    """Study parameters as arrays, stored with the cache and compared by _load to decide reuse."""
    patch = fields[0].measure.patch
    return dict(ks=np.asarray(ks), Q=Q, mu=np.asarray(float(mu)),
                patch=np.array([patch.Lx, patch.Ly, patch.N]),
                labels=np.array([f.label for f in fields]),
                ref=np.array(repr(ref if isinstance(ref, str) or ref is None else list(np.asarray(ref, float)))))


def _load(path, params):
    """Cached results if the file exists and was computed with the same parameters, else None."""
    if not Path(path).exists():
        return None
    with np.load(path) as d:
        for key, value in params.items():
            if f"param_{key}" not in d.files:
                return None
            stored = d[f"param_{key}"]
            if stored.shape != value.shape:
                return None
            same = np.allclose(stored, value) if value.dtype.kind == "f" else np.array_equal(stored, value)
            if not same:
                return None
        return {k: d[k] for k in d.files if not k.startswith("param_")}


def resolution_study(fields, ks, Q, mu=1.0, ref="cop", cache=None, verbose=True):
    """Bounds for every field in `fields` (a list of `pfb.fields.Field`), k x k taxels for k in ks, twists Q about ref.

    Returns a dict of arrays: truth, M0_lo, M0_up of shape (n_fields, n_q); the keys in PER_K of shape
    (n_fields, n_k, n_q); dmax (max taxel diameter, shape (n_k,)), fN (n_fields,), ks, labels and families.
    With cache=path, the results are loaded from path if it holds a study with the same parameters, and saved
    there otherwise.
    """
    Q = np.atleast_2d(np.asarray(Q, float))
    params = _params(fields, ks, Q, mu, ref)
    if cache is not None:
        found = _load(cache, params)
        if found is not None:
            if verbose:
                print(f"loaded cached study from {cache}")
            return found

    patch = fields[0].measure.patch
    nf, nk, nq = len(fields), len(ks), len(Q)
    out = {key: np.zeros((nf, nq)) for key in ("truth", "M0_lo", "M0_up")}
    out |= {key: np.zeros((nf, nk, nq)) for key in PER_K}
    grids = [TaxelGrid(patch, k, k) for k in ks]
    t0 = time.time()
    for i, field in enumerate(fields):
        nu = field.measure
        p = nu.cop if isinstance(ref, str) and ref == "cop" else (np.zeros(2) if ref is None else np.asarray(ref, float))
        Q0 = shift_twist(Q, p)
        out["truth"][i] = support(nu, Q0, mu)
        out["M0_lo"][i], out["M0_up"][i] = reduced_M0(patch, measure_M0(nu), Q0, mu)
        for j, t in enumerate(grids):
            F = t.downsample(nu)
            out["MT_lo"][i, j], out["MT_up"][i, j] = reduced_MT(t, (nu.fN, nu.m1, F), Q0, mu)
            out["MF_lo"][i, j], out["MF_up"][i, j] = reduced_MF(t, F, Q0, mu)
            out["plugin"][i, j] = support(plugin_uniform(t, F), Q0, mu)
        if verbose:
            print(f"[{i + 1}/{nf}] {field.label}  ({time.time() - t0:.0f} s)")

    out |= dict(dmax=np.array([t.diam.max() for t in grids]), fN=np.array([f.measure.fN for f in fields]),
                ks=np.asarray(ks), labels=params["labels"], families=np.array([f.family for f in fields]))
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **out, **{f"param_{k}": v for k, v in params.items()})
        if verbose:
            print(f"saved {cache}")
    return out
