"""All the bounds over a population of fields and a range of taxel resolutions (examples 08 and 09).

`resolution_study` evaluates, for every field, every k x k taxel grid and every direction q:
    truth            h_{W(nu)}(q)                                   (Prop 3.2)
    M0_lo, M0_up     Thm 4.3 and Cor 4.6 (independent of k)
    MT_lo, MT_up     Thms 6.1 and 6.2
    MF_lo, MF_up     Thm 7.2, and the sweep over C_MF of Thm 7.3 (a value <= the exact upper bound)
    ref_lo, ref_up   certified fixed-reference bounds of Prop 7.4 (centre of C_MF)
    plugin           support of the plug-in estimate `plugin_uniform` (not a bound)
The results can be cached in an .npz file, which is reused only if the study parameters match.
"""

import time
from pathlib import Path

import numpy as np

from .bounds import measure_M0
from .estimate import plugin_uniform
from .geometry import TaxelGrid
from .measure import support
from .reduced import certified_ref_MF, reduced_M0, reduced_MF, reduced_MT

PER_K = ("MT_lo", "MT_up", "MF_lo", "MF_up", "ref_lo", "ref_up", "plugin")


def _params(fields, ks, Q, mu, mf_kwargs):
    """Study parameters as arrays, stored with the cache and compared by _load to decide reuse."""
    patch = fields[0].measure.patch
    return dict(ks=np.asarray(ks), Q=Q, mu=np.asarray(float(mu)),
                patch=np.array([patch.Lx, patch.Ly, patch.N]),
                labels=np.array([f.label for f in fields]),
                mf_kwargs=np.array(repr(sorted(mf_kwargs.items()))))


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


def resolution_study(fields, ks, Q, mu=1.0, mf_kwargs=None, cache=None, verbose=True):
    """Bounds for every field in `fields` (a list of `pfb.fields.Field`), k x k taxels for k in ks, directions Q.

    Returns a dict of arrays: truth, M0_lo, M0_up of shape (n_fields, n_q); the keys in PER_K of shape
    (n_fields, n_k, n_q); dmax (max taxel diameter, shape (n_k,)), fN (n_fields,), ks, labels and families.
    mf_kwargs is passed to `reduced_MF` (default n_c=5, polish=True). With cache=path, the results are
    loaded from path if it holds a study with the same parameters, and saved there otherwise.
    """
    Q = np.atleast_2d(np.asarray(Q, float))
    mf_kwargs = {"n_c": 5, "polish": True, **(mf_kwargs or {})}
    params = _params(fields, ks, Q, mu, mf_kwargs)
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
        out["truth"][i] = support(nu, Q, mu)
        out["M0_lo"][i], out["M0_up"][i] = reduced_M0(patch, measure_M0(nu), Q, mu)
        for j, t in enumerate(grids):
            F = t.downsample(nu)
            out["MT_lo"][i, j], out["MT_up"][i, j] = reduced_MT(t, (nu.fN, nu.cop, F), Q, mu)
            out["MF_lo"][i, j], out["MF_up"][i, j] = reduced_MF(t, F, Q, mu, **mf_kwargs)
            out["ref_lo"][i, j], out["ref_up"][i, j] = certified_ref_MF(t, F, Q, mu)
            out["plugin"][i, j] = support(plugin_uniform(t, F), Q, mu)
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
