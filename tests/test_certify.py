"""Checks of the certified load factor (pfb/certify.py) against direction grids, brute force and the truth."""

import itertools

import numpy as np
import pytest

import pfb
from pfb.certify import PadTerm, load_factor_lower, load_factor_true

MU = 0.8
TOL = 1e-5
PATCH = pfb.Patch(1.0, 1.0, N=32)
NU = pfb.ramp(PATCH, (0.6, 0.45), 0.4, 1.3, 2.4)
LOADS = [np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.15]), np.array([0.3, -0.8, 0.2])]
O = np.array([0.5, 0.5])  # reference point of the loads (pad centre)


def truth(nu, w, o=O):
    m = nu.w > 0
    return load_factor_true(nu.patch.points[m], nu.w[m], w, MU, o)


@pytest.mark.parametrize("w", LOADS)
def test_exact_certificate_below_direction_grid(w):
    """M_T with reference at the CoP: s*_lo is attained, h_lo(q*) / <q*, w> = s*_lo (Thm 6.1 at the binding q*),
    and it is below the minimum over a finite direction grid (which overestimates it: an unsafe certificate)."""
    tx = pfb.TaxelGrid(PATCH, 4, 4)
    fN, c, F = pfb.measure_MT(NU, tx)
    s, q = load_factor_lower(PadTerm.from_taxels(tx, F, cop=c), w, MU, o=c, return_q=True)
    assert np.isclose(pfb.socp_lower_MT(tx, F, c, q, MU) / (q @ w), s, rtol=1e-5)
    Q = pfb.directions(1500)
    Q = np.vstack([Q, -Q])
    Q = Q[Q @ w > 1e-9]
    grid = np.min(pfb.socp_lower_MT(tx, F, c, Q, MU) / (Q @ w))
    assert s <= grid + TOL


def test_matches_brute_force_over_concentrated_measures():
    """M_F, 2 taxels: s*_lo equals the minimum over point loads c_i on a fine grid of each closure."""
    patch = pfb.Patch(1.0, 0.5, N=8)
    tx = pfb.TaxelGrid(patch, 2, 1)
    F, w, o = np.array([0.6, 0.4]), np.array([0.4, 0.2, 0.3]), np.array([0.5, 0.25])
    s = load_factor_lower(PadTerm.from_taxels(tx, F), w, MU, o)
    cand = [np.stack(np.meshgrid(np.linspace(l[0], h[0], 6), np.linspace(l[1], h[1], 4)), -1).reshape(-1, 2)
            for l, h in zip(tx.lo, tx.hi)]
    brute = min(load_factor_true(np.array([c1, c2]), F, w, MU, o) for c1, c2 in itertools.product(*cand))
    assert s <= brute + TOL and brute - s < 0.02 * brute + TOL


@pytest.mark.parametrize("w", LOADS)
@pytest.mark.parametrize("k", [1, 4, 16])
def test_soundness_and_ordering(w, k):
    """plain <= M_0, M_F <= M_T <= truth, for loads about the pad centre."""
    tx = pfb.TaxelGrid(PATCH, k, k)
    fN, c, F = pfb.measure_MT(NU, tx)
    one = pfb.TaxelGrid(PATCH, 1, 1)
    s = dict(T=load_factor_lower(PadTerm.from_taxels(tx, F, cop=c), w, MU, O),
             F=load_factor_lower(PadTerm.from_taxels(tx, F), w, MU, O),
             M0=load_factor_lower(PadTerm.from_taxels(one, [fN], cop=c), w, MU, O),
             plain=load_factor_lower(PadTerm.from_taxels(one, [fN]), w, MU, O))
    true = truth(NU, w)
    assert s["T"] <= true + TOL
    assert s["F"] <= s["T"] + TOL and s["M0"] <= s["T"] + TOL
    assert s["plain"] <= min(s["F"], s["M0"]) + TOL


def test_resolution_tightens_certificate():
    w = LOADS[0]
    vals = []
    for k in (2, 4, 8, 16):
        tx = pfb.TaxelGrid(PATCH, k, k)
        fN, c, F = pfb.measure_MT(NU, tx)
        vals.append(load_factor_lower(PadTerm.from_taxels(tx, F, cop=c), w, MU, O))
    assert all(v0 <= v1 + TOL for v0, v1 in zip(vals, vals[1:]))  # nested refinement (Prop 8.2)
    assert vals[-1] > 0.85 * truth(NU, w)


def test_two_pads_sum_is_sound_and_larger():
    """Two pads facing each other (pad 2 mirrored): the Minkowski certificate is <= the true combined factor."""
    tx = pfb.TaxelGrid(PATCH, 4, 4)
    nu2 = pfb.hertz(PATCH, (0.4, 0.55), 0.3)
    R2 = np.diag([-1.0, 1.0])  # pad 2 coordinates mirrored in x, same plane origin
    t2 = np.array([1.0, 0.0])
    pad1 = PadTerm.from_taxels(tx, tx.downsample(NU), cop=NU.cop)
    pad2 = PadTerm.from_taxels(tx, tx.downsample(nu2), cop=nu2.cop, R=R2, t=t2)
    for w in LOADS:
        s1, s2 = load_factor_lower(pad1, w, MU, O), load_factor_lower(pad2, w, MU, O)
        s12 = load_factor_lower([pad1, pad2], w, MU, O)
        pts = np.vstack([NU.patch.points, PATCH.points @ R2.T + t2])
        true = load_factor_true(pts, np.r_[NU.w, nu2.w], w, MU, O)
        assert max(s1, s2) <= s12 + TOL and s12 <= true + TOL


@pytest.mark.parametrize("w", LOADS)
@pytest.mark.parametrize("cop", [True, False])
def test_certificate_is_attained(w, cop):
    """Tightness: the worst-case point loads F_i delta_{c_i} are compatible (c_i in the taxel closures, measured CoP
    under M_T) and their exact load factor equals s*_lo."""
    tx = pfb.TaxelGrid(PATCH, 4, 4)
    fN, c, F = pfb.measure_MT(NU, tx)
    s, worst = load_factor_lower(PadTerm.from_taxels(tx, F, cop=c if cop else None), w, MU, O, return_worst=True)
    pts = worst[0]
    assert np.all(pts >= tx.lo - 1e-6) and np.all(pts <= tx.hi + 1e-6)
    if cop:
        assert np.allclose(F @ pts / fN, c, atol=1e-6)
    assert np.isclose(load_factor_true(pts, F, w, MU, O), s, rtol=1e-4, atol=TOL)


def test_two_pad_certificate_is_attained():
    tx = pfb.TaxelGrid(PATCH, 4, 4)
    nu2 = pfb.hertz(PATCH, (0.4, 0.55), 0.3)
    R2, t2 = np.diag([-1.0, 1.0]), np.array([1.0, 0.0])
    F1, F2 = tx.downsample(NU), tx.downsample(nu2)
    pads = [PadTerm.from_taxels(tx, F1, cop=NU.cop), PadTerm.from_taxels(tx, F2, cop=nu2.cop, R=R2, t=t2)]
    for w in LOADS:
        s, worst = load_factor_lower(pads, w, MU, O, return_worst=True)
        true = load_factor_true(np.vstack(worst), np.r_[F1, F2], w, MU, O)
        assert np.isclose(true, s, rtol=1e-4, atol=TOL)


def test_coarse_measurement_analytic():
    """M_0 (one taxel with CoP, reference at the CoP): s* = mu f_N / ||f|| for a pure force, 0 with any torque."""
    one = pfb.TaxelGrid(PATCH, 1, 1)
    fN, c = NU.w.sum(), NU.cop
    pad = PadTerm.from_taxels(one, [fN], cop=c)
    f = np.array([0.3, -0.4])
    assert np.isclose(load_factor_lower(pad, np.r_[f, 0.0], MU, c), MU * fN / np.linalg.norm(f), rtol=1e-5)
    assert abs(load_factor_lower(pad, np.r_[f, 0.05], MU, c)) < 1e-6
    assert abs(load_factor_lower(pad, np.array([0.0, 0.0, 1.0]), MU, c)) < 1e-6
