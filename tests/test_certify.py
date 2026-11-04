"""Checks of the certified and upper load factors (pfb/certify.py) against direction grids, brute force and the truth."""

import itertools

import numpy as np
import pytest

import pfb
from pfb.certify import PadTerm, classify, load_factor_lower, load_factor_true, load_factor_upper

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
    """M_T with reference at the CoP: s*_lo is attained, h_lo(q*) / <q*, w> = s*_lo (thm:combined_lower at the binding q*),
    and it is below the minimum over a finite direction grid (which overestimates it: an unsafe certificate)."""
    tx = pfb.TaxelGrid(PATCH, 4, 4)
    fN, m1, F = pfb.measure_MT(NU, tx)
    c = m1 / fN
    s, q = load_factor_lower(PadTerm.from_taxels(tx, F, cop=c), w, MU, o=c, return_q=True)
    assert np.isclose(pfb.socp_lower_MT(tx, F, m1, pfb.shift_twist(q, c), MU) / (q @ w), s, rtol=1e-5)
    Q = pfb.directions(1500)
    Q = np.vstack([Q, -Q])
    Q = Q[Q @ w > 1e-9]
    grid = np.min(pfb.socp_lower_MT(tx, F, m1, pfb.shift_twist(Q, c), MU) / (Q @ w))
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
    fN, m1, F = pfb.measure_MT(NU, tx)
    c = m1 / fN
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
        fN, m1, F = pfb.measure_MT(NU, tx)
        c = m1 / fN
        vals.append(load_factor_lower(PadTerm.from_taxels(tx, F, cop=c), w, MU, O))
    assert all(v0 <= v1 + TOL for v0, v1 in zip(vals, vals[1:]))  # nested refinement (prop:taxel_refinement_MF)
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
    fN, m1, F = pfb.measure_MT(NU, tx)
    c = m1 / fN
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


# --- upper load factor ------------------------------------------------------------------------

@pytest.mark.parametrize("w", LOADS)
@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_upper_brackets_truth_and_is_ordered(w, k):
    """s_lo <= s* <= s_up for M_F and M_T; the CoP tightens both; the verdict follows the bracket."""
    tx = pfb.TaxelGrid(PATCH, k, k)
    fN, m1, F = pfb.measure_MT(NU, tx)
    c = m1 / fN
    true = truth(NU, w)
    for cop in (None, c):
        pad = PadTerm.from_taxels(tx, F, cop=cop)
        lo, up = load_factor_lower(pad, w, MU, O), load_factor_upper(pad, w, MU, O)
        assert lo <= true + TOL and true <= up + TOL
        verdict, _, _ = classify(pad, 0.98 * true * w, MU, O)  # a load the true contact carries: never "slip"
        assert verdict in ("hold", "undecided")
    upF = load_factor_upper(PadTerm.from_taxels(tx, F), w, MU, O)
    upT = load_factor_upper(PadTerm.from_taxels(tx, F, cop=c), w, MU, O)
    assert upT <= upF + TOL


@pytest.mark.parametrize("w", LOADS)
@pytest.mark.parametrize("cop", [True, False])
def test_upper_primal_equals_dual_and_is_attained(w, cop):
    """prop:load_factor (iii): the primal and dual SOCPs agree, and the most favourable corner measure has load factor s_up."""
    tx = pfb.TaxelGrid(PATCH, 4, 4)
    fN, m1, F = pfb.measure_MT(NU, tx)
    pad = PadTerm.from_taxels(tx, F, cop=m1 / fN if cop else None)
    s, best = load_factor_upper(pad, w, MU, O, return_best=True)
    lam, corners = best[0]
    assert np.allclose(lam.sum(axis=1), F, atol=1e-6)
    if cop:
        assert np.allclose(np.einsum("ik,ikd->d", lam, corners), m1, atol=1e-6)
    keep = lam.ravel() > 1e-9
    s_best = load_factor_true(corners.reshape(-1, 2)[keep], lam.ravel()[keep], w, MU, O)
    assert np.isclose(s_best, s, rtol=1e-4, atol=TOL)


def test_upper_tightens_under_refinement():
    w = LOADS[0]
    vals = []
    for k in (1, 2, 4, 8, 16):
        tx = pfb.TaxelGrid(PATCH, k, k)
        vals.append(load_factor_upper(PadTerm.from_taxels(tx, tx.downsample(NU)), w, MU, O))
    assert all(v1 <= v0 + TOL for v0, v1 in zip(vals, vals[1:]))
    assert vals[-1] < 1.2 * truth(NU, w)


def test_two_pads_upper_is_sound():
    tx = pfb.TaxelGrid(PATCH, 4, 4)
    nu2 = pfb.hertz(PATCH, (0.4, 0.55), 0.3)
    R2, t2 = np.diag([-1.0, 1.0]), np.array([1.0, 0.0])
    pads = [PadTerm.from_taxels(tx, tx.downsample(NU), cop=NU.cop),
            PadTerm.from_taxels(tx, tx.downsample(nu2), cop=nu2.cop, R=R2, t=t2)]
    pts = np.vstack([NU.patch.points, PATCH.points @ R2.T + t2])
    for w in LOADS:
        true = load_factor_true(pts, np.r_[NU.w, nu2.w], w, MU, O)
        lo, up = load_factor_lower(pads, w, MU, O), load_factor_upper(pads, w, MU, O)
        assert lo <= true + TOL <= up + 2 * TOL
        s, best = load_factor_upper(pads, w, MU, O, return_best=True)
        P = np.vstack([c.reshape(-1, 2) for _, c in best])
        L = np.concatenate([l.ravel() for l, _ in best])
        assert np.isclose(load_factor_true(P[L > 1e-9], L[L > 1e-9], w, MU, O), s, rtol=1e-4, atol=TOL)


def test_certain_slip_is_detected():
    """A load well beyond the possible set is classified as slip, a small one as hold."""
    tx = pfb.TaxelGrid(PATCH, 8, 8)
    fN, m1, F = pfb.measure_MT(NU, tx)
    pad = PadTerm.from_taxels(tx, F, cop=m1 / fN)
    w = LOADS[1]
    up = load_factor_upper(pad, w, MU, O)
    lo = load_factor_lower(pad, w, MU, O)
    assert classify(pad, 1.1 * up * w, MU, O)[0] == "slip"
    assert classify(pad, 0.9 * lo * w, MU, O)[0] == "hold"
