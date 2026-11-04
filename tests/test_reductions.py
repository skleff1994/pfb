"""Checks of the exact finite-dimensional forms (`pfb.reduced`) against the grid LPs (`pfb.bounds`).

The grid bounds are inner approximations of the continuum bounds (half-open cells on a finite grid), so for every q:
    reduced lower <= grid lower <= h_true <= grid upper <= reduced upper,
with the differences vanishing as the support grid is refined (lem:half_open).
"""

import numpy as np
import pytest

import pfb

MU = 0.7
TOL = 1e-6
QS = pfb.directions(8, torque_scale=2.0)
QS = np.vstack([QS, pfb.shift_twist(QS, (0.5, 0.3))])


def rho(x, y):
    """Smooth pressure with full support (a generic, non-degenerate test case)."""
    return 0.2 + np.exp(-((x - 0.3) ** 2 + (y - 0.35) ** 2) / 0.02) + 0.5 * np.exp(-((x - 0.75) ** 2 + (y - 0.2) ** 2) / 0.01)


def setup(N=24, k=4):
    patch = pfb.Patch(1.0, 0.6, N=N)
    nu = pfb.from_pressure(patch, rho, fN=2.0)
    taxels = pfb.TaxelGrid(patch, k, k)
    return patch, nu, taxels, pfb.measure_MT(nu, taxels)


@pytest.mark.parametrize("which", ["MT", "MF"])
def test_reduced_contains_grid_and_converges(which):
    """[thm:MF, thm:combined_lower/upper, lem:half_open] grid inside reduced; the gap vanishes as N grows."""
    errors = []
    for N in (12, 48):
        _, nu, tx, yT = setup(N)
        if which == "MT":
            (rlo, rup), (glo, gup) = pfb.reduced_MT(tx, yT, QS, MU), pfb.bounds_MT(tx, yT, QS, MU)
        else:
            (rlo, rup), (glo, gup) = pfb.reduced_MF(tx, yT[2], QS, MU), pfb.bounds_MF(tx, yT[2], QS, MU)
        h = pfb.support(nu, QS, MU)
        assert np.all(rlo <= glo + TOL) and np.all(glo <= h + TOL) and np.all(h <= gup + TOL) and np.all(gup <= rup + TOL)
        errors.append(max((glo - rlo).max(), (rup - gup).max()))
    assert errors[1] < 0.5 * errors[0]


def test_half_open_cop_on_boundary_of_compatible_set():
    """[lem:half_open] Every taxel carries its force on its closed left edge, so the CoP is extreme in the compatible set;
    the half-open grid bounds still converge to the closure values."""
    gaps = []
    for N in (16, 64):
        patch = pfb.Patch(2.0, 2.0, N=N)
        tx = pfb.TaxelGrid(patch, 2, 2)
        nu = pfb.from_diracs(patch, [(0, 0.5), (1, 0.5), (0, 1.5), (1, 1.5)], [1.0, 2.0, 1.5, 0.5])
        yT = pfb.measure_MT(nu, tx)
        lo, _ = pfb.cop_set_MF(tx, yT[2])
        assert np.isclose(nu.cop[0], lo[0])
        (rlo, rup), (glo, gup) = pfb.reduced_MT(tx, yT, QS, MU), pfb.bounds_MT(tx, yT, QS, MU)
        gaps.append(max((glo - rlo).max(), (rup - gup).max()))
    assert gaps[1] < 0.5 * gaps[0] and gaps[1] < 0.1


def test_MF_closed_form_is_cellwise_min_max():
    """[thm:MF] Closed forms = per-taxel min / max of g over a fine sampling of each closure."""
    _, nu, tx, (_, _, F) = setup()
    lo, up = pfb.reduced_MF(tx, F, QS, MU)
    s = np.linspace(0, 1, 41)
    for q, l, u in zip(QS, lo, up):
        gmin, gmax = [], []
        for a, b in zip(tx.lo, tx.hi):
            X = np.stack(np.meshgrid(a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])), -1).reshape(-1, 2)
            g = pfb.integrand(X, q)
            gmin.append(g.min()); gmax.append(g.max())
        assert np.isclose(u, MU * F @ np.array(gmax))
        assert MU * F @ np.array(gmin) - 1e-3 <= l <= MU * F @ np.array(gmin) + TOL


def test_vertex_solution_is_sparse():
    """[thm:combined_upper, cor:coarse_upper_polygon] at most r + 2 (resp. 3) nonzero corner weights."""
    _, nu, tx, (fN, m1, F) = setup()
    for q in QS:
        _, lam = pfb.vertex_lp_MT(tx, F, m1, q, MU, return_weights=True)
        assert np.count_nonzero(lam > 1e-9 * fN) <= tx.r + 2
    single = pfb.TaxelGrid(tx.patch, 1, 1)
    for q in QS:
        _, lam = pfb.vertex_lp_MT(single, np.array([fN]), m1, q, MU, return_weights=True)
        assert np.count_nonzero(lam > 1e-9 * fN) <= 3


def test_M0_is_single_taxel_MT():
    """[rem:MT_special] M_0 is the single-taxel case of M_T."""
    patch, nu, _, (fN, m1, _) = setup()
    single = pfb.TaxelGrid(patch, 1, 1)
    lo0, up0 = pfb.reduced_M0(patch, (fN, m1), QS, MU)
    loT, upT = pfb.reduced_MT(single, (fN, m1, np.array([fN])), QS, MU)
    assert np.allclose(lo0, loT, atol=1e-6) and np.allclose(up0, upT)


def test_MT_inside_MF_and_points_are_compatible():
    """[rem:MT_special, lem:concentrate_spread] M_F bounds bracket M_T; the SOCP minimizer is a compatible measure."""
    _, nu, tx, yT = setup()
    fN, m1, F = yT
    loF, upF = pfb.reduced_MF(tx, F, QS, MU)
    loT, upT = pfb.reduced_MT(tx, yT, QS, MU)
    assert np.all(loF <= loT + TOL) and np.all(upT <= upF + TOL)
    for q in QS[:4]:
        val, C = pfb.socp_lower_MT(tx, F, m1, q, MU, return_points=True)
        assert np.all(C >= tx.lo - 1e-6) and np.all(C <= tx.hi + 1e-6)
        assert np.allclose(F @ C, m1, atol=1e-6)
        assert np.isclose(MU * F @ pfb.integrand(C, q), val, rtol=1e-5, atol=1e-7)


def test_zero_force_taxels():
    """Homogeneous coordinates: taxels without force are allowed (u_i = 0), and the bounds stay exact."""
    patch = pfb.Patch(1.0, 1.0, N=32)
    nu = pfb.gaussian(patch, (0.25, 0.3), 0.06, fN=1.0)
    nu = pfb.Measure(patch, np.where(patch.points[:, 0] < 0.5, nu.w, 0.0))  # right half unloaded
    tx = pfb.TaxelGrid(patch, 4, 4)
    yT = pfb.measure_MT(nu, tx)
    assert np.any(yT[2] == 0)
    (rlo, rup), (glo, gup) = pfb.reduced_MT(tx, yT, QS, MU), pfb.bounds_MT(tx, yT, QS, MU)
    assert np.all(np.isfinite(rlo)) and np.all(rlo <= glo + TOL) and np.all(gup <= rup + TOL)
