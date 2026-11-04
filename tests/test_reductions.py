"""Checks of the finite-dimensional reductions (Sections 4, 6, 7) against the grid LPs.

The grid bounds are inner approximations of the continuum bounds, so for every q:
    reduced lower <= grid lower <= h_true <= grid upper <= reduced upper,
with the differences shrinking as the support grid is refined.
"""

import numpy as np
import pytest

import pfb

MU = 0.7
TOL = 1e-6
QS = pfb.directions(8, torque_scale=2.0)
Q_TORSION = np.array([0.0, 0.0, 1.0])


def rho(x, y):
    """Smooth pressure with full support (a generic, non-degenerate test case)."""
    return 0.2 + np.exp(-((x - 0.3) ** 2 + (y - 0.35) ** 2) / 0.02) + 0.5 * np.exp(-((x - 0.75) ** 2 + (y - 0.2) ** 2) / 0.01)


def setup(N=24, k=4):
    patch = pfb.Patch(1.0, 0.6, N=N)
    nu = pfb.from_pressure(patch, rho, fN=2.0)
    taxels = pfb.TaxelGrid(patch, k, k)
    return patch, nu, taxels, pfb.measure_MT(nu, taxels)


def test_thm_6_1_6_2_reduced_contains_grid_and_converges():
    errors = []
    for N in (12, 48):
        _, _, tx, yT = setup(N)
        (rlo, rup), (glo, gup) = pfb.reduced_MT(tx, yT, QS, MU), pfb.bounds_MT(tx, yT, QS, MU)
        assert np.all(rlo <= glo + TOL) and np.all(gup <= rup + TOL)
        errors.append(max((glo - rlo).max(), (rup - gup).max()))
    assert errors[1] < errors[0]  # grid -> continuum as N grows


def test_rem_5_9_strict_model_matches_closure():
    """Rectangular taxels: the strict half-open model (grid LPs) gives the reduced (closure-model) values, even when the CoP
    is on the boundary of C_MF. Remark 5.9 only proves that the closure bounds are conservative; this checks equality numerically.

    Every taxel carries its force on its closed left edge, so the CoP has the smallest x-coordinate in C_MF.
    The grid bounds still converge to the reduced (closure) bounds as the support grid is refined.
    """
    gaps = []
    for N in (16, 64):
        patch = pfb.Patch(2.0, 2.0, N=N)
        tx = pfb.TaxelGrid(patch, 2, 2)
        nu = pfb.from_diracs(patch, [(0, 0.5), (1, 0.5), (0, 1.5), (1, 1.5)], [1.0, 2.0, 1.5, 0.5])
        yT = pfb.measure_MT(nu, tx)
        lo, hi = pfb.cop_set_MF(tx, yT[2])
        assert np.isclose(yT[1][0], lo[0])  # CoP on the boundary of C_MF
        (rlo, rup), (glo, gup) = pfb.reduced_MT(tx, yT, QS, MU), pfb.bounds_MT(tx, yT, QS, MU)
        gaps.append(max((glo - rlo).max(), (rup - gup).max()))
    assert gaps[1] < 0.5 * gaps[0] and gaps[1] < 0.05


def test_thm_6_2_vertex_solution_is_sparse():
    _, nu, tx, (fN, c, F) = setup()
    for q in QS:
        _, lam = pfb.vertex_lp_MT(tx, F, c, q, MU, return_weights=True)
        assert np.count_nonzero(lam > 1e-9 * fN) <= tx.r + 2
    # Cor 4.6: single taxel -> at most 3 vertices.
    single = pfb.TaxelGrid(tx.patch, 1, 1)
    for q in QS:
        _, lam = pfb.vertex_lp_MT(single, np.array([fN]), c, q, MU, return_weights=True)
        assert np.count_nonzero(lam > 1e-9 * fN) <= 3


def test_rem_6_3_M0_is_single_taxel_MT():
    patch, nu, _, (fN, c, _) = setup()
    single = pfb.TaxelGrid(patch, 1, 1)
    lo0, up0 = pfb.reduced_M0(patch, (fN, c), QS, MU)
    loT, upT = pfb.reduced_MT(single, (fN, c, np.array([fN])), QS, MU)
    assert np.allclose(lo0, loT, atol=1e-6) and np.allclose(up0, upT)


def test_prop_7_1_thm_7_2_7_3_MF_is_MT_over_cop():
    _, nu, tx, (fN, c_star, F) = setup()
    q = QS[::2]
    loF, upF = pfb.reduced_MF(tx, F, q, MU)
    loT, upT = pfb.reduced_MT(tx, (fN, c_star, F), q, MU)
    assert np.all(loF <= loT + TOL) and np.all(upT <= upF + TOL)
    # The joint SOCP equals the minimum over c in C_MF of the M_T SOCP (checked on a grid of c).
    lo_c, hi_c = pfb.cop_set_MF(tx, F)
    cs = [np.array([x, y]) for x in np.linspace(lo_c[0], hi_c[0], 5) for y in np.linspace(lo_c[1], hi_c[1], 5)]
    min_over_c = np.min([pfb.socp_lower_MT(tx, F, c, q, MU) for c in cs], axis=0)
    assert np.all(loF <= min_over_c + TOL)
    assert np.all(min_over_c - loF <= 0.05 * MU * fN)  # close: the c-grid is coarse
    # Grid M_F bounds are inside the reduced ones.
    glo, gup = pfb.bounds_MF(tx, F, q, MU, n_c=5, polish=False)
    assert np.all(loF <= glo + TOL) and np.all(gup <= upF + TOL)


def test_prop_7_4_certified_fixed_reference():
    _, nu, tx, (fN, c_star, F) = setup()
    loF, upF = pfb.reduced_MF(tx, F, QS, MU, n_c=5)
    for c_hat in (None, c_star, np.array([0.5, 0.3])):
        lo, up = pfb.certified_ref_MF(tx, F, QS, MU, c_hat)
        assert np.all(lo <= loF + TOL) and np.all(upF <= up + TOL)


def test_prop_8_1_corrected_chain():
    patch, nu, tx, yT = setup()
    fN, c, F = yT
    h = pfb.support(nu, QS, MU)
    lo0, up0 = pfb.reduced_M0(patch, (fN, c), QS, MU)
    loF, upF = pfb.reduced_MF(tx, F, QS, MU, n_c=5)
    loT, upT = pfb.reduced_MT(tx, yT, QS, MU)
    assert np.allclose(lo0, MU * fN * np.linalg.norm(QS[:, :2], axis=1))
    assert np.all(lo0 <= loF + TOL) and np.all(loF <= loT + TOL) and np.all(loT <= h + TOL)
    assert np.all(h <= upT + TOL) and np.all(upT <= np.minimum(up0, upF) + TOL)


def test_prop_8_3_gap_bounds_reduced():
    patch = pfb.Patch(1.0, 0.6, N=24)
    nu = pfb.from_pressure(patch, rho, fN=2.0)
    for k in (1, 2, 4, 8):
        tx = pfb.TaxelGrid(patch, k, k)
        yT = pfb.measure_MT(nu, tx)
        s = MU * np.abs(QS[:, 2]) * (yT[2] @ tx.diam)
        loT, upT = pfb.reduced_MT(tx, yT, QS, MU)
        loF, upF = pfb.reduced_MF(tx, yT[2], QS, MU, n_c=5, polish=False)
        assert np.all(upT - loT <= s + TOL) and np.all(upF - loF <= 2 * s + TOL)


@pytest.mark.parametrize("corner", [(0.0, 0.0), (1.0, 0.6)])
def test_extreme_point_cop_no_torsion(corner):
    """CoP at an extreme point of P: no torsional capacity. The corollary stating this is commented out in the draft."""
    patch = pfb.Patch(1.0, 0.6, N=12)
    # CoP at a corner (extreme point): the only compatible measure is the Dirac, no torsion.
    _, up = pfb.reduced_M0(patch, (1.0, np.array(corner)), Q_TORSION, MU)
    assert abs(up) <= TOL
    # Interior CoP: some compatible measure has torsional capacity.
    _, up = pfb.reduced_M0(patch, (1.0, np.array([0.4, 0.2])), Q_TORSION, MU)
    assert up > 0.1
