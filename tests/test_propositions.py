"""Each test checks one statement of the draft numerically, on a few random measures."""

import numpy as np
import pytest

import pfb

MU = 0.7
TOL = 1e-7
rng = np.random.default_rng(0)
PATCH = pfb.Patch(1.0, 0.6, N=24)
QS = pfb.directions(12, torque_scale=2.0)
MEASURES = [pfb.random_measure(PATCH, rng, fN=2.0),
            pfb.random_measure(PATCH, rng, fN=1.0, n_atoms=5),
            pfb.gaussian(PATCH, (0.3, 0.4), 0.15, fN=3.0)]


@pytest.mark.parametrize("nu", MEASURES)
def test_prop_3_2_support_attained_by_gamma_star(nu):
    for q in QS:
        g = pfb.gamma_star(nu, q, MU)
        assert np.all(np.linalg.norm(g, axis=1) <= MU + TOL)  # Coulomb (13)
        fT, tau = pfb.wrench(nu, g)
        assert np.isclose(q[:2] @ fT + q[2] * tau, pfb.support(nu, q, MU))


def test_ex_4_8_counterexample():
    L, fN = 2.0, 3.0
    seg = pfb.Patch(L, 0.0, N=10)
    nu1 = pfb.from_diracs(seg, [(L / 2, 0)], [fN])
    nu2 = pfb.from_diracs(seg, [(0, 0), (L, 0)], [fN / 2, fN / 2])
    assert pfb.measure_M0(nu1)[0] == pfb.measure_M0(nu2)[0]
    assert np.allclose(pfb.measure_M0(nu1)[1], pfb.measure_M0(nu2)[1])
    q = [0, 0, 1]
    assert pfb.support(nu1, q, MU) == 0.0
    assert np.isclose(pfb.support(nu2, q, MU), MU * fN * L / 2)


def test_thm_4_3_coarse_lower_bound():
    # Gaussian centred on the patch centre (a grid node): the CoP is that node, so the
    # minimizer f_N delta_c of Thm 4.3 is representable on the grid.
    nu = pfb.gaussian(PATCH, (0.5, 0.3), 0.2, fN=2.0)
    y0 = pfb.measure_M0(nu)
    lo_exact, _ = pfb.bounds_M0(PATCH, y0, QS, MU)
    lo_lp, _ = pfb.bounds_M0(PATCH, y0, QS, MU, lower="lp")
    assert np.allclose(lo_exact, lo_lp, atol=1e-6)
    # a = 0: no guaranteed torsional capacity.
    assert pfb.bounds_M0(PATCH, y0, [0, 0, 1], MU)[0] == 0.0


@pytest.mark.parametrize("nu", MEASURES)
def test_thm_4_5_upper_attained_by_at_most_3_atoms(nu):
    y0 = pfb.measure_M0(nu)
    for q in QS:
        val, w = pfb.extremal_M0(PATCH, y0, q, MU)
        ext = pfb.Measure(PATCH, w)
        assert len(ext.atoms) <= 3
        assert np.allclose(ext.cop, y0[1]) and np.isclose(ext.fN, y0[0])
        assert np.isclose(pfb.support(ext, q, MU, c=y0[1]), val)


@pytest.mark.parametrize("nu", MEASURES)
def test_prop_5_6_compatible_cop_set(nu):
    tx = pfb.TaxelGrid(PATCH, 4, 2)
    F = pfb.measure_MF(nu, tx)
    lo, hi = pfb.cop_set_MF(tx, F)
    # Redistribute mass randomly inside each taxel: same F, CoP must stay in C_MF.
    for _ in range(20):
        w = rng.exponential(size=PATCH.K) ** 4
        w *= (F / np.bincount(tx.index, weights=w, minlength=tx.r))[tx.index]
        c = pfb.Measure(PATCH, w).cop
        assert np.all(c >= lo - TOL) and np.all(c <= hi + TOL)
    # Diameter bound (52).
    assert np.linalg.norm(hi - lo) <= F @ tx.diam / F.sum() + TOL <= tx.diam.max() + TOL


@pytest.mark.parametrize("nu", MEASURES[:2])
def test_prop_8_2_and_8_3_refinement(nu):
    tx = [pfb.TaxelGrid(PATCH, 1, 1), pfb.TaxelGrid(PATCH, 2, 2), pfb.TaxelGrid(PATCH, 4, 4)]
    q = QS[::3]
    h = pfb.support(nu, q, MU)
    MT = [pfb.bounds_MT(t, pfb.measure_MT(nu, t), q, MU) for t in tx]
    MF = [pfb.bounds_MF(t, pfb.measure_MF(nu, t), q, MU, n_c=5) for t in tx]
    for (lo, up), t in zip(MT, tx):
        F = t.downsample(nu)
        assert np.all(lo <= h + TOL) and np.all(h <= up + TOL)
        assert np.all(up - lo <= MU * np.abs(q[:, 2]) * (F @ t.diam) + TOL)  # (81)
    for (lo, up), t in zip(MF, tx):
        F = t.downsample(nu)
        assert np.all(up - lo <= 2 * MU * np.abs(q[:, 2]) * (F @ t.diam) + TOL)  # (82)
    # Monotone tightening under nested refinement (80); exact for the discrete M_T problem.
    for (lo0, up0), (lo1, up1) in zip(MT, MT[1:]):
        assert np.all(lo0 <= lo1 + TOL) and np.all(up1 <= up0 + TOL)


@pytest.mark.parametrize("nu", MEASURES[:2])
def test_prop_8_1_and_6_4_orderings(nu):
    tx = pfb.TaxelGrid(PATCH, 4, 2)
    q = QS[::2]
    h = pfb.support(nu, q, MU)
    yT = pfb.measure_MT(nu, tx)
    lo0, up0 = pfb.bounds_M0(PATCH, pfb.measure_M0(nu), q, MU)
    loF, upF = pfb.bounds_MF(tx, yT[2], q, MU, n_c=7, polish=True)
    loT, upT = pfb.bounds_MT(tx, yT, q, MU)
    loC, upC = pfb.cellwise_MT(tx, yT, q, MU)
    # (76)-(77). M_F uses an approximate outer search, hence the looser tolerance there.
    assert np.all(lo0 <= loF + TOL)  # (76): the lower bounds of M_0 and M_F are ordered
    assert np.all(lo0 <= loT + TOL) and np.all(upT <= up0 + TOL)
    assert np.all(loF <= loT + 1e-3) and np.all(upT <= upF + 1e-3)
    assert np.all(loT <= h + TOL) and np.all(h <= upT + TOL)
    # (67): cellwise bounds bracket the exact M_T bounds.
    assert np.all(loC <= loT + TOL) and np.all(upT <= upC + TOL)
