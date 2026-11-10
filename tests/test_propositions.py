"""Each test checks one statement of the theory note numerically (labels in brackets), on a few measures.

Torques and twists are about the origin of the patch coordinates unless a reference point is passed.
"""

import numpy as np
import pytest

import pfb
from pfb.certify import load_factor_true

MU = 0.7
TOL = 1e-7
rng = np.random.default_rng(0)
PATCH = pfb.Patch(1.0, 0.6, N=24)
QS = pfb.directions(12, torque_scale=2.0)
QS = np.vstack([QS, pfb.shift_twist(QS, (0.5, 0.3))])  # also twists about the pad centre
MEASURES = [pfb.random_measure(PATCH, rng, fN=2.0),
            pfb.random_measure(PATCH, rng, fN=1.0, n_atoms=5),
            pfb.gaussian(PATCH, (0.3, 0.4), 0.15, fN=3.0)]


@pytest.mark.parametrize("nu", MEASURES)
def test_prop_support_attained_by_gamma_star(nu):
    """[prop:support] The field gamma_star is admissible and its wrench attains h(q), about the origin and about o."""
    for o in (None, np.array([0.4, 0.1])):
        for q in QS:
            g = pfb.gamma_star(nu, q, MU, o)
            assert np.all(np.linalg.norm(g, axis=1) <= MU + TOL)
            fT, tau = pfb.wrench(nu, g, o)
            assert np.isclose(q[:2] @ fT + q[2] * tau, pfb.support(nu, q, MU, o))


@pytest.mark.parametrize("nu", MEASURES)
def test_lem_varignon(nu):
    """[lem:varignon] h_{X_o}(q) = h_X(S_o^{-1} q), and the power <q, w> does not depend on the reference point."""
    o = np.array([0.7, -0.2])
    assert np.allclose(pfb.support(nu, QS, MU, o), pfb.support(nu, pfb.shift_twist(QS, o), MU))
    for q in QS[:5]:
        fT, tau = pfb.wrench(nu, pfb.gamma_star(nu, q, MU))       # wrench about the origin
        fo, tau_o = pfb.wrench(nu, pfb.gamma_star(nu, q, MU), o)  # the same field, torque about o
        assert np.isclose(tau_o, tau - (o[0] * fT[1] - o[1] * fT[0]))  # T_o
        q_o = np.r_[q[:2] + q[2] * (pfb.J @ o), q[2]]                 # S_o
        assert np.isclose(q_o[:2] @ fo + q_o[2] * tau_o, q[:2] @ fT + q[2] * tau)


def test_lem_linearity():
    """[lem:linearity] h_{X(nu1 + nu2)} = h_{X(nu1)} + h_{X(nu2)} and h_{X(t nu)} = t h_{X(nu)}."""
    a, b = MEASURES[0], MEASURES[2]
    assert np.allclose(pfb.support(a + b, QS, MU), pfb.support(a, QS, MU) + pfb.support(b, QS, MU))
    assert np.allclose(pfb.support(2.5 * a, QS, MU), 2.5 * pfb.support(a, QS, MU))


def test_ex_segment_counterexample():
    """[ex:segment] Same (f_N, m_1), different support for the rotation about the CoP."""
    L, fN = 2.0, 3.0
    seg = pfb.Patch(L, 0.0, N=10)
    nu1 = pfb.from_diracs(seg, [(L / 2, 0)], [fN])
    nu2 = pfb.from_diracs(seg, [(0, 0), (L, 0)], [fN / 2, fN / 2])
    assert pfb.measure_M0(nu1)[0] == pfb.measure_M0(nu2)[0]
    assert np.allclose(pfb.measure_M0(nu1)[1], pfb.measure_M0(nu2)[1])
    q_c = pfb.shift_twist([0, 0, 1], nu1.cop)
    assert abs(pfb.support(nu1, q_c, MU)) < 1e-12
    assert np.isclose(pfb.support(nu2, q_c, MU), MU * fN * L / 2)
    assert np.isclose(pfb.reduced_M0(seg, pfb.measure_M0(nu1), q_c, MU)[1], MU * fN * L / 2)


def test_thm_coarse_lower_and_no_torsion_about_cop():
    """[thm:coarse_lower, cor:coarse_inner] Closed form = grid LP; zero for the rotations about the CoP."""
    # Gaussian centred on the patch centre (a grid node): the minimizer f_N delta_c is representable on the grid.
    nu = pfb.gaussian(PATCH, (0.5, 0.3), 0.2, fN=2.0)
    y0 = pfb.measure_M0(nu)
    lo_exact, _ = pfb.bounds_M0(PATCH, y0, QS, MU)
    lo_lp, _ = pfb.bounds_M0(PATCH, y0, QS, MU, lower="lp")
    assert np.allclose(lo_exact, lo_lp, atol=1e-6)
    for b in (1.0, -2.0):
        assert abs(pfb.coarse_lower(y0, pfb.shift_twist([0, 0, b], nu.cop), MU)) < 1e-9
    # Homogeneous coordinates: f_N = 0 is allowed.
    assert pfb.coarse_lower((0.0, np.zeros(2)), QS, MU).max() == 0.0


@pytest.mark.parametrize("nu", MEASURES)
def test_cor_coarse_inner_uniform_field(nu):
    """[cor:coarse_inner] (f, c x f) with ||f|| <= mu f_N is carried by every measure with the same (f_N, m_1)."""
    c = nu.cop
    for f in rng.normal(size=(4, 2)):
        f *= 0.999 * MU * nu.fN / np.linalg.norm(f)
        w = np.r_[f, c[0] * f[1] - c[1] * f[0]]
        assert load_factor_true(nu.patch.points[nu.w > 0], nu.w[nu.w > 0], w, MU) >= 1 - 1e-5


@pytest.mark.parametrize("nu", MEASURES)
def test_thm_coarse_upper_at_most_3_atoms(nu):
    """[prop:coarse_upper_atomic] The grid LP returns a compatible measure with at most 3 atoms attaining the bound."""
    y0 = pfb.measure_M0(nu)
    for q in QS:
        val, w = pfb.extremal_M0(PATCH, y0, q, MU)
        ext = pfb.Measure(PATCH, w)
        assert len(ext.atoms) <= 3
        assert np.allclose(ext.m1, y0[1]) and np.isclose(ext.fN, y0[0])
        assert np.isclose(pfb.support(ext, q, MU), val)


@pytest.mark.parametrize("nu", MEASURES)
def test_cor_coarse_upper_triangles_equal_lp(nu):
    """[rem:coarse_triangles] Max over the vertex triangles containing the CoP = patch-vertex LP."""
    y0 = pfb.measure_M0(nu)
    _, up = pfb.reduced_M0(PATCH, y0, QS, MU)
    assert np.allclose(pfb.coarse_upper_triangles(PATCH, y0, QS, MU), up, rtol=1e-7, atol=1e-9)


def test_prop_inner_outer_possible_set_is_convex():
    """[prop:inner_outer (i)] Wrenches of two compatible measures: their midpoint is carried by the average measure."""
    tx = pfb.TaxelGrid(PATCH, 4, 2)
    nu1 = pfb.random_measure(PATCH, rng, fN=1.0)
    F = tx.downsample(nu1)
    w2 = rng.exponential(size=PATCH.K) ** 4  # another measure with the same taxel forces
    nu2 = pfb.Measure(PATCH, w2 * (F / np.bincount(tx.index, weights=w2, minlength=tx.r))[tx.index])
    assert np.allclose(tx.downsample(nu2), F)
    mix = 0.5 * (nu1 + nu2)
    for q1, q2 in zip(QS[:6], QS[6:12]):
        w = 0.5 * (pfb.support_points(nu1, q1, MU) + pfb.support_points(nu2, q2, MU))
        assert load_factor_true(PATCH.points, mix.w, w, MU) >= 1 - 1e-5


@pytest.mark.parametrize("nu", MEASURES[:2])
def test_prop_grid_chain_monotone(nu):
    """[prop:grid_chain, lem:monotone] Nested dyadic grids: the M_F and M_T bounds tighten; exact for the grid LPs."""
    patch = pfb.Patch(1.0, 0.6, N=32)
    nu = pfb.random_measure(patch, np.random.default_rng(len(nu.atoms)), fN=nu.fN)
    grids = [pfb.TaxelGrid(patch, 2**p, 2**p) for p in range(4)] + [pfb.TaxelGrid(patch, 16, 8)]
    for bounds, meas in ((pfb.bounds_MF, pfb.measure_MF), (pfb.bounds_MT, pfb.measure_MT)):
        B = [bounds(t, meas(nu, t), QS[::3], MU) for t in grids]
        for (lo0, up0), (lo1, up1) in zip(B, B[1:]):
            assert np.all(lo0 <= lo1 + TOL) and np.all(up1 <= up0 + TOL)
    for B in ([pfb.reduced_MF(t, t.downsample(nu), QS, MU) for t in grids],
              [pfb.reduced_MT(t, pfb.measure_MT(nu, t), QS[::3], MU) for t in grids]):
        for (lo0, up0), (lo1, up1) in zip(B, B[1:]):
            assert np.all(lo0 <= lo1 + 1e-6) and np.all(up1 <= up0 + 1e-6)


@pytest.mark.parametrize("nu", MEASURES[:2])
def test_prop_taxel_refinement_orderings(nu):
    """[prop:taxel_refinement] max(lo M0, lo MF) <= lo MT <= h <= up MT <= min(up M0, up MF) (grid LPs, exact)."""
    tx = pfb.TaxelGrid(PATCH, 4, 2)
    q = QS[::2]
    h = pfb.support(nu, q, MU)
    yT = pfb.measure_MT(nu, tx)
    lo0, up0 = pfb.bounds_M0(PATCH, pfb.measure_M0(nu), q, MU)
    loF, upF = pfb.bounds_MF(tx, yT[2], q, MU)
    loT, upT = pfb.bounds_MT(tx, yT, q, MU)
    assert np.all(np.maximum(lo0, loF) <= loT + TOL) and np.all(loT <= h + TOL)
    assert np.all(h <= upT + TOL) and np.all(upT <= np.minimum(up0, upF) + TOL)


@pytest.mark.parametrize("nu", MEASURES)
def test_lem_taxel_reduction(nu):
    """[lem:taxel_reduction] mu sum_i F_i g(u_i/F_i) <= h(nu) <= mu sum_ik lambda_ik g(v_ik), with the bilinear
    corner weights lambda_ik = int alpha_ik dnu on each taxel (about the origin and about a point o)."""
    x = PATCH.points
    for k in (1, 2, 4):
        t = pfb.TaxelGrid(PATCH, k, k)
        F = t.downsample(nu)
        V = pfb.taxel_vertices(t)  # (r, 4, 2): lo, (hi_x, lo_y), hi, (lo_x, hi_y)
        size = (t.hi - t.lo)[t.index]
        xi = (x - t.lo[t.index]) / size  # local coordinates in the closed taxel
        alpha = np.stack([(1 - xi[:, 0]) * (1 - xi[:, 1]), xi[:, 0] * (1 - xi[:, 1]),
                          xi[:, 0] * xi[:, 1], (1 - xi[:, 0]) * xi[:, 1]], axis=1)
        lam, u = np.zeros((t.r, 4)), np.zeros((t.r, 2))
        np.add.at(lam, t.index, nu.w[:, None] * alpha)
        np.add.at(u, t.index, nu.w[:, None] * x)
        assert np.allclose(lam.sum(1), F) and np.allclose(np.einsum("ik,ikd->id", lam, V), u)
        assert np.all(lam >= -TOL)
        pos = F > 1e-12
        for o in (None, np.array([0.4, 0.1])):
            h = pfb.support(nu, QS, MU, o)
            low = MU * pfb.integrand(u[pos] / F[pos, None], QS, o) @ F[pos]
            up = MU * (pfb.integrand(V.reshape(-1, 2), QS, o) @ lam.reshape(-1))
            assert np.all(low <= h + TOL) and np.all(h <= up + TOL)


@pytest.mark.parametrize("nu", MEASURES[:2])
def test_prop_gap_bounds(nu):
    """[prop:taxel_resolution_MT] gap <= mu sum_i F_i (max - min of g) <= mu |b| f_N D, for M_F and M_T."""
    for k in (1, 2, 4):
        t = pfb.TaxelGrid(PATCH, k, k)
        yT = pfb.measure_MT(nu, t)
        s = MU * np.abs(QS[:, 2]) * (yT[2] @ t.diam)
        stated = MU * np.abs(QS[:, 2]) * yT[0] * t.diam.max()  # the statement: mu |b| f_N D on a uniform grid
        loF, upF = pfb.reduced_MF(t, yT[2], QS, MU)
        loT, upT = pfb.reduced_MT(t, yT, QS, MU)
        assert np.all(upT - loT <= upF - loF + 1e-6) and np.all(upF - loF <= s + TOL)
        assert np.allclose(s, stated) and np.all(upT - loT <= stated + 1e-6)


def test_thm_rate_first_order_and_halving():
    """[thm:rate] For a measure with a density, gap / h approaches the first-order constant mu |b| C_nu and one dyadic
    step divides the gap by about 2. The error is also checked against an explicit O(h^2) bound (bounded density),
    a quantitative version that the theory note states only as a limit."""
    patch = pfb.Patch(1.0, 1.0, N=128)
    a = 0.3
    nu = pfb.hertz(patch, (0.45, 0.55), a, fN=1.0)
    rho_max = nu.w.max() / patch.quad[patch.quad > 0].min()  # bound on the discrete density
    for q in (pfb.shift_twist([0, 0, 1.0], nu.cop), pfb.shift_twist([1.0, 0, 5.0], (0.5, 0.5))):
        b = abs(q[2])
        p = pfb.J @ q[:2] / q[2]
        d = patch.points - p
        n = np.linalg.norm(d, axis=1)
        n[n == 0] = np.inf
        C = nu.w @ ((np.abs(d[:, 0]) + np.abs(d[:, 1])) / n)
        gaps, rel = [], []
        for k in (16, 32, 64):
            t = pfb.TaxelGrid(patch, k, k)
            lo, up = pfb.reduced_MF(t, t.downsample(nu), q, MU)
            h, D = 1.0 / k, np.sqrt(2) / k
            err = abs(up - lo - MU * b * C * h)
            assert err <= MU * b * rho_max * D**2 * (15 / 4 * np.sqrt(np.pi) + 9 * np.pi / 4 * D)
            rel.append(err / (MU * b * C * h))  # relative distance of gap / h to the limit
            gaps.append(up - lo)
        assert 0.45 < gaps[2] / gaps[1] < 0.55 and 0.45 < gaps[1] / gaps[0] < 0.55
        assert rel[-1] < 0.1 and rel[-1] <= rel[0] + 1e-9  # gap / h converges to the limit of the theorem
        assert nu.fN - 1e-9 <= C <= np.sqrt(2) * nu.fN + 1e-9


@pytest.mark.parametrize("corner", [(0.0, 0.0), (1.0, 0.6)])
def test_cop_at_vertex_no_torsion_about_cop(corner):
    """CoP at a vertex of P: the only compatible measure is the point load, no torsion about the CoP."""
    patch = pfb.Patch(1.0, 0.6, N=12)
    c = np.array(corner)
    q_c = pfb.shift_twist([0, 0, 1.0], c)
    _, up = pfb.reduced_M0(patch, (1.0, c), q_c, MU)
    assert abs(up) <= 1e-6
    c = np.array([0.4, 0.2])
    _, up = pfb.reduced_M0(patch, (1.0, c), pfb.shift_twist([0, 0, 1.0], c), MU)
    assert up > 0.1
