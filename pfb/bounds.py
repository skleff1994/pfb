"""Measurement-conditioned bounds on the support function of W(nu), Sections 4-7.

Each GMP of the draft is solved as an LP over the weights of a measure on the support grid
(variables w_k >= 0, linear moment constraints, objective mu * sum_k w_k g(x_k; q, c)).
The true measure lives on the same grid, so it is always feasible: for every q,
lower <= h_true <= upper holds exactly for the discrete problems. Compared to the
continuum bounds, the discrete ones are inner approximations (lower bound slightly too
high, upper bound slightly too low) with an O(grid spacing) error.

All bound functions take q of shape (3,) or (m, 3) and return (lower, upper) arrays of
shape () or (m,).
"""

import numpy as np
from scipy import sparse
from scipy.optimize import linprog, minimize

from .measure import integrand

# --- measurements (Table 2) --------------------------------------------------------------

def measure_M0(nu):
    """Coarse measurement y_0 = (f_N, c), eq. (38) (the draft writes y_0 = (f_N, f_N c))."""
    return nu.fN, nu.cop


def measure_MF(nu, taxels):
    """Taxel forces y_F = (F_1, ..., F_r), eq. (45)."""
    return taxels.downsample(nu)


def measure_MT(nu, taxels):
    """Combined measurement y_T = (f_N, c, F), eqs. (47)-(48)."""
    return nu.fN, nu.cop, taxels.downsample(nu)


# --- LP machinery ------------------------------------------------------------------------

def gmp(points, q, c, A_eq, b_eq, mu=1.0, sense="max", return_measure=False):
    """Solve  sup/inf_w  mu sum_k w_k ||a + bJ(x_k - c)||  s.t.  A_eq w = b_eq, w >= 0.

    Returns the optimal value (nan if infeasible), and the optimal weights if asked.
    Dual simplex returns a vertex, i.e. a measure with at most rank(A_eq) atoms.
    """
    sign = -1.0 if sense == "max" else 1.0
    cost = mu * integrand(points, q, c)
    res = linprog(sign * cost, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs-ds")
    val = sign * res.fun if res.status == 0 else np.nan
    if return_measure:
        return val, (res.x if res.status == 0 else None)
    return val


def cop_rows(points, c):
    """Rows of the CoP consistency constraint int (x - c) dnu = 0, shape (2, K)."""
    return (points - c).T


def taxel_rows(taxels):
    """Indicator rows nu(P_i), shape (r, K), sparse."""
    K = taxels.patch.K
    return sparse.csr_matrix((np.ones(K), (taxels.index, np.arange(K))), shape=(taxels.r, K))


def _batch(fn, q):
    """Apply fn(q_single) -> (lo, up) to one q or to every row of a batch."""
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        return fn(q)
    lo, up = zip(*(fn(qi) for qi in q))
    return np.array(lo), np.array(up)


# --- M_0: total force and CoP (Section 4) ------------------------------------------------

def bounds_M0(patch, y0, q, mu=1.0, lower="exact"):
    """(h_lower, h_upper) given the coarse measurement y0 = (f_N, c).

    lower="exact" uses the closed form mu f_N ||a|| of Thm 4.3; lower="lp" solves the
    discrete GMP instead (useful as a cross-check). The upper bound is always the LP.
    """
    fN, c = y0
    A = np.vstack([np.ones(patch.K), cop_rows(patch.points, c)])
    b = np.array([fN, 0.0, 0.0])

    def one(qi):
        """(lower, upper) for one direction qi."""
        up = gmp(patch.points, qi, c, A, b, mu, "max")
        lo = mu * fN * np.linalg.norm(qi[:2]) if lower == "exact" else gmp(patch.points, qi, c, A, b, mu, "min")
        return lo, up

    return _batch(one, q)


def extremal_M0(patch, y0, q, mu=1.0, sense="max"):
    """Optimal value and measure weights of the M_0 GMP (to inspect the at most 3 atoms of Thm 4.5)."""
    fN, c = y0
    A = np.vstack([np.ones(patch.K), cop_rows(patch.points, c)])
    return gmp(patch.points, q, c, A, [fN, 0.0, 0.0], mu, sense, return_measure=True)


# --- M_T: taxel forces and CoP (Section 6) -----------------------------------------------

def bounds_MT(taxels, yT, q, mu=1.0):
    """Exact (discrete) M_T bounds, eqs. (54)-(55)."""
    _, c, F = yT
    pts = taxels.patch.points
    A = sparse.vstack([taxel_rows(taxels), sparse.csr_matrix(cop_rows(pts, c))]).tocsr()
    b = np.concatenate([F, [0.0, 0.0]])
    return _batch(lambda qi: (gmp(pts, qi, c, A, b, mu, "min"), gmp(pts, qi, c, A, b, mu, "max")), q)


def cellwise_g(taxels, q, c):
    """g_i^min and g_i^max of eqs. (61)-(62) over the (continuous) taxel rectangles.

    With p = c + J a / b one has ||a + bJ(x - c)|| = |b| ||x - p||, so g_min is |b| times the
    distance from p to P_i and g_max is |b| times the largest distance from p to a corner.
    """
    a, b = np.asarray(q[:2], float), float(q[2])
    if b == 0.0:
        g = np.full(taxels.r, np.linalg.norm(a))
        return g, g
    p = c + np.array([-a[1], a[0]]) / b  # c + J a / b
    lo, hi = taxels.lo, taxels.hi
    near = np.maximum(np.maximum(lo - p, p - hi), 0.0)
    far = np.maximum(np.abs(p - lo), np.abs(p - hi))
    return abs(b) * np.linalg.norm(near, axis=1), abs(b) * np.linalg.norm(far, axis=1)


def cellwise_MT(taxels, yT, q, mu=1.0):
    """Constructive cellwise bounds mu sum_i F_i g_i^{min/max}(q; c*), Prop 6.4 and eq. (67)."""
    _, c, F = yT

    def one(qi):
        """(lower, upper) for one direction qi: per-taxel extremes of g, weighted by the taxel forces."""
        gmin, gmax = cellwise_g(taxels, qi, c)
        return mu * F @ gmin, mu * F @ gmax

    return _batch(one, q)


# --- M_F: taxel forces only (Section 7) --------------------------------------------------

def cop_set_MF(taxels, F):
    """Compatible CoP set of Prop. 5.6 as a rectangle (lo, hi) = sum_i (F_i / f_N) P_i."""
    s = F / F.sum()
    return s @ taxels.lo, s @ taxels.hi


def fixed_ref_MF(taxels, F, c_hat, q, mu=1.0):
    """Fixed-reference decoupled approximation, eq. (73) (no CoP constraint).

    Not a bound on the M_F problem, since the true CoP differs from c_hat: see
    `certified_ref_MF` for the corrected version (Prop. 7.4).
    """
    return cellwise_MT(taxels, (None, np.asarray(c_hat, float), F), q, mu)


def outer_search(f, lo_c, hi_c, sense, n_c=7, polish=True, xatol=1e-4):
    """Optimize c -> f(c) over the rectangle [lo_c, hi_c] (the CoP set C_MF).

    Grid search on n_c x n_c points, then optionally Nelder-Mead from the best grid point.
    f returns nan where infeasible. Returns (best value, best c). The search is approximate:
    for sense="min" the result may sit slightly above the true infimum, and for "max" below the supremum.
    """
    s = 1.0 if sense == "min" else -1.0  # minimize s * value
    axes = [np.linspace(l, h, n_c) if h > l else np.array([l]) for l, h in zip(lo_c, hi_c)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
    vals = np.array([f(c) for c in grid])
    vals = np.where(np.isnan(vals), np.inf, s * vals)
    k = int(vals.argmin())
    best_c, best = grid[k], vals[k]
    free = hi_c > lo_c  # do not move along a degenerate axis (segment patch)
    if polish and free.any():
        def g(z):
            """Signed objective at the free coordinates z of c (clipped to the box; NaN, i.e. infeasible, -> inf)."""
            c = best_c.copy(); c[free] = np.clip(z, lo_c[free], hi_c[free])
            v = f(c)
            return np.inf if np.isnan(v) else s * v

        res = minimize(g, best_c[free], method="Nelder-Mead",
                       options=dict(xatol=xatol, fatol=1e-10, maxiter=200))
        if res.fun < best:
            best = res.fun
            best_c = best_c.copy(); best_c[free] = np.clip(res.x, lo_c[free], hi_c[free])
    return s * best, best_c


def bounds_MF(taxels, F, q, mu=1.0, n_c=7, polish=True, return_cop=False):
    """Bi-level M_F bounds on the support grid, Prop 7.1, eq. (68).

    Outer: `outer_search` over the rectangle C_MF(y_F) (polish=False skips the Nelder-Mead
    refinement, ~3x faster). Inner: grid LP with the taxel constraints and the CoP consistency
    constraint int (x - c) dnu = 0, which is infeasible when c is not in C_MF.
    Because the outer search is approximate, the returned lower bound may sit slightly above
    the true inf over c, and the upper bound slightly below the true sup (~1e-5 without
    polishing on the examples, at the solver tolerance with it).
    With return_cop=True, also returns the CoPs achieving the lower and upper values.
    """
    pts = taxels.patch.points
    T = taxel_rows(taxels)
    b_eq = np.concatenate([F, [0.0, 0.0]])
    lo_c, hi_c = cop_set_MF(taxels, F)
    xatol = 1e-4 * taxels.patch.diam

    def inner(c, qi, sense):
        """Grid LP of the M_T problem for a candidate CoP c (taxel rows + CoP consistency); NaN if c is not in C_MF."""
        A = sparse.vstack([T, sparse.csr_matrix(cop_rows(pts, c))]).tocsr()
        return gmp(pts, qi, c, A, b_eq, mu, sense)

    def one(qi):
        """(lower, upper) for one direction qi: min and max over the CoP c of the inner LP."""
        lo, c_lo = outer_search(lambda c: inner(c, qi, "min"), lo_c, hi_c, "min", n_c, polish, xatol)
        up, c_up = outer_search(lambda c: inner(c, qi, "max"), lo_c, hi_c, "max", n_c, polish, xatol)
        return ((lo, up), (c_lo, c_up)) if return_cop else (lo, up)

    if return_cop:
        q = np.asarray(q, float)
        return one(q) if q.ndim == 1 else [one(qi) for qi in q]
    return _batch(one, q)
