"""Measurement-conditioned bounds on the support function, solved as LPs on the support grid.

Torques and twists are about the origin of the patch coordinates, so every bound is a generalized moment
problem (GMP): the objective mu int ||a + bJx|| dnu is linear in the measure, and the measurements are linear
moment constraints. On the support grid it becomes an LP over the node weights w_k >= 0. The true measure lives
on the same grid, so it is always feasible: for every q, lower <= h_true <= upper holds exactly for the discrete
problems. Compared to the continuum bounds, the discrete ones are inner approximations (lower bound slightly too
high, upper bound slightly too low): the grid has half-open taxels and misses the corners owned by neighbours,
with an O(|b| grid spacing) error. `pfb.reduced` gives the exact finite-dimensional forms.

Measurements use homogeneous coordinates: y_0 = (f_N, m_1) with m_1 = int x dnu = f_N c, valid for f_N = 0.
All bound functions take q of shape (3,) or (m, 3) and return (lower, upper) arrays of shape () or (m,).
"""

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .measure import integrand

# --- measurements -------------------------------------------------------------------------

def measure_M0(nu):
    """Coarse measurement y_0 = (f_N, m_1): resultant normal force and first moment."""
    return nu.fN, nu.m1


def measure_MF(nu, taxels):
    """Taxel forces y_F = (F_1, ..., F_r)."""
    return taxels.downsample(nu)


def measure_MT(nu, taxels):
    """Combined measurement y_T = (f_N, m_1, F)."""
    return nu.fN, nu.m1, taxels.downsample(nu)


# --- LP machinery ------------------------------------------------------------------------

def gmp(points, q, A_eq, b_eq, mu=1.0, sense="max", return_measure=False):
    """Solve  sup/inf_w  mu sum_k w_k ||a + bJ x_k||  s.t.  A_eq w = b_eq, w >= 0.

    Returns the optimal value (nan if infeasible), and the optimal weights if asked.
    Dual simplex returns a vertex, i.e. a measure with at most rank(A_eq) atoms.
    """
    sign = -1.0 if sense == "max" else 1.0
    cost = mu * integrand(points, q)
    res = linprog(sign * cost, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs-ds")
    val = sign * res.fun if res.status == 0 else np.nan
    if return_measure:
        return val, (res.x if res.status == 0 else None)
    return val


def moment_rows(points):
    """Rows of the zeroth and first moments int 1 dnu, int x dnu, shape (3, K)."""
    return np.vstack([np.ones(len(points)), points.T])


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


def coarse_lower(y0, q, mu=1.0):
    """Exact coarse lower bound mu ||f_N a + bJ m_1|| (thm:coarse_lower), vectorized over q."""
    fN, m1 = y0
    q = np.asarray(q, float)
    Jm1 = np.array([-m1[1], m1[0]])
    return mu * np.linalg.norm(fN * q[..., :2] + q[..., 2:3] * Jm1, axis=-1)


# --- M_0: total force and first moment ----------------------------------------------------

def bounds_M0(patch, y0, q, mu=1.0, lower="exact"):
    """(h_lower, h_upper) given the coarse measurement y0 = (f_N, m_1).

    lower="exact" uses the closed form mu ||f_N a + bJ m_1||; lower="lp" solves the discrete GMP instead
    (useful as a cross-check). The upper bound is always the LP.
    """
    fN, m1 = y0
    A = moment_rows(patch.points)
    b = np.r_[fN, m1]

    def one(qi):
        """(lower, upper) for one direction qi."""
        up = gmp(patch.points, qi, A, b, mu, "max")
        lo = coarse_lower(y0, qi, mu) if lower == "exact" else gmp(patch.points, qi, A, b, mu, "min")
        return lo, up

    return _batch(one, q)


def extremal_M0(patch, y0, q, mu=1.0, sense="max"):
    """Optimal value and measure weights of the M_0 GMP (to inspect the at most 3 atoms of the upper bound)."""
    fN, m1 = y0
    return gmp(patch.points, q, moment_rows(patch.points), np.r_[fN, m1], mu, sense, return_measure=True)


# --- M_F and M_T ------------------------------------------------------------------------

def bounds_MF(taxels, F, q, mu=1.0):
    """Discrete M_F bounds: grid LPs with the taxel constraints only."""
    pts = taxels.patch.points
    A = taxel_rows(taxels)
    F = np.asarray(F, float)
    return _batch(lambda qi: (gmp(pts, qi, A, F, mu, "min"), gmp(pts, qi, A, F, mu, "max")), q)


def bounds_MT(taxels, yT, q, mu=1.0):
    """Discrete M_T bounds: grid LPs with the taxel constraints and the first-moment constraint."""
    _, m1, F = yT
    pts = taxels.patch.points
    A = sparse.vstack([taxel_rows(taxels), sparse.csr_matrix(pts.T)]).tocsr()
    b = np.concatenate([F, m1])
    return _batch(lambda qi: (gmp(pts, qi, A, b, mu, "min"), gmp(pts, qi, A, b, mu, "max")), q)


def cellwise_g(taxels, q, o=None):
    """min and max of g(x; q) = ||a + bJ(x - o)|| over each taxel closure (rectangles), shape (r,) each.

    With p = o + J a / b (the instantaneous center of rotation) one has g = |b| ||x - p||, so g_min is |b| times
    the distance from p to the taxel and g_max is |b| times the largest distance from p to a corner.
    These are the exact M_F bounds once weighted by the taxel forces (thm:MF).
    """
    a, b = np.asarray(q[:2], float), float(q[2])
    o = np.zeros(2) if o is None else np.asarray(o, float)
    if b == 0.0:
        g = np.full(taxels.r, np.linalg.norm(a))
        return g, g
    p = o + np.array([-a[1], a[0]]) / b  # o + J a / b
    lo, hi = taxels.lo, taxels.hi
    near = np.maximum(np.maximum(lo - p, p - hi), 0.0)
    far = np.maximum(np.abs(p - lo), np.abs(p - hi))
    return abs(b) * np.linalg.norm(near, axis=1), abs(b) * np.linalg.norm(far, axis=1)


def cop_set_MF(taxels, F):
    """Compatible CoPs given the taxel forces (closure model), the rectangle sum_i (F_i / f_N) P_i, as (lo, hi).

    The bounds do not use this set (rem:cop_taxel); it is used in figures.
    """
    s = F / F.sum()
    return s @ taxels.lo, s @ taxels.hi
