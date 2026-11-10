"""Exact finite-dimensional forms of the bounds on the continuous patch (rectangular taxel closures).

Unlike `bounds.py`, which solves the GMPs as LPs on a support grid, these functions implement the closed forms
and convex programs of the theory note directly on the taxel geometry. Torques and twists are about the origin;
measurements are in homogeneous coordinates (f_N, m_1 = int x dnu).

    reduced_M0         closed-form lower bound mu ||f_N a + bJ m_1|| (thm:coarse_lower) and the patch-vertex LP
                       (cor:coarse_upper_polygon), as the single-taxel case of M_T
    coarse_upper_triangles  the same upper bound as a maximum over vertex triangles containing the CoP
    reduced_MF         closed forms mu sum_i F_i min / max of g over each taxel (thm:MF)
    reduced_MT         SOCP lower bound in the local first moments u_i (thm:combined_lower) and the corner LP
                       upper bound (thm:combined_upper)

The values are exact, as infima and suprema over compatible measures, for the half-open taxels of a uniform grid
(boundary push in the proofs of thm:combined_lower and thm:combined_upper). The grid LPs of
`bounds.py` use strict half-open cells on a finite grid, so for every q
    reduced lower <= grid lower <= h_true <= grid upper <= reduced upper,
with the differences vanishing as the support grid is refined.
"""

import itertools

import cvxpy as cp
import numpy as np
from scipy import sparse

from .bounds import _batch, cellwise_g, coarse_lower, gmp
from .geometry import J, TaxelGrid


def taxel_vertices(taxels):
    """Corners of each taxel closure [lo, hi], shape (r, 4, 2) (duplicates on a segment)."""
    lo, hi = taxels.lo, taxels.hi
    return np.stack([lo, np.c_[hi[:, 0], lo[:, 1]], hi, np.c_[lo[:, 0], hi[:, 1]]], axis=1)


# --- M_T: taxel forces and first moment ---------------------------------------------------

def vertex_lp_MT(taxels, F, m1, q, mu=1.0, return_weights=False):
    """Corner LP of thm:combined_upper for one direction q and first moment m1.

    max mu sum_ik lambda_ik ||a + bJ v_ik||  s.t.  sum_k lambda_ik = F_i,  sum_ik lambda_ik v_ik = m1.
    Infeasible (nan) if m1 is outside sum_i F_i closure(P_i). With return_weights=True, also returns the optimal
    weights as an (r, 4) array: a vertex solution has at most r + 2 nonzeros.
    """
    V = taxel_vertices(taxels).reshape(-1, 2)
    rows = sparse.kron(sparse.eye(taxels.r), np.ones((1, 4)))
    A = sparse.vstack([rows, sparse.csr_matrix(V.T)]).tocsr()
    b = np.concatenate([np.asarray(F, float), np.asarray(m1, float)])
    val, lam = gmp(V, q, A, b, mu, "max", return_measure=True)
    if return_weights:
        return val, (None if lam is None else lam.reshape(taxels.r, 4))
    return val


class _TaxelSOCP:
    """Lower-bound SOCP of thm:combined_lower:

        min mu sum_i ||F_i a + bJ u_i||  s.t.  u_i in F_i closure(P_i),  sum_i u_i = m_1,

    in the local first moments u_i (F_i = 0 forces u_i = 0). Built once; (a, b) are cvxpy parameters,
    so solving for many directions only re-solves the same compiled problem.
    """

    def __init__(self, taxels, F, m1, mu=1.0):
        F = np.asarray(F, float)
        self.mu, self.F = mu, F
        self.a, self.b = cp.Parameter(2), cp.Parameter()
        U = cp.Variable((taxels.r, 2))
        Fc = F[:, None]
        V = Fc @ cp.reshape(self.a, (1, 2), order="C") + self.b * (U @ J.T)
        cons = [U >= Fc * taxels.lo, U <= Fc * taxels.hi, cp.sum(U, axis=0) == np.asarray(m1, float)]
        self.U = U
        self.prob = cp.Problem(cp.Minimize(cp.sum(cp.norm(V, 2, axis=1))), cons)

    def __call__(self, q):
        self.a.value, self.b.value = np.asarray(q[:2], float), float(q[2])
        self.prob.solve(solver=cp.CLARABEL)
        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            return np.nan
        return self.mu * self.prob.value

    def points(self):
        """Local CoPs c_i = u_i / F_i of the minimizing point loads (nan for F_i = 0)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.F[:, None] > 0, self.U.value / self.F[:, None], np.nan)


def socp_lower_MT(taxels, F, m1, q, mu=1.0, return_points=False):
    """thm:combined_lower: SOCP lower bound given the taxel forces F and the first moment m1, for q (3,) or (m, 3).

    With return_points=True (single q), also returns the optimal point loads c_i, shape (r, 2): the minimizing
    measure is sum_i F_i delta_{c_i} (lem:taxel_reduction).
    """
    solve = _TaxelSOCP(taxels, F, m1, mu)
    q = np.asarray(q, float)
    if return_points:
        if q.ndim != 1:
            raise ValueError("return_points=True needs a single direction q")
        return solve(q), solve.points()
    return solve(q) if q.ndim == 1 else np.array([solve(qi) for qi in q])


def reduced_MT(taxels, yT, q, mu=1.0):
    """(lower, upper) from thm:combined_lower and thm:combined_upper, given yT = (f_N, m_1, F)."""
    _, m1, F = yT
    lower = _TaxelSOCP(taxels, F, m1, mu)
    return _batch(lambda qi: (lower(qi), vertex_lp_MT(taxels, F, m1, qi, mu)), q)


# --- M_0: total force and first moment -----------------------------------------------------

def reduced_M0(patch, y0, q, mu=1.0):
    """(lower, upper): mu ||f_N a + bJ m_1|| and the patch-vertex LP (the single-taxel corner LP)."""
    fN, m1 = y0
    single = TaxelGrid(patch, 1, 1)
    return _batch(lambda qi: (coarse_lower(y0, qi, mu), vertex_lp_MT(single, np.array([fN]), m1, qi, mu)), q)


def coarse_upper_triangles(patch, y0, q, mu=1.0):
    """Coarse upper bound as the maximum over the vertex triangles of the patch that contain the CoP
    (rem:coarse_triangles), with homogeneous barycentric weights. Vectorized over q.
    """
    fN, m1 = y0
    q = np.atleast_2d(np.asarray(q, float))
    V = taxel_vertices(TaxelGrid(patch, 1, 1))[0]
    best = np.full(len(q), -np.inf)
    for tri in itertools.combinations(range(4), 3):
        M = np.vstack([np.ones(3), V[list(tri)].T])  # [1; v] lambda = (f_N, m_1)
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        lam = np.linalg.solve(M, np.r_[fN, m1])
        if np.any(lam < -1e-12 * max(fN, 1.0)):
            continue  # this triangle does not contain the CoP
        g = np.linalg.norm(q[:, None, :2] + q[:, None, 2:3] * (V[list(tri)] @ J.T)[None], axis=2)
        best = np.maximum(best, mu * g @ lam)
    if fN == 0:
        best[:] = 0.0
    return best[0] if len(best) == 1 else best


# --- M_F: taxel forces only -----------------------------------------------------------------

def reduced_MF(taxels, F, q, mu=1.0):
    """(lower, upper) = mu sum_i F_i (min, max) of g over the taxel closures (thm:MF): exact closed forms."""
    F = np.asarray(F, float)

    def one(qi):
        """(lower, upper) for one direction qi."""
        gmin, gmax = cellwise_g(taxels, qi)
        return mu * F @ gmin, mu * F @ gmax

    return _batch(one, q)
