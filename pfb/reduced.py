"""Finite-dimensional reductions of the bounds on the continuous patch (Sections 4, 6, 7).

Unlike `bounds.py`, which solves the GMPs as LPs on a support grid, these functions implement
the formulas of the draft directly on the taxel geometry (rectangular taxel closures):

    reduced_M0        Thm 4.3 (closed form) and Cor 4.6 (vertex LP), as the r = 1 case of M_T (Rem 6.3)
    reduced_MT        Thm 6.1 (SOCP lower bound) and Thm 6.2 (vertex LP upper bound)
    reduced_MF        Thm 7.2 (joint SOCP lower bound) and Thm 7.3 (max over c in C_MF of the M_T LP)
    certified_ref_MF  Prop 7.4 (fixed-reference bounds with the CoP-uncertainty correction)

The values are exact for the taxel model of Definition 5.2 (a load on an edge shared by two taxels may be counted by
either). The grid LPs use strict half-open cells. The closure model contains the strict one, so the reduced bounds are
conservative for it (Remark 5.9); for rectangular taxels the two gave the same values in all tests
(`test_rem_5_9_strict_model_matches_closure`).
Since the grid bounds are inner approximations, for every q
    reduced lower <= grid lower <= h_true <= grid upper <= reduced upper.
"""

import cvxpy as cp
import numpy as np
from scipy import sparse

from .bounds import _batch, cellwise_g, cop_rows, cop_set_MF, gmp, outer_search
from .geometry import J, TaxelGrid


def taxel_vertices(taxels):
    """Corners of each taxel closure [lo, hi], shape (r, 4, 2) (duplicates on a segment)."""
    lo, hi = taxels.lo, taxels.hi
    return np.stack([lo, np.c_[hi[:, 0], lo[:, 1]], hi, np.c_[lo[:, 0], hi[:, 1]]], axis=1)


# --- M_T: taxel forces and CoP (Section 6) -----------------------------------------------

def vertex_lp_MT(taxels, F, c, q, mu=1.0, return_weights=False):
    """Vertex LP of Thm 6.2 for one direction q and reference CoP c.

    max mu sum_ik lambda_ik ||a + bJ(v_ik - c)||  s.t.  sum_k lambda_ik = F_i,  sum_ik lambda_ik (v_ik - c) = 0.
    Infeasible (nan) if c is outside the closure of C_MF. With return_weights=True, also
    returns the optimal weights as an (r, 4) array: a vertex solution has at most r + 2 nonzeros.
    """
    V = taxel_vertices(taxels).reshape(-1, 2)
    rows = sparse.kron(sparse.eye(taxels.r), np.ones((1, 4)))
    A = sparse.vstack([rows, sparse.csr_matrix(cop_rows(V, c))]).tocsr()
    b = np.concatenate([F, [0.0, 0.0]])
    val, lam = gmp(V, q, c, A, b, mu, "max", return_measure=True)
    if return_weights:
        return val, (None if lam is None else lam.reshape(taxels.r, 4))
    return val


class _TaxelSOCP:
    """Lower-bound SOCPs of Thm 6.1 (CoP fixed to c) and Thm 7.2 (c = None: CoP free).

        min mu sum_i F_i ||a + bJ(c_i - c)||  s.t.  c_i in closure(P_i)  [and sum_i F_i c_i = f_N c]

    with c = sum_i F_i c_i / f_N when the CoP is free. Built once; (a, b) are cvxpy parameters,
    so solving for many directions only re-solves the same compiled problem.
    """

    def __init__(self, taxels, F, mu=1.0, c=None):
        F = np.asarray(F, float)
        self.mu, self.fN = mu, F.sum()
        self.a, self.b = cp.Parameter(2), cp.Parameter()
        C = cp.Variable((taxels.r, 2))
        ref = (F @ C) / self.fN if c is None else np.asarray(c, float)
        ones = np.ones((taxels.r, 1))
        D = C - ones @ cp.reshape(ref, (1, 2), order="C")  # c_i - c, row-wise
        V = ones @ cp.reshape(self.a, (1, 2), order="C") + self.b * (D @ J.T)
        cons = [C >= taxels.lo, C <= taxels.hi]
        if c is not None:
            cons.append(F @ C == self.fN * ref)
        self.C = C
        self.prob = cp.Problem(cp.Minimize(F @ cp.norm(V, 2, axis=1)), cons)

    def __call__(self, q):
        self.a.value, self.b.value = np.asarray(q[:2], float), float(q[2])
        self.prob.solve(solver=cp.CLARABEL)
        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            return np.nan
        return self.mu * self.prob.value


def _solve_socp(solve, q, return_points):
    """Evaluate a _TaxelSOCP at one direction (3,) or a batch (m, 3); optionally the optimal points c_i."""
    q = np.asarray(q, float)
    if return_points:
        if q.ndim != 1:
            raise ValueError("return_points=True needs a single direction q")
        return solve(q), solve.C.value
    return solve(q) if q.ndim == 1 else np.array([solve(qi) for qi in q])


def socp_lower_MT(taxels, F, c, q, mu=1.0, return_points=False):
    """Thm 6.1: SOCP lower bound with known CoP c, for q of shape (3,) or (m, 3).

    With return_points=True (single q), also returns the optimal points c_i, shape (r, 2): the
    minimizing measure is sum_i F_i delta_{c_i} (Lemma 5.5).
    """
    return _solve_socp(_TaxelSOCP(taxels, F, mu, c), q, return_points)


def socp_lower_MF(taxels, F, q, mu=1.0, return_points=False):
    """Thm 7.2: joint convex lower bound with unknown CoP, for q of shape (3,) or (m, 3).

    With return_points=True (single q), also returns the optimal points c_i, shape (r, 2).
    """
    return _solve_socp(_TaxelSOCP(taxels, F, mu, None), q, return_points)


def reduced_MT(taxels, yT, q, mu=1.0):
    """(lower, upper) from Thms 6.1 and 6.2, given yT = (f_N, c, F)."""
    _, c, F = yT
    lower = _TaxelSOCP(taxels, F, mu, c)
    return _batch(lambda qi: (lower(qi), vertex_lp_MT(taxels, F, c, qi, mu)), q)


# --- M_0: total force and CoP (Section 4) ------------------------------------------------

def reduced_M0(patch, y0, q, mu=1.0):
    """(lower, upper): mu f_N ||a|| (Thm 4.3) and the patch-vertex LP (Cor 4.6 = Thm 6.2 with r = 1)."""
    fN, c = y0
    single = TaxelGrid(patch, 1, 1)
    F = np.array([fN])
    return _batch(lambda qi: (mu * fN * np.linalg.norm(qi[:2]), vertex_lp_MT(single, F, c, qi, mu)), q)


# --- M_F: taxel forces only (Section 7) --------------------------------------------------

def reduced_MF(taxels, F, q, mu=1.0, n_c=7, polish=True):
    """(lower, upper) from Thm 7.2 (global SOCP) and Thm 7.3.

    The upper bound is max over c in the closure of C_MF of the vertex LP (Prop 7.1), computed
    with `outer_search`, so it may sit slightly below the true maximum.
    """
    F = np.asarray(F, float)
    lower = _TaxelSOCP(taxels, F, mu, None)
    lo_c, hi_c = cop_set_MF(taxels, F)
    xatol = 1e-4 * taxels.patch.diam

    def one(qi):
        """(lower, upper) for one direction qi: the global SOCP, and the max over c of the vertex LP."""
        up, _ = outer_search(lambda c: vertex_lp_MT(taxels, F, c, qi, mu), lo_c, hi_c, "max", n_c, polish, xatol)
        return lower(qi), up

    return _batch(one, q)


def certified_ref_MF(taxels, F, q, mu=1.0, c_hat=None):
    """Prop 7.4: fixed-reference cellwise bounds corrected by mu |b| f_N delta(c_hat).

    delta(c_hat) = max distance from c_hat to C_MF (attained at a corner of the rectangle).
    The default c_hat, the centre of C_MF, minimizes delta. The lower bound is also clipped
    below by the universal bound mu f_N ||a|| of eq. (34). This clip is not part of eq. (74), but it is valid
    since every compatible measure satisfies eq. (34) (Prop 8.1).
    """
    F = np.asarray(F, float)
    fN = F.sum()
    lo_c, hi_c = cop_set_MF(taxels, F)
    c_hat = (lo_c + hi_c) / 2 if c_hat is None else np.asarray(c_hat, float)
    delta = np.linalg.norm(np.maximum(np.abs(c_hat - lo_c), np.abs(c_hat - hi_c)))

    def one(qi):
        """(lower, upper) for one direction qi, eq. (74) with the CoP-uncertainty correction."""
        gmin, gmax = cellwise_g(taxels, qi, c_hat)
        corr = mu * abs(qi[2]) * fN * delta
        return max(mu * F @ gmin - corr, mu * fN * np.linalg.norm(qi[:2])), mu * F @ gmax + corr

    return _batch(one, q)
