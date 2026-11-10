"""Certified and upper load factors: how much of a demanded tangential wrench a contact surely carries, or surely cannot.

A load w = (f_1, f_2, tau) is a tangential force and a torque about a fixed reference point o, in the
frame of the pad (or a common grasp frame). For a measure nu, the load factor is
    s*(nu) = max {s >= 0 : s w in X_o(nu)} = min over q with <q, w> = 1 of h(q),
with the support h(q) = mu int ||a + bJ(x - o)|| dnu of the admissible wrench set about o.
Given the measurement (taxel forces, optionally the CoP), two numbers bracket s* for every compatible measure:

    s_lo = min over compatible nu of s*(nu)  (certified load factor; prop:load_factor, prop:load_factor_lower)
        Concentrating the force of each taxel at its local CoP shrinks the wrench set, so the minimum is attained by
        point loads sum_i F_i delta_{c_i}, c_i in the closure of P_i (and sum_i F_i c_i = f_N c if the CoP c is
        measured). The program is bilinear in (b, c_i); with u_i = b c_i it is the smaller of two SOCPs.
        Tight: the point loads F_i delta_{u_i / b} are a compatible measure with load factor s_lo (return_worst=True).
    s_up = max {s : s w in closure of the possible set X^+}  (upper load factor; prop:load_factor, prop:load_factor_upper)
        X^+ is convex (fixed reference point), with support h_up(q), so s_up = min over <q, w> = 1 of h_up(q),
        an SOCP: for taxel forces alone h_up = mu sum_i F_i max_corners ||a + bJ(v - o)||, and with the CoP the
        corner LP of h_up is dualized inside the minimization. Spreading every taxel force to its corners enlarges
        the wrench set, so s_up is also the load factor of a most favourable corner-supported compatible measure
        (return_best=True, primal SOCP).

    s_lo >= 1: every compatible pressure carries w.   s_up < 1: none does (slip is certain).   Otherwise undecided.

    load_factor_lower   certified s_lo for one or several pads (M_T if the CoP is given, M_F otherwise;
                        a single taxel covering the pad gives M_0, or an unsensed pad with unknown CoP)
    load_factor_upper   upper s_up, same inputs
    classify            "hold", "slip" or "undecided", with (s_lo, s_up)
    load_factor_true    exact s*(nu) for a known discrete measure (the truth, or a plug-in estimate)
"""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .geometry import J


@dataclass
class PadTerm:
    """One pad in the certificate: rectangular taxels [lo, hi] in pad coordinates, forces F, optional CoP.

    R (2x2, orthogonal, a reflection is allowed) and t map pad coordinates to the common frame:
    x_common = R x_pad + t. Leave the defaults when the load is expressed in pad coordinates.
    """
    lo: np.ndarray
    hi: np.ndarray
    F: np.ndarray
    cop: np.ndarray | None = None
    R: np.ndarray = None
    t: np.ndarray = None

    @classmethod
    def from_taxels(cls, taxels, F, cop=None, R=None, t=None):
        """PadTerm of a pfb.TaxelGrid (its taxel rectangles lo, hi) with taxel forces F and an optional CoP."""
        return cls(taxels.lo, taxels.hi, np.asarray(F, float), None if cop is None else np.asarray(cop, float), R, t)


def _socp(terms, w, mu, o, sign):
    """V_sigma of prop:load_factor_lower: the certified load factor restricted to sign * b >= 0.

    Variables q = (a, b) with <q, w> = 1 and u_i = b c_i per taxel (perspective substitution). For
    sign * b >= 0, c_i in [lo_i, hi_i] becomes the linear cone constraints sign (u_i - b lo_i) >= 0 and
    sign (b hi_i - u_i) >= 0, and the CoP constraint sum_i F_i c_i = f_N c becomes F @ U = f_N b c,
    so the objective mu sum_i F_i ||a + J(R u_i + b (t - o))|| is a sum of norms of affine maps: an SOCP.
    Solved with Clarabel (interior point, handles SOC constraints natively). An "optimal_inaccurate"
    status is accepted, so the returned value is then only as accurate as the solver tolerance; the
    certificate is sound up to that tolerance.
    Returns (value, q, [U per pad]), or (inf, None, None) if this half is infeasible.
    """
    a, b = cp.Variable(2), cp.Variable()
    cons = [w[:2] @ a + w[2] * b == 1, sign * b >= 0]
    obj, Us = 0, []
    for T in terms:
        R = np.eye(2) if T.R is None else np.asarray(T.R, float)
        t = np.zeros(2) if T.t is None else np.asarray(T.t, float)
        r = len(T.F)
        U = cp.Variable((r, 2))  # u_i = b c_i, pad coordinates
        cons += [sign * (U - b * T.lo) >= 0, sign * (b * T.hi - U) >= 0]
        if T.cop is not None:
            cons.append(T.F @ U == T.F.sum() * b * T.cop)
        ones = np.ones((r, 1))
        bx = U @ R.T + ones @ cp.reshape(b * (t - o), (1, 2), order="C")  # b (x_i - o), common frame
        V = ones @ cp.reshape(a, (1, 2), order="C") + bx @ J.T
        obj = obj + T.F @ cp.norm(V, 2, axis=1)
        Us.append(U)
    prob = cp.Problem(cp.Minimize(mu * obj), cons)
    prob.solve(solver=cp.CLARABEL)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return np.inf, None, None
    return prob.value, np.r_[a.value, b.value], [U.value for U in Us]


def load_factor_lower(pads, w, mu=1.0, o=(0.0, 0.0), return_q=False, return_worst=False):
    """Certified load factor s*_lo of the load w = (f_1, f_2, tau) about o, for one pad or a list of PadTerm.

    Several pads (independent measurements, patches in a common plane): the load is carried if it splits
    into wrenches admissible for each pad, i.e. w in X_o(nu_1) + X_o(nu_2) + ... (Minkowski sum) for every pair
    of compatible measures. Its support is h_1 + h_2 + ..., so s*_lo = min over <q, w> = 1 of the sum of the
    per-pad lower bounds, one program with a shared q. This intersection of sums contains the sum of the
    per-pad guaranteed sets W_1^- + W_2^-, so it is never smaller than certifying each pad separately.
    s*_lo >= 1 certifies that every compatible pressure carries w.
    return_q: also return the binding direction q = (a, b) (normalized by <q, w> = 1).
    return_worst: also return a worst-case compatible measure, the point loads c_i (one array (r, 2) per
    pad, common frame) with forces F_i, whose exact load factor is s*_lo; None if b = 0 at the optimum
    (then any c_i in the closures, with the CoP, is a worst case).
    """
    pads = [pads] if isinstance(pads, PadTerm) else list(pads)
    w, o = np.asarray(w, float), np.asarray(o, float)
    if not np.any(w):
        raise ValueError("the load w must be nonzero")
    s, q, Us = min((_socp(pads, w, mu, o, sign) for sign in (1.0, -1.0)), key=lambda r: r[0])
    out = (s,) + ((q,) if return_q else ())
    if return_worst:
        worst = None
        if q is not None and abs(q[2]) > 1e-9:
            worst = [(U / q[2]) @ (np.eye(2) if T.R is None else np.asarray(T.R, float)).T
                     + (0 if T.t is None else np.asarray(T.t, float)) for T, U in zip(pads, Us)]
        out += (worst,)
    return out if len(out) > 1 else s


def _frame(T):
    """Rotation/reflection R and translation t of a pad (identity by default)."""
    R = np.eye(2) if T.R is None else np.asarray(T.R, float)
    t = np.zeros(2) if T.t is None else np.asarray(T.t, float)
    return R, t


def _upper_dual(terms, w, mu, o):
    """s_up = min over <q, w> = 1 of sum_pads h_up(q), one SOCP with a shared q = (a, b).

    Pad with taxel forces only:  h_up(q) = mu sum_i F_i t_i,  t_i >= ||a + bJ(v_ik - o)|| for the 4 corners v_ik.
    Pad with CoP c:  h_up(q) = min_{y, z} sum_i F_i y_i + z^T m_1  s.t.  y_i + z^T v_ik >= mu ||a + bJ(v_ik - o)||
    (LP duality of the corner LP), with m_1 = f_N c. Corners and m_1 are taken in the common frame.
    """
    a, b = cp.Variable(2), cp.Variable()
    cons = [w[:2] @ a + w[2] * b == 1]
    obj = 0
    for T in terms:
        R, t = _frame(T)
        lo, hi = np.asarray(T.lo, float), np.asarray(T.hi, float)
        V = np.stack([lo, np.c_[hi[:, 0], lo[:, 1]], hi, np.c_[lo[:, 0], hi[:, 1]]], axis=1)  # (r, 4, 2), pad coords
        Vc = V @ R.T + t - o  # corners relative to o, common frame
        r = len(T.F)
        A = cp.reshape(a, (2, 1), order="C") @ np.ones((1, r))
        G = [cp.norm(A + b * (J @ Vc[:, k].T), 2, axis=0) for k in range(4)]  # (r,) each: ||a + bJ(v_ik - o)||
        if T.cop is None:
            tt = cp.Variable(r)
            cons += [tt >= g for g in G]
            obj = obj + mu * (T.F @ tt)
        else:
            y, z = cp.Variable(r), cp.Variable(2)
            m1 = T.F.sum() * (R @ np.asarray(T.cop, float) + t - o)  # first moment about o, common frame
            cons += [y + Vc[:, k] @ z >= mu * G[k] for k in range(4)]
            obj = obj + T.F @ y + z @ m1
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.CLARABEL)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return np.nan, None
    return prob.value, np.r_[a.value, b.value]


def _upper_primal(terms, w, mu, o):
    """s_up as the primal SOCP over corner-supported compatible measures and their corner friction forces.

    max s  s.t.  lambda_ik >= 0, sum_k lambda_ik = F_i [, sum_ik lambda_ik v_ik = f_N c],
                 ||rho_ik|| <= mu lambda_ik,  sum rho = s f,  sum (v_ik - o) x rho_ik = s tau.
    Returns (s, [lambda (r, 4) per pad], [corners (r, 4, 2) per pad, common frame]).
    """
    s = cp.Variable()
    cons, force, torque, lams, corners = [], 0, 0, [], []
    for T in terms:
        R, t = _frame(T)
        lo, hi = np.asarray(T.lo, float), np.asarray(T.hi, float)
        V = np.stack([lo, np.c_[hi[:, 0], lo[:, 1]], hi, np.c_[lo[:, 0], hi[:, 1]]], axis=1)
        Vc = V @ R.T + t
        r = len(T.F)
        lam = cp.Variable((r, 4), nonneg=True)
        rho = [cp.Variable((r, 2)) for _ in range(4)]
        cons.append(cp.sum(lam, axis=1) == T.F)
        if T.cop is not None:
            m1 = T.F.sum() * (R @ np.asarray(T.cop, float) + t)
            cons.append(sum(Vc[:, k].T @ lam[:, k] for k in range(4)) == m1)
        for k in range(4):
            cons.append(cp.norm(rho[k], 2, axis=1) <= mu * lam[:, k])
            d = Vc[:, k] - o
            force = force + cp.sum(rho[k], axis=0)
            torque = torque + cp.sum(cp.multiply(d[:, 0], rho[k][:, 1]) - cp.multiply(d[:, 1], rho[k][:, 0]))
        lams.append(lam)
        corners.append(Vc)
    cons += [force == s * w[:2], torque == s * w[2]]
    prob = cp.Problem(cp.Maximize(s), cons)
    prob.solve(solver=cp.CLARABEL)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return np.nan, None, None
    return s.value, [l.value for l in lams], corners


def load_factor_upper(pads, w, mu=1.0, o=(0.0, 0.0), return_q=False, return_best=False):
    """Upper load factor s_up of the load w = (f_1, f_2, tau) about o, for one pad or a list of PadTerm.

    No compatible pressure carries more than s_up w: if s_up < 1, the contact slips whatever the actual
    distribution (quasi-statically). Several pads: s_up = min over <q, w> = 1 of the sum of the per-pad upper
    supports (the Minkowski sum of convex possible sets is convex).
    return_q: also return the binding direction q (from the dual SOCP).
    return_best: also return a most favourable compatible measure, as (weights (r, 4), corners (r, 4, 2)) per pad
    in the common frame, from the primal SOCP (its load factor is s_up).
    """
    pads = [pads] if isinstance(pads, PadTerm) else list(pads)
    w, o = np.asarray(w, float), np.asarray(o, float)
    if not np.any(w):
        raise ValueError("the load w must be nonzero")
    s, q = _upper_dual(pads, w, mu, o)
    out = (s,) + ((q,) if return_q else ())
    if return_best:
        _, lams, corners = _upper_primal(pads, w, mu, o)
        out += (None if lams is None else list(zip(lams, corners)),)
    return out if len(out) > 1 else s


def classify(pads, w, mu=1.0, o=(0.0, 0.0), tol=1e-6):
    """Decision rule (cor:load_factor_decision): ("hold" | "slip" | "undecided", s_lo, s_up) for the load w about o.

    "hold": every compatible pressure carries w (s_lo >= 1). "slip": none does (s_up < 1). The tolerance is
    applied in the conservative direction for "hold" and "slip".
    """
    s_lo = load_factor_lower(pads, w, mu, o)
    s_up = load_factor_upper(pads, w, mu, o)
    verdict = "hold" if s_lo >= 1 + tol else "slip" if s_up < 1 - tol else "undecided"
    return verdict, s_lo, s_up


def load_factor_true(points, forces, w, mu=1.0, o=(0.0, 0.0)):
    """Exact s*(nu) = max {s : s w in X_o(nu)} for nu = sum_k forces[k] delta_{points[k]}, w about o.

    Solved in the friction field: gamma_k with ||gamma_k|| <= mu, sum_k F_k gamma_k = s f and
    sum_k F_k (x_k - o) x gamma_k = s tau. Use it for the truth or for a point estimate (plug-in).
    """
    x, N, w = np.asarray(points, float), np.asarray(forces, float), np.asarray(w, float)
    keep = N > 0
    x, N = x[keep], N[keep]
    d = x - np.asarray(o, float)
    G, s = cp.Variable((len(N), 2)), cp.Variable()
    torque = N @ (cp.multiply(d[:, 0], G[:, 1]) - cp.multiply(d[:, 1], G[:, 0]))
    prob = cp.Problem(cp.Maximize(s), [cp.norm(G, 2, axis=1) <= mu, N @ G == s * w[:2], torque == s * w[2]])
    prob.solve(solver=cp.CLARABEL)
    return s.value if prob.status in ("optimal", "optimal_inaccurate") else np.nan
