"""Certified load factors: how much of a demanded tangential wrench a contact is guaranteed to carry.

A load w = (f_1, f_2, tau) is a tangential force and a torque about a fixed reference point o, in the
frame of the pad (or a common grasp frame). For a measure nu, the load factor is
    s*(nu) = max {s >= 0 : s w in W(nu)} = min over q with <q, w> = 1 of h(q),
with the support h(q) = mu int ||a + bJ(x - o)|| dnu taken about o (Prop 3.2 with reference o).
The load w is carried by every compatible measure iff s*_lo >= 1, where s*_lo = min over nu of s*(nu).

Computing s*_lo exactly. Collapsing the force of each taxel onto its local CoP lowers h(q) for every q
at once (Lemma 5.5), so the minimum is attained by measures sum_i F_i delta_{c_i} with c_i in the
closure of P_i (and sum_i F_i c_i = f_N c if the CoP c is measured). Hence
    s*_lo = min  mu sum_i F_i ||a + bJ(c_i - o)||   s.t.  <q, w> = 1,  c_i in closure(P_i)  [, CoP].
This is bilinear in (b, c_i). With u_i = b c_i, the objective is affine inside the norms, and
u_i in b closure(P_i) is a convex cone for b >= 0 and for b <= 0: s*_lo is the smaller of two SOCPs.
The certificate is tight: at the optimum, the point loads F_i delta_{u_i / b} are a compatible measure
whose exact load factor is s*_lo (return_worst=True).

    load_factor_lower   certified s*_lo for one or several pads (M_T if the CoP is given, M_F otherwise;
                        a single taxel covering the pad gives M_0, or an unsensed pad with unknown CoP)
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
    """V_sigma of Prop. prop:load_factor(ii): the certified load factor restricted to sign * b >= 0.

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
    into wrenches admissible for each pad, i.e. w in W(nu_1) + W(nu_2) + ... (Minkowski sum) for every pair
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


def load_factor_true(points, forces, w, mu=1.0, o=(0.0, 0.0)):
    """Exact s*(nu) = max {s : s w in W(nu)} for nu = sum_k forces[k] delta_{points[k]}, w about o.

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
