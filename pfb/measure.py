"""Normal force measures on the support grid, and the support function of their admissible wrench set.

Wrenches (f_T, tau) and twists q = (a, b) are written about a fixed reference point o (default: the origin of
the patch coordinates). The admissible set X_o(nu) has support h(q) = mu int ||a + bJ(x - o)|| dnu
("support function of the admissible tangential wrench set"). Passing o = nu.cop gives the set about the CoP.
"""

import numpy as np

from .geometry import J


class Measure:
    """Finite nonnegative measure nu_N = sum_k w_k delta_{x_k} on the patch support grid."""

    def __init__(self, patch, w):
        w = np.asarray(w, dtype=float)
        if w.shape != (patch.K,):
            raise ValueError(f"expected {patch.K} weights, got shape {w.shape}")
        if (w < 0).any():
            raise ValueError("a normal force measure must be nonnegative")
        self.patch, self.w = patch, w

    @property
    def fN(self):
        """Zeroth moment m_0 = nu(P), the resultant normal force."""
        return float(self.w.sum())

    @property
    def m1(self):
        """First moment m_1 = int x dnu (homogeneous coordinates of the CoP: m_1 = f_N c)."""
        return self.w @ self.patch.points

    @property
    def cop(self):
        """Center of pressure c[nu] = m_1 / m_0 (requires m_0 > 0)."""
        return self.m1 / self.fN

    @property
    def atoms(self):
        """Points carrying nonzero weight (relative threshold 1e-9)."""
        return self.patch.points[self.w > 1e-9 * self.fN]

    def __add__(self, other):
        return Measure(self.patch, self.w + other.w)

    def __mul__(self, s):
        return Measure(self.patch, s * self.w)

    __rmul__ = __mul__

    def __repr__(self):
        return f"Measure(fN={self.fN:.4g}, cop={np.round(self.cop, 4)}, atoms={len(self.atoms)})"


# --- constructors -------------------------------------------------------------------------

def from_pressure(patch, rho, fN=None):
    """Measure with density rho(x, y) w.r.t. area (length on a segment).

    rho is a vectorized callable. If fN is given, the result is rescaled to total force fN.
    """
    x, y = patch.points.T
    w = np.clip(rho(x, y), 0.0, None) * patch.quad # clip to 0 (\rho_N positive) and multiply by dA or dL (area / length element)
    m = Measure(patch, w)
    return m if fN is None else m * (fN / m.fN)


def from_diracs(patch, points, forces):
    """sum_j forces[j] delta_{points[j]}; each point must coincide with a support-grid node."""
    w = np.zeros(patch.K)
    for p, f in zip(np.atleast_2d(points), np.atleast_1d(forces)):
        d = np.linalg.norm(patch.points - p, axis=1)
        k = int(d.argmin())
        if d[k] > 1e-9 * patch.diam:
            raise ValueError(f"point {p} is not a grid node (nearest {patch.points[k]})")
        w[k] += f
    return Measure(patch, w)


def uniform(patch, fN=1.0):
    """Uniform pressure over the whole patch, total force fN."""
    return from_pressure(patch, lambda x, y: np.ones_like(x), fN)


def gaussian(patch, center, sigma, fN=1.0):
    """Gaussian pressure of width sigma centred at center (clipped to the patch), total force fN."""
    cx, cy = center
    return from_pressure(patch, lambda x, y: np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2)), fN)


def random_measure(patch, rng, fN=1.0, n_atoms=None):
    """Random measure: exponential weights on all nodes, or on n_atoms random nodes."""
    w = rng.exponential(size=patch.K)
    if n_atoms is not None:
        mask = np.zeros(patch.K, bool)
        mask[rng.choice(patch.K, n_atoms, replace=False)] = True
        w *= mask
    return Measure(patch, w * (fN / w.sum()))


# --- support function -----------------------------------------------------------------------

def _ref(o):
    """Reference point as an array (default: the origin)."""
    return np.zeros(2) if o is None else np.asarray(o, dtype=float)


def integrand(points, q, o=None):
    """g(x; q) = ||a + b J (x - o)||_2 for q = (a1, a2, b): the speed of the point x under the twist q about o.

    points: (K, 2); q: (3,) or (m, 3). Returns (K,) or (m, K).
    """
    q = np.asarray(q, dtype=float)
    a, b = q[..., :2], q[..., 2]
    v = a[..., None, :] + b[..., None, None] * ((points - _ref(o)) @ J.T)
    return np.linalg.norm(v, axis=-1)


def support(measure, q, mu=1.0, o=None):
    """h_{X_o(nu)}(q) = mu int ||a + bJ(x - o)|| dnu, about the fixed point o (default origin). Vectorized over q.

    o = measure.cop gives the support of the set of wrenches about the CoP of this measure.
    """
    return mu * integrand(measure.patch.points, q, o) @ measure.w


def gamma_star(measure, q, mu=1.0, o=None):
    """Friction ratio field attaining the support in direction q (twist about o), shape (K, 2)."""
    a, b = np.asarray(q[:2], float), float(q[2])
    v = a + b * (measure.patch.points - _ref(o)) @ J.T
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(n > 0, mu * v / np.where(n > 0, n, 1.0), 0.0)


def wrench(measure, gamma, o=None):
    """Resultant (f_T, tau) about o of a friction ratio field gamma (K, 2)."""
    d = measure.patch.points - _ref(o)
    fT = measure.w @ gamma
    tau = measure.w @ (d[:, 0] * gamma[:, 1] - d[:, 1] * gamma[:, 0])
    return fT, float(tau)


def support_points(measure, q, mu=1.0, o=None):
    """Points of X_o(nu) attaining the support in each direction q: the wrench of gamma_star.

    q: (3,) or (m, 3). Returns (3,) or (m, 3) wrenches (f_T1, f_T2, tau) about o. Sampling many
    directions gives points on the boundary of X_o(nu), whose convex hull approximates it.
    """
    q = np.asarray(q, dtype=float)
    out = []
    for qi in np.atleast_2d(q):
        fT, tau = wrench(measure, gamma_star(measure, qi, mu, o), o)
        out.append([fT[0], fT[1], tau])
    out = np.array(out)
    return out[0] if q.ndim == 1 else out
