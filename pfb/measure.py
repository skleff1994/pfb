"""Normal force measures on the support grid, and the support function of W(nu) (Prop. 3.2)."""

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
        """Zeroth moment m_0 = nu(P), eq. (8)."""
        return float(self.w.sum())

    @property
    def m1(self):
        """First moment m_1 = int x dnu, eq. (9)."""
        return self.w @ self.patch.points

    @property
    def cop(self):
        """Center of pressure c[nu] = m_1 / m_0, eq. (10)."""
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
    """Measure with density rho(x, y) w.r.t. area (length on a segment), eq. (4).

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

def integrand(points, q, c):
    """g(x; q, c) = ||a + b J (x - c)||_2 for q = (a1, a2, b).

    points: (K, 2); q: (3,) or (m, 3). Returns (K,) or (m, K).
    """
    q = np.asarray(q, dtype=float)
    a, b = q[..., :2], q[..., 2]
    v = a[..., None, :] + b[..., None, None] * ((points - c) @ J.T)
    return np.linalg.norm(v, axis=-1)


def support(measure, q, mu=1.0, c=None):
    """h_{W(nu)}(q) = mu int ||a + bJ(x - c[nu])|| dnu, eq. (19). Vectorized over q."""
    c = measure.cop if c is None else c
    return mu * integrand(measure.patch.points, q, c) @ measure.w


def gamma_star(measure, q, mu=1.0, c=None):
    """Optimal friction ratio field of eq. (23), shape (K, 2)."""
    c = measure.cop if c is None else c
    a, b = np.asarray(q[:2], float), float(q[2])
    v = a + b * (measure.patch.points - c) @ J.T
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(n > 0, mu * v / np.where(n > 0, n, 1.0), 0.0)


def wrench(measure, gamma, c=None):
    """Resultant (f_T, tau_N) of eq. (16) for a friction ratio field gamma (K, 2)."""
    c = measure.cop if c is None else c
    d = measure.patch.points - c
    fT = measure.w @ gamma
    tau = measure.w @ (d[:, 0] * gamma[:, 1] - d[:, 1] * gamma[:, 0])
    return fT, float(tau)


def support_points(measure, q, mu=1.0, c=None):
    """Points of W(nu) attaining the support in each direction q: the wrench of gamma_star.

    q: (3,) or (m, 3). Returns (3,) or (m, 3) wrenches (f_T1, f_T2, tau_N). Sampling many
    directions gives points on the boundary of W(nu), whose convex hull approximates it.
    """
    q = np.asarray(q, dtype=float)
    out = []
    for qi in np.atleast_2d(q):
        fT, tau = wrench(measure, gamma_star(measure, qi, mu, c), c)
        out.append([fT[0], fT[1], tau])
    out = np.array(out)
    return out[0] if q.ndim == 1 else out
