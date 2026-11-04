"""Statically determinate boss forces from the ATI wrench (H2), and image <-> surface registration.

Frame: the gel-surface frame `gelsight_contact_surface`, wrench taken about its origin, normal = y,
surface plane = (x, z). Boss i carries a normal load F_i at (x_i, y_c, z_i), and the bosses carry
tangential loads summing to (F_x, F_z) applied in the same plane y = y_c (flat-topped bosses
indent equally; y_c = -indentation depth, plus any error in the assumed sensor stack height).
With r x f for r = (x, y_c, z):

    F_y = sum F_i
    tau_x = -sum z_i F_i + y_c F_z
    tau_z =  sum x_i F_i - y_c F_x
    tau_y = sum (z_i f_x,i - x_i f_z,i)       torsion: tangential loads only, not used here

So the normal loads follow from (F_y, tau_x, tau_z) - not from tau_y - and the tangential totals
enter only through the plane offset y_c, independently of how they are distributed over the bosses:

    A F = b,   A = [[1 ... 1], [-z_1 ... -z_n], [x_1 ... x_n]],   b = (F_y, tau_x - y_c F_z, tau_z + y_c F_x)

1 boss: F = F_y and (tau_z, -tau_x) / F_y is its position (a check, and the registration data).
2 bosses: 3 equations, 2 unknowns; the residual is a consistency check.
3 bosses: square; full rank iff not collinear (det A = 2 x signed triangle area). No redundancy left.
The moment rows are scaled by 1/d, d the RMS boss distance from the boss centroid, so the
condition number and the residual are in force units.

All forces keep the sign of F_y (compression is negative in the legacy convention f_y_des = -preload).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class BossSolution:
    F: np.ndarray            # (n,) boss normal loads, sign of F_y [N]
    residual: float          # norm of the scaled residual [N]; 0 for a square full-rank system
    rank: int
    cond: float              # condition number of the scaled matrix
    scale: float             # d [m]
    cov: np.ndarray | None   # (n, n) covariance of F if noise levels were given and the system is full column rank


def system(positions, wrench, y_c=0.0):
    """A (3, n) and b (3,) for boss positions (n, 2) = (x_i, z_i) and the surface wrench (6,)."""
    P = np.atleast_2d(np.asarray(positions, float))
    fx, fy, fz, tx, _, tz = np.asarray(wrench, float)
    A = np.vstack([np.ones(len(P)), -P[:, 1], P[:, 0]])
    b = np.array([fy, tx - y_c * fz, tz + y_c * fx])
    return A, b


def characteristic_length(positions):
    P = np.atleast_2d(np.asarray(positions, float))
    d = np.sqrt(np.mean(np.sum((P - P.mean(axis=0)) ** 2, axis=1)))
    return d if d > 0 else 1.0


def solve(positions, wrench, y_c=0.0, sigma_force=None, sigma_torque=None, rcond=1e-9):
    """Boss normal loads from one surface wrench (6,). See the module docstring for the equations.

    sigma_force [N], sigma_torque [N m]: measurement noise of F_y and of tau_x, tau_z, used for the covariance.
    """
    A, b = system(positions, wrench, y_c)
    d = characteristic_length(positions)
    S = np.diag([1.0, 1.0 / d, 1.0 / d])
    As, bs = S @ A, S @ b
    F, _, rank, sv = np.linalg.lstsq(As, bs, rcond=rcond)
    residual = float(np.linalg.norm(As @ F - bs))
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 and len(sv) == A.shape[1] else np.inf
    cov = None
    if sigma_force is not None and sigma_torque is not None and rank == A.shape[1]:
        Winv = np.diag([sigma_force ** 2, sigma_torque ** 2, sigma_torque ** 2])
        AtWA = A.T @ np.linalg.inv(Winv) @ A
        cov = np.linalg.inv(AtWA)
    return BossSolution(F, residual, int(rank), cond, d, cov)


def cop(wrench, y_c=0.0):
    """Centre of pressure (x, z) of the normal load in the surface plane, from the surface wrench (6,) or (N, 6)."""
    W = np.asarray(wrench, float)
    fx, fy, fz, tx, tz = W[..., 0], W[..., 1], W[..., 2], W[..., 3], W[..., 5]
    return np.stack([(tz + y_c * fx) / fy, -(tx - y_c * fz) / fy], axis=-1)


def fit_registration(pixels, points):
    """Affine map from image pixels (u, v) to surface coordinates (x, z): points ~ M @ [u, v, 1].

    Fitted by least squares on single-boss presses (pixel centroid vs ATI CoP). Returns (M (2, 3), rms [m]).
    """
    U = np.asarray(pixels, float)
    X = np.asarray(points, float)
    H = np.column_stack([U, np.ones(len(U))])
    M, *_ = np.linalg.lstsq(H, X, rcond=None)
    rms = float(np.sqrt(np.mean(np.sum((H @ M - X) ** 2, axis=1))))
    return M.T, rms


def apply_registration(M, pixels):
    U = np.atleast_2d(np.asarray(pixels, float))
    return U @ M[:, :2].T + M[:, 2]
