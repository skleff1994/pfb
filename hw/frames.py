"""Rigid wrench transforms.

A wrench w = (f, tau) is expressed in frame A and taken about A's origin. To express it in frame B
about B's origin, with the pose of A in B given by (R, t) (x_B = R x_A + t):

    f_B   = R f_A
    tau_B = R tau_A + t x f_B           (moment about a new point P: tau_P = tau_O + (O - P) x f)

Moving only the reference point by r (new point at r, same coordinates): tau' = tau - r x f.
"""

import numpy as np


def transform_wrench(W, R, t):
    """Wrenches W (..., 6) expressed in A about A's origin -> in B about B's origin; pose of A in B is (R, t)."""
    W = np.asarray(W, float)
    R, t = np.asarray(R, float), np.asarray(t, float)
    f = W[..., :3] @ R.T
    tau = W[..., 3:] @ R.T + np.cross(t, f)
    return np.concatenate([f, tau], axis=-1)


def shift_reference(W, r):
    """Same coordinates, reference point moved from the origin to r (r may be (..., 3))."""
    W = np.asarray(W, float)
    f = W[..., :3]
    return np.concatenate([f, W[..., 3:] - np.cross(np.asarray(r, float), f)], axis=-1)


def invert(R, t):
    """Inverse pose: if x_B = R x_A + t then x_A = R^T x_B - R^T t."""
    R = np.asarray(R, float)
    return R.T, -R.T @ np.asarray(t, float)


def quat_to_matrix(q):
    """Rotation matrix of a unit quaternion (x, y, z, w) (ROS order)."""
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
