"""Depth -> pressure maps fitted on boss forces (hardware protocol, step H2).

Each sample is one frame at the end of a dwell: the depth map, one integration region per boss
(its connected contact component) and the boss normal loads from `boss_solve` (compression positive).
Pressure is integrated over each region and compared with the boss load:

    winkler   p = k d                         linear in k
    power     p = kappa d^alpha               kappa in closed form, alpha by 1-D search
    kernel    p = (K * d), K radial rings     linear in the ring weights (non-local, ridge-regularized)

Units: depth [mm], area [mm^2], force [N], so k is in N/mm^3 and p in N/mm^2 (MPa).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, optimize


@dataclass
class Sample:
    depth_mm: np.ndarray        # (H, W)
    regions: list               # list of (H, W) bool masks, one per boss
    F: np.ndarray               # (n_boss,) normal loads, compression positive [N]
    mm_per_px: float
    meta: dict = field(default_factory=dict)

    @property
    def px_area(self):
        return self.mm_per_px ** 2


@dataclass
class Fit:
    model: str
    params: dict
    residuals: np.ndarray       # predicted - measured, all bosses of all samples [N]
    rms: float
    rel_rms: float              # rms / rms of the measured loads

    def pressure(self, depth_mm, mask):
        """Pressure map [N/mm^2] inside mask (0 outside)."""
        d = np.where(mask, np.clip(depth_mm, 0.0, None), 0.0)
        if self.model == "winkler":
            p = self.params["k"] * d
        elif self.model == "power":
            p = self.params["kappa"] * d ** self.params["alpha"]
        elif self.model == "kernel":
            p = sum(w * ndimage.convolve(d, K, mode="constant")
                    for w, K in zip(self.params["weights"], ring_kernels(self.params["radius_px"])))
        else:
            raise ValueError(self.model)
        return np.where(mask, p, 0.0)

    def predict(self, sample):
        """Predicted boss loads of a sample [N]."""
        mask = np.any(sample.regions, axis=0)
        p = self.pressure(sample.depth_mm, mask)
        return np.array([p[r].sum() * sample.px_area for r in sample.regions])


def ring_kernels(radius_px):
    """Kernels K_j, j = 0..radius_px: indicator of the pixels at rounded distance j, normalized to unit sum."""
    r = np.arange(-radius_px, radius_px + 1)
    dist = np.rint(np.hypot(*np.meshgrid(r, r, indexing="ij"))).astype(int)
    kernels = []
    for j in range(radius_px + 1):
        K = (dist == j).astype(float)
        kernels.append(K / K.sum())
    return kernels


def _stack(samples):
    return np.concatenate([s.F for s in samples])


def _features(samples, g):
    """Integrated feature per boss: sum over the region of g(depth) * pixel area."""
    out = []
    for s in samples:
        d = np.clip(s.depth_mm, 0.0, None)
        gd = g(d)
        out.extend(gd[r].sum() * s.px_area for r in s.regions)
    return np.asarray(out)


def _result(model, params, samples):
    f = Fit(model, params, np.empty(0), 0.0, 0.0)
    pred = np.concatenate([f.predict(s) for s in samples])
    meas = _stack(samples)
    f.residuals = pred - meas
    f.rms = float(np.sqrt(np.mean(f.residuals ** 2)))
    f.rel_rms = f.rms / float(np.sqrt(np.mean(meas ** 2)))
    return f


def fit_winkler(samples):
    x, y = _features(samples, lambda d: d), _stack(samples)
    return _result("winkler", {"k": float(x @ y / (x @ x))}, samples)


def fit_power(samples, alpha_bounds=(0.5, 3.0)):
    y = _stack(samples)

    def cost(alpha):
        x = _features(samples, lambda d: d ** alpha)
        kappa = x @ y / (x @ x)
        return np.sum((kappa * x - y) ** 2)

    alpha = optimize.minimize_scalar(cost, bounds=alpha_bounds, method="bounded").x
    x = _features(samples, lambda d: d ** alpha)
    return _result("power", {"kappa": float(x @ y / (x @ x)), "alpha": float(alpha)}, samples)


def fit_kernel(samples, radius_px=6, ridge=1e-3):
    """Ring weights w by ridge least squares; pressure is evaluated (convolved) before masking."""
    kernels = ring_kernels(radius_px)
    cols = []
    for K in kernels:
        col = []
        for s in samples:
            mask = np.any(s.regions, axis=0)
            d = np.where(mask, np.clip(s.depth_mm, 0.0, None), 0.0)
            c = ndimage.convolve(d, K, mode="constant")
            col.extend(c[r].sum() * s.px_area for r in s.regions)
        cols.append(col)
    X, y = np.asarray(cols).T, _stack(samples)
    scale = np.linalg.norm(X, axis=0)
    Xs = X / scale
    w = np.linalg.solve(Xs.T @ Xs + ridge * np.eye(len(kernels)), Xs.T @ y) / scale
    return _result("kernel", {"weights": w.tolist(), "radius_px": radius_px}, samples)


FITTERS = {"winkler": fit_winkler, "power": fit_power, "kernel": fit_kernel}


def label_regions(mask, min_px=20):
    """Connected contact components, largest first: list of (region mask, centroid (u, v) in px, area px)."""
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        r = lab == i
        area = int(r.sum())
        if area >= min_px:
            v, u = np.nonzero(r)
            out.append((r, np.array([u.mean(), v.mean()]), area))
    return sorted(out, key=lambda e: -e[2])


def match_regions(centroids_px, expected_px):
    """Assign detected regions to bosses (Hungarian on pixel distance). Returns region index per boss."""
    from scipy.optimize import linear_sum_assignment
    C = np.linalg.norm(np.asarray(expected_px)[:, None, :] - np.asarray(centroids_px)[None, :, :], axis=-1)
    rows, cols = linear_sum_assignment(C)
    order = np.empty(len(expected_px), int)
    order[rows] = cols
    return order


def pressure_cop_px(fit, sample):
    """Centre of pressure (u, v) [px] of the predicted pressure map, to compare with the ATI CoP via registration."""
    mask = np.any(sample.regions, axis=0)
    p = fit.pressure(sample.depth_mm, mask)
    v, u = np.indices(p.shape)
    return np.array([(p * u).sum(), (p * v).sum()]) / p.sum()
