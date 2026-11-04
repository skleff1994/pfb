"""Visualization: patch and pressure, the wrench set W(nu), its support function and the bounds.

Import explicitly with `import pfb.plot as pp` (the core package does not need matplotlib).

Sets drawn from a support-type function h over finitely many directions q_i are the polytopes
K_h = {w : <q_i, w> <= h(q_i)}. For h = h_true this approximates W(nu) from outside. For the
upper bound it approximates the outer set W+(y), and for the lower bound the inner set W-(y)
of eq. (30). In a coordinate plane, K_h built from in-plane q is the projection of the set onto
that plane for h_true and the upper bound. For the lower bound it contains the projection of W-(y).
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, HalfspaceIntersection

from .bounds import (bounds_M0, bounds_MF, bounds_MT, cop_set_MF, measure_M0, measure_MF,
                     measure_MT)
from .measure import support, support_points
from .reduced import reduced_M0, reduced_MF, reduced_MT

COLOR = {"M0": "#2a78d6", "MF": "#eb6834", "MT": "#1baf7a", "truth": "#0b0b0b"}
LABEL = {"M0": "$\\mathcal{M}_0$", "MF": "$\\mathcal{M}_F$", "MT": "$\\mathcal{M}_T$"}
MUTED = "#52514e"
AXIS = ["$f_{T,1}$", "$f_{T,2}$", "$\\tau_N$"]
PLANES = {"fx-fy": (0, 1), "fx-tau": (0, 2), "fy-tau": (1, 2)}


# --- directions and bounds -----------------------------------------------------------------

def plane_directions(n, plane="fx-tau", torque_scale=1.0):
    """n directions q evenly spread on the unit circle of a coordinate plane of (a1, a2, b).

    Returns (Q, theta) with Q of shape (n, 3). `torque_scale` multiplies the b component.
    """
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    Q = np.zeros((n, 3))
    i, j = PLANES[plane]
    Q[:, i], Q[:, j] = np.cos(theta), np.sin(theta)
    Q[:, 2] *= torque_scale
    return Q, theta


def all_bounds(nu, taxels, Q, mu=1.0, which=("M0", "MF", "MT"), mf_kwargs=None, method="reduced"):
    """Dict {name: (lower, upper)} of the requested bounds over the directions Q, plus 'truth'.

    method="reduced" uses the finite-dimensional formulations of the draft (`pfb.reduced`),
    method="grid" the LPs on the support grid (`pfb.bounds`).
    """
    M0, MF, MT = (reduced_M0, reduced_MF, reduced_MT) if method == "reduced" else (bounds_M0, bounds_MF, bounds_MT)
    out = {"truth": support(nu, Q, mu)}
    if "M0" in which:
        out["M0"] = M0(taxels.patch, measure_M0(nu), Q, mu)
    if "MF" in which:
        out["MF"] = MF(taxels, measure_MF(nu, taxels), Q, mu, **(mf_kwargs or {}))
    if "MT" in which:
        out["MT"] = MT(taxels, measure_MT(nu, taxels), Q, mu)
    return out


# --- geometry helpers ------------------------------------------------------------------------

def polygon_from_support(normals, h, tol=1e-9):
    """Vertices of the 2D polygon {w : <n_i, w> <= h_i}, normals sorted by angle (closed loop).

    Consecutive lines are intersected, and only vertices satisfying all constraints are kept.
    Degenerate (flat) sets come out as a polygon of zero area, which still plots as a segment.
    """
    n, h = np.asarray(normals, float), np.asarray(h, float)
    scale = tol * max(1.0, np.abs(h).max())
    verts = []
    for k in range(len(n)):
        A = np.array([n[k], n[(k + 1) % len(n)]])
        if abs(np.linalg.det(A)) < 1e-12:
            continue
        v = np.linalg.solve(A, [h[k], h[(k + 1) % len(n)]])
        if np.all(n @ v <= h + scale):
            verts.append(v)
    verts = np.array(verts)
    return np.vstack([verts, verts[:1]]) if len(verts) else verts


def polytope_from_support(Q, h):
    """ConvexHull of the 3D polytope {w : <q_i, w> <= h_i}, or None if 0 is not interior."""
    h = np.asarray(h, float)
    if np.any(h <= 1e-9 * np.abs(h).max()):
        return None
    hs = HalfspaceIntersection(np.c_[Q, -h], interior_point=np.zeros(3))
    return ConvexHull(hs.intersections)


# --- plots -------------------------------------------------------------------------------

def plot_patch(nu, taxels=None, ax=None, cop_set=True):
    """Pressure of nu over the patch, taxel grid, CoP and (optionally) the compatible CoP set C_MF."""
    ax = ax or plt.gca()
    P = nu.patch
    rho = nu.w / P.quad  # density w.r.t. area (length on a segment)
    if P.is_segment:
        ax.fill_between(P.points[:, 0], rho, color="#2a78d6", alpha=0.25, lw=0)
        ax.plot(P.points[:, 0], rho, color="#2a78d6", lw=1.5)
        ax.set_ylabel("pressure")
        if taxels is not None:
            for x in np.unique(np.r_[taxels.lo[:, 0], taxels.hi[:, 0]]):
                ax.axvline(x, color=MUTED, lw=0.6)
        ax.axvline(nu.cop[0], color=COLOR["truth"], ls=":", lw=1.2, label="CoP")
    else:
        n = P.N + 1
        X, Y = P.points[:, 0].reshape(n, n), P.points[:, 1].reshape(n, n)
        m = ax.pcolormesh(X, Y, rho.reshape(n, n), shading="nearest", cmap="Blues")
        plt.colorbar(m, ax=ax, label="pressure", shrink=0.8)
        if taxels is not None:
            for lo, hi in zip(taxels.lo, taxels.hi):
                ax.add_patch(plt.Rectangle(lo, *(hi - lo), fill=False, ec=MUTED, lw=0.6))
            if cop_set:
                lo, hi = cop_set_MF(taxels, taxels.downsample(nu))
                ax.add_patch(plt.Rectangle(lo, *(hi - lo), fill=False, ec=COLOR["MF"], lw=1.5, ls="--",
                                           label="$\\mathcal{C}_{\\mathcal{M}_F}$"))
        ax.plot(*nu.cop, "x", color=COLOR["truth"], ms=8, mew=2, label="CoP")
        ax.set_aspect("equal")
        ax.set_ylabel("$x_2$")
    ax.set_xlabel("$x_1$")
    ax.set_title("patch, pressure and taxels", loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    return ax


def plot_wrench_set(nu, mu=1.0, ax=None, n=600, bounds=None):
    """3D view of W(nu) (convex hull of support points), plus optional bound polytopes.

    bounds: {name: (Q, lower, upper)}. The upper bound polytope (W+) is drawn as a faint shell
    and the lower one (W-) as a more opaque surface, both in the hypothesis colour.
    """
    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")
    from .geometry import directions
    Q = directions(n // 2, torque_scale=1.0 / nu.patch.diam)
    Q = np.vstack([Q, -Q])
    pts = support_points(nu, Q, mu)
    hull = ConvexHull(pts)
    ax.add_collection3d(Poly3DCollection(pts[hull.simplices], facecolor="#9b9a96", edgecolor="none",
                                         alpha=0.35, label="$W(\\nu^*)$"))
    for name, (Qb, lo, up) in (bounds or {}).items():
        for h, solid in ((up, False), (lo, True)):
            poly = polytope_from_support(Qb, h)
            if poly is None:
                continue
            tri = poly.points[poly.simplices]
            ax.add_collection3d(Poly3DCollection(tri, facecolor=COLOR[name], edgecolor="none",
                                                 alpha=0.5 if solid else 0.08))
            pts = np.vstack([pts, poly.points])
    lo, hi = pts.min(0), pts.max(0)
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
    ax.set_box_aspect((1, 1, 0.7))  # torque range is usually much smaller; stretch it for readability
    ax.set_xlabel(AXIS[0]); ax.set_ylabel(AXIS[1]); ax.set_zlabel(AXIS[2])
    ax.set_title("admissible wrench set", loc="left", fontsize=10)
    return ax


def plot_projection(Q, plane, B, ax=None):
    """Sets K_h in a coordinate plane for the truth and for each bound in B = all_bounds(...).

    Upper bound sets are drawn as solid lines and lower bound sets as dashed lines. The true
    projection of W(nu) is filled.
    """
    ax = ax or plt.gca()
    i, j = PLANES[plane]
    N = Q[:, [i, j]]
    V = polygon_from_support(N, B["truth"])
    ax.fill(V[:, 0], V[:, 1], color="#9b9a96", alpha=0.35, lw=0)
    ax.plot(V[:, 0], V[:, 1], color=COLOR["truth"], lw=1.5, label="$W(\\nu^*)$")
    for name in ("M0", "MF", "MT"):
        if name not in B:
            continue
        lo, up = B[name]
        Vu, Vl = polygon_from_support(N, up), polygon_from_support(N, lo)
        ax.plot(Vu[:, 0], Vu[:, 1], color=COLOR[name], lw=1.5, label=f"{LABEL[name]} upper")
        if len(Vl):
            ax.plot(Vl[:, 0], Vl[:, 1], "--", color=COLOR[name], lw=1.2, label=f"{LABEL[name]} lower")
    ax.set_xlabel(AXIS[i]); ax.set_ylabel(AXIS[j])
    ax.set_title(f"projection on ({AXIS[i]}, {AXIS[j]})", loc="left", fontsize=10)
    return ax


def plot_support(theta, B, ax=None, polar=False):
    """Support function vs. direction angle: truth as a line, each bound as a band."""
    ax = ax or plt.gca()
    th = np.r_[theta, theta[:1] + 2 * np.pi] if polar else theta
    close = (lambda v: np.r_[v, v[:1]]) if polar else (lambda v: v)
    for name in ("M0", "MF", "MT"):
        if name in B:
            lo, up = map(close, B[name])
            ax.fill_between(th, lo, up, color=COLOR[name], alpha=0.2, lw=0)
            ax.plot(th, lo, color=COLOR[name], lw=1.2, label=LABEL[name])
            ax.plot(th, up, color=COLOR[name], lw=1.2)
    ax.plot(th, close(B["truth"]), color=COLOR["truth"], lw=1.5, label="$h_{W(\\nu^*)}$")
    if not polar:
        ax.set_xlabel("direction angle $\\theta$")
        ax.set_xticks(np.pi * np.arange(5) / 2, ["0", "$\\pi/2$", "$\\pi$", "$3\\pi/2$", "$2\\pi$"])
    ax.set_title("support function and bounds", loc="left", fontsize=10)
    return ax


def plot_band(ax, x, Y, color, label=None, ls="-", percentiles=(25, 75), marker="o"):
    """Median of Y over its first axis (the samples) as a line, with a percentile band.

    Y has shape (n_samples, len(x)); NaNs are ignored. Used for statistics over a field population.
    """
    lo, med, hi = np.nanpercentile(Y, [percentiles[0], 50, percentiles[1]], axis=0)
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
    ax.plot(x, med, ls, color=color, label=label, marker=marker, ms=4, lw=1.5)
    return ax


def overview(nu, taxels, mu=1.0, plane="fx-tau", n_plane=60, which=("M0", "MF", "MT"),
             n_3d=150, mf_kwargs=None, method="reduced"):
    """One figure: patch | 3D wrench set with M_T sets | support in `plane` | projection.

    M_F is the slow one (outer search over the CoP). mf_kwargs is passed to the M_F bound
    function, e.g. dict(polish=False, n_c=5). `method` is passed to `all_bounds`.
    """
    s = 1.0 / nu.patch.diam  # balance force and torque units in the sampled directions
    Q, theta = plane_directions(n_plane, plane, torque_scale=s if plane != "fx-fy" else 1.0)
    B = all_bounds(nu, taxels, Q, mu, which, mf_kwargs, method)
    from .geometry import directions
    Q3 = directions(n_3d // 2, torque_scale=s)
    Q3 = np.vstack([Q3, -Q3])
    lo3, up3 = (reduced_MT if method == "reduced" else bounds_MT)(taxels, measure_MT(nu, taxels), Q3, mu)

    fig = plt.figure(figsize=(17, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.1, 1.2, 1, 1], wspace=0.35)
    plot_patch(nu, taxels, fig.add_subplot(gs[0]))
    ax3 = fig.add_subplot(gs[1], projection="3d")
    plot_wrench_set(nu, mu, ax3, bounds={"MT": (Q3, lo3, up3)})
    ax3.set_title("$W(\\nu^*)$ (grey), $\\mathcal{M}_T$ inner / outer", loc="left", fontsize=10)
    plot_support(theta, B, fig.add_subplot(gs[2]))
    ax = plot_projection(Q, plane, B, fig.add_subplot(gs[3]))
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    return fig, B
