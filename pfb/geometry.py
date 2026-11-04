"""Contact patch, support grid and taxel partitions.

Everything lives on a rectangular patch P = [0, Lx] x [0, Ly] (Ly = 0 gives a segment).
Measures are represented by nonnegative weights on the nodes of a fixed "support grid"
of P, so every GMP of the draft becomes a finite LP over those weights.
"""

import numpy as np

# Skew-symmetric matrix of eq. (17): x × y = (J x)^T y.
J = np.array([[0.0, -1.0], [1.0, 0.0]])


class Patch:
    """Rectangular contact patch [0, Lx] x [0, Ly] with an (N+1) x (N+1) node grid.

    If Ly == 0 the patch is the segment [0, Lx] x {0} and the grid has N+1 nodes.
    """

    def __init__(self, Lx, Ly=0.0, N=48):
        self.Lx, self.Ly, self.N = float(Lx), float(Ly), int(N)
        self.is_segment = self.Ly == 0.0
        nx = self.N + 1
        ny = 1 if self.is_segment else self.N + 1
        # Integer node indices (i, j); taxel membership is decided on these, not on floats.
        i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        self.ij = np.stack([i.ravel(), j.ravel()], axis=1)
        self.h = np.array([self.Lx / self.N, 0.0 if self.is_segment else self.Ly / self.N])
        self.points = self.ij * self.h  # (K, 2)
        # Trapezoidal quadrature weights (length or area carried by each node).
        tx = np.full(nx, self.h[0]); tx[[0, -1]] /= 2
        ty = np.ones(1) if self.is_segment else np.full(ny, self.h[1])
        if not self.is_segment:
            ty[[0, -1]] /= 2
        self.quad = (tx[:, None] * ty[None, :]).ravel()

    @property
    def K(self):
        """Number of nodes of the support grid."""
        return len(self.points)

    @property
    def diam(self):
        """Diameter of the patch (its diagonal), the length scale of the torques."""
        return float(np.hypot(self.Lx, self.Ly))

    def __repr__(self):
        return f"Patch(Lx={self.Lx}, Ly={self.Ly}, N={self.N}, K={self.K})"


class TaxelGrid:
    """Partition of a patch into nx x ny rectangular taxels P_i.

    Cells are half-open [lo, hi) except the last one along each axis, so that every
    support-grid node belongs to exactly one taxel. Requires N divisible by nx (and ny).
    """

    def __init__(self, patch, nx, ny=1):
        if patch.is_segment and ny != 1:
            raise ValueError("a segment patch only admits ny = 1")
        if patch.N % nx or (not patch.is_segment and patch.N % ny):
            raise ValueError(f"N={patch.N} must be divisible by nx={nx} and ny={ny}")
        self.patch, self.nx, self.ny = patch, int(nx), int(ny)
        sx = patch.N // nx                            # number of patch points per taxel along x 
        sy = 1 if patch.is_segment else patch.N // ny # number of patch points per taxel along y
        ix = np.minimum(patch.ij[:, 0] // sx, nx - 1) # taxel x coordinate of each patch point
        iy = np.minimum(patch.ij[:, 1] // sy, ny - 1) # taxel y coordinate of each patch point
        self.index = iy * nx + ix  # taxel id of every support-grid node, shape (K,) (taxel ids range from 0 to nx*ny-1)
        # Taxel rectangles: lo/hi corners, shape (r, 2).
        cx, cy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy") # taxel grid
        cx, cy = cx.ravel(), cy.ravel()  # id = cy * nx + cx              
        size = np.array([patch.Lx / nx, patch.Ly / ny])
        self.lo = np.stack([cx, cy], axis=1) * size
        self.hi = self.lo + size

    @property
    def r(self):
        """Total number of taxels nx * ny."""
        return self.nx * self.ny # total number of taxels

    @property
    def diam(self):
        """diam(P_i) for every taxel, shape (r,)."""
        return np.linalg.norm(self.hi - self.lo, axis=1)

    def downsample(self, measure):
        """Taxel forces F_i = nu(P_i), eq. (45)."""
        return np.bincount(self.index, weights=measure.w, minlength=self.r) 
        # Sum up the measure over each taxel : number of points in each taxel weighted by measure w

    def refine(self, k=2):
        """Nested refinement: every taxel split into k x k (k x 1 on a segment)."""
        return TaxelGrid(self.patch, self.nx * k, 1 if self.patch.is_segment else self.ny * k)

    def __repr__(self):
        return f"TaxelGrid({self.nx}x{self.ny}, max diam={self.diam.max():.3g})"


def directions(n, torque_scale=1.0):
    """n roughly uniform directions q = (a1, a2, b) on the unit sphere (Fibonacci lattice).

    Only the upper half (b >= 0) is returned, since h(q) = h(-q) for all support
    functions here. `torque_scale` multiplies b, e.g. 1/L to balance force and torque units.
    """
    k = np.arange(n) + 0.5
    b = k / n  # uniform in [0, 1] -> uniform on the upper hemisphere
    phi = np.pi * (1 + 5**0.5) * k
    s = np.sqrt(1 - b**2)
    return np.stack([s * np.cos(phi), s * np.sin(phi), torque_scale * b], axis=1)
