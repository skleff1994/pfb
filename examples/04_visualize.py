"""Visual tour: patch and taxels, the true wrench set W(nu), its support and the bounds.

Figures:
  04_overview.png    patch | 3D W(nu) with the M_T inner/outer sets | support | (f_T1, tau) projection
  04_contraction.png (f_T1, tau) projection of the M_T sets for k x k taxels, k = 1, 2, 4, 8
  04_segment.png     the Example 4.8 counterexample: two measures with the same M_0, different W
"""

import numpy as np

import pfb
import pfb.plot as pp
from _style import plt, save

mu = 0.5
patch = pfb.Patch(0.05, 0.05, N=48)
taxels = pfb.TaxelGrid(patch, 4, 4)
nu = (pfb.gaussian(patch, (0.03, 0.01), 0.01, fN=1.0)) 
      # + pfb.gaussian(patch, (0.75, 0.2), 0.08, fN=0.5))

# --- overview --------------------------------------------------------------------------
fig, _ = pp.overview(nu, taxels, mu, plane="fx-tau", n_plane=48, mf_kwargs=dict(n_c=5, polish=False))
save(fig, "04_overview.png")

# --- contraction of the M_T sets under refinement ------------------------------------------
Q, _ = pp.plane_directions(48, "fx-tau", torque_scale=1 / patch.diam)
ks = [1, 2, 4, 8]
fig, axes = plt.subplots(1, len(ks), figsize=(14, 3.6), sharex=True, sharey=True)
for ax, k in zip(axes, ks):
    t = pfb.TaxelGrid(patch, k, k)
    pp.plot_projection(Q, "fx-tau", pp.all_bounds(nu, t, Q, mu, which=("MT",)), ax)
    ax.set_title(f"$\\mathcal{{M}}_T$, {k}x{k} taxels", loc="left", fontsize=10)
axes[0].legend(fontsize=8, loc="lower left")
fig.tight_layout()
save(fig, "04_contraction.png")

# --- segment counterexample (Example 4.8) --------------------------------------------------
L, fN = 1.0, 1.0
seg = pfb.Patch(L, 0.0, N=40)
seg_taxels = pfb.TaxelGrid(seg, 4)
nus = {"Dirac at the midpoint": pfb.from_diracs(seg, [(L / 2, 0)], [fN]),
       "two half-Diracs at the ends": pfb.from_diracs(seg, [(0, 0), (L, 0)], [fN / 2, fN / 2]),
       "uniform": pfb.uniform(seg, fN)}
Q, _ = pp.plane_directions(72, "fx-tau", torque_scale=1 / L)
B0 = pp.all_bounds(nus["uniform"], seg_taxels, Q, mu, which=("M0",))  # same M_0 for all three
fig, ax = plt.subplots(figsize=(5, 4))
V = pp.polygon_from_support(Q[:, [0, 2]], B0["M0"][1])
ax.fill(V[:, 0], V[:, 1], color=pp.COLOR["M0"], alpha=0.15, lw=0)
ax.plot(V[:, 0], V[:, 1], color=pp.COLOR["M0"], label="$\\mathcal{M}_0$ upper bound ($W^+$)")
for (name, m), ls in zip(nus.items(), ["-", "--", ":"]):
    V = pp.polygon_from_support(Q[:, [0, 2]], pfb.support(m, Q, mu))
    ax.plot(V[:, 0], V[:, 1], ls, color=pp.COLOR["truth"], lw=1.5, label=name)
ax.set_xlabel(pp.AXIS[0]); ax.set_ylabel(pp.AXIS[2])
ax.set_title("same $(f_N, c)$, different $W(\\nu)$", loc="left", fontsize=10)
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
save(fig, "04_segment.png")
