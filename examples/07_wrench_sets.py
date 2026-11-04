"""F1: guaranteed (inner, W-) and possible (outer, W+) wrench sets for M_0, M_F and M_T.

One partial, tilted contact (a ramp clipped by the pad edge, outside the parametric families).
Rows: 4x4 and 16x16 taxels. Columns: pressure | M_0 | M_F | M_T, each in the (f_T1, tau_N) plane.
The true W(nu) is filled grey, W+ is the solid line (upper bound) and W- the dashed line (lower
bound); see `pfb.plot` for how the sets are built from the support bounds. M_0 does not depend on
the taxels, and its inner set is the force segment (no guaranteed torsion, Thm 4.3).
Figures: 07_wrench_sets.png and $PFB_DOC_FIG/wrench_sets.pdf.
"""

import numpy as np

import pfb
import pfb.plot as pp
from _style import plt, save

mu = 1.0
patch = pfb.Patch(1.0, 1.0, N=64)
nu = pfb.ramp(patch, center=(0.82, 0.45), radius=0.4, tilt=1.3, angle=0.75 * np.pi)
Q, _ = pp.plane_directions(72, "fx-tau", torque_scale=1 / patch.diam)

rows = [4, 16]
fig = plt.figure(figsize=(15, 7.2))
gs = fig.add_gridspec(len(rows), 4, width_ratios=[1.15, 1, 1, 1], wspace=0.5, hspace=0.35)
ref = None
for r, k in enumerate(rows):
    taxels = pfb.TaxelGrid(patch, k, k)
    B = pp.all_bounds(nu, taxels, Q, mu, mf_kwargs=dict(n_c=5, polish=True))
    ax = pp.plot_patch(nu, taxels, fig.add_subplot(gs[r, 0]))
    ax.set_title(f"pressure, {k}x{k} taxels", loc="left", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    for c, name in enumerate(("M0", "MF", "MT"), start=1):
        ax = fig.add_subplot(gs[r, c], sharex=ref, sharey=ref)
        ref = ref or ax
        pp.plot_projection(Q, "fx-tau", {"truth": B["truth"], name: B[name]}, ax)
        ax.set_title(f"{pp.LABEL[name]}, {k}x{k} taxels", loc="left", fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
    print(f"{k}x{k}: mean width  " + "  ".join(f"{n}: {np.mean(B[n][1] - B[n][0]):.4f}" for n in ("M0", "MF", "MT")))
fig.suptitle("Guaranteed (dashed) and possible (solid) wrench sets, partial tilted contact, $\\mu$ = 1",
             x=0.01, ha="left")
save(fig, "07_wrench_sets.png", pdf_stem="wrench_sets")
