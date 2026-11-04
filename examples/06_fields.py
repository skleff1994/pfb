"""Gallery of the pressure families of `pfb.fields` (one draw per family, 4x4 taxels, unit pad).

Each panel shows the pressure (own colour scale), the taxel grid and the CoP. The "partial"
family is clipped by the pad edge. Figures: 06_fields.png and $PFB_DOC_FIG/fields.pdf.
"""

import numpy as np

import pfb
from _style import COLOR, MUTED, plt, save

patch = pfb.Patch(1.0, 1.0, N=64)
taxels = pfb.TaxelGrid(patch, 4, 4)
fields = pfb.population(patch, n_per_family=1, seed=0)
TITLE = {"hertz": "Hertz", "xydas_kao": "Xydas-Kao", "flat_punch": "flat punch", "ramp": "ramp (tilted)",
         "partial": "partial (clipped by the pad)", "line": "line contact", "asperities": "asperities"}
n = patch.N + 1
X, Y = (patch.points[:, k].reshape(n, n) for k in (0, 1))

fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
for ax, field in zip(axes.flat, fields):
    rho = field.measure.w / patch.quad
    ax.pcolormesh(X, Y, rho.reshape(n, n), shading="nearest", cmap="Blues", vmin=0,
                  vmax=np.percentile(rho[rho > 0], 99), rasterized=True)
    for lo, hi in zip(taxels.lo, taxels.hi):
        ax.add_patch(plt.Rectangle(lo, *(hi - lo), fill=False, ec=MUTED, lw=0.5))
    ax.plot(*field.measure.cop, "x", color=COLOR["truth"], ms=7, mew=2)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(TITLE[field.family], loc="left", fontsize=10)
    ax.text(0, -0.08, field.label, transform=ax.transAxes, fontsize=6.5, color=MUTED, va="top")
axes.flat[-1].axis("off")
fig.suptitle("Pressure families (x: CoP, grid: 4x4 taxels)", x=0.01, ha="left")
fig.tight_layout()
save(fig, "06_fields.png", pdf_stem="fields")
