"""Contraction of the bounds under nested taxel refinement (Props. 8.2 and 8.3).

For k x k taxels with k = 1, 2, 4, 8, 16, 32, plots the gap upper - lower of the M_F and M_T
bounds (finite-dimensional reductions, `pfb.reduced`) against max_i diam(P_i), next to the
theoretical bounds of Prop. 8.3:
    M_F: gap <= 2 mu |b| f_N max diam,     M_T: gap <= mu |b| f_N max diam.
The last table column is the extra tightening due to the CoP: (M_F width - M_T width) / M_F width.
The slope on the log-log plot gives the empirical rate. The last panel shows the contraction
of the compatible CoP set C_MF (Prop. 5.6).
"""

import numpy as np

import pfb
from _style import COLOR, MUTED, plt, save

mu, Lx, Ly = 0.5, 1.0, 0.6
patch = pfb.Patch(Lx, Ly, N=64)
nu = (pfb.gaussian(patch, (0.3, 0.35), 0.12, fN=1.5)
      + pfb.gaussian(patch, (0.75, 0.2), 0.08, fN=0.5))
fN = nu.fN
directions = {"pure torsion  q = (0, 0, 1)": np.array([0.0, 0.0, 1.0]),
              "mixed  q = (1, 0, 1/L)/norm": np.array([1.0, 0.0, 1.0 / Lx]) / np.sqrt(1 + Lx**-2)}
ks = [1, 2, 4, 8, 16, 32]
grids = [pfb.TaxelGrid(patch, k, k) for k in ks]
dmax = np.array([t.diam.max() for t in grids])

results = {}
for name, q in directions.items():
    h = pfb.support(nu, q, mu)
    print(f"\n{name}:  true h = {h:.5f}")
    print(f"{'k':>3} {'max diam':>9} | {'MF lower':>9} {'MF upper':>9} | {'MT lower':>9} {'MT upper':>9} | {'CoP gain':>8}")
    rows = []
    for k, t in zip(ks, grids):
        loF, upF = pfb.reduced_MF(t, pfb.measure_MF(nu, t), q, mu, n_c=5)
        loT, upT = pfb.reduced_MT(t, pfb.measure_MT(nu, t), q, mu)
        rows.append((loF, upF, loT, upT))
        gain = 1 - (upT - loT) / (upF - loF)
        print(f"{k:>3} {t.diam.max():9.4f} | {loF:9.5f} {upF:9.5f} | {loT:9.5f} {upT:9.5f} | {gain:8.1%}")
    results[name] = np.array(rows)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, (name, q) in zip(axes, directions.items()):
    R, b = results[name], abs(q[2])
    ax.loglog(dmax, R[:, 1] - R[:, 0], "o-", color=COLOR["MF"], ms=5, label="$\\mathcal{M}_F$ gap")
    ax.loglog(dmax, R[:, 3] - R[:, 2], "o-", color=COLOR["MT"], ms=5, label="$\\mathcal{M}_T$ gap")
    ax.loglog(dmax, 2 * mu * b * fN * dmax, "--", color=COLOR["MF"], lw=1.2, label="$\\mathcal{M}_F$ bound, Prop 8.3")
    ax.loglog(dmax, mu * b * fN * dmax, "--", color=COLOR["MT"], lw=1.2, label="$\\mathcal{M}_T$ bound, Prop 8.3")
    ax.set_title(name, loc="left", fontsize=10)
    ax.set_xlabel("max taxel diameter")
    ax.legend(fontsize=8, labelcolor=MUTED, loc="lower right")
axes[0].set_ylabel("upper - lower")

# Compatible CoP set C_MF: diameter vs. the bound (52).
ax = axes[2]
diamC, wdiam = [], []
for t in grids:
    F = t.downsample(nu)
    lo, hi = pfb.cop_set_MF(t, F)
    diamC.append(np.linalg.norm(hi - lo))
    wdiam.append(F @ t.diam / F.sum())
ax.loglog(dmax, diamC, "o-", color=COLOR["MF"], ms=5, label="diam $\\mathcal{C}_{\\mathcal{M}_F}$")
ax.loglog(dmax, wdiam, "--", color=MUTED, lw=1.2, label="$\\sum_i F_i$ diam$(P_i)/f_N$  (Prop 5.6)")
ax.set_title("compatible CoP set", loc="left", fontsize=10)
ax.set_xlabel("max taxel diameter")
ax.legend(fontsize=8, labelcolor=MUTED, loc="lower right")
fig.suptitle(f"Refinement k x k taxels, k = {ks}", x=0.01, y=1.03, ha="left")
fig.tight_layout()
save(fig, "03_refinement.png")
