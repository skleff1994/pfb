"""True support function vs. the M_0, M_F and M_T bounds for one pressure field.

Directions q(theta) = (cos theta, 0, sin theta / L) sweep from pure force (theta = 0) through
pure torsion (theta = pi/2) and back. The bounds are the finite-dimensional reductions of the
draft (`pfb.reduced`). On directions over the whole sphere, the script also checks:
  - the orderings of Prop. 8.1 and the cellwise / certified bounds (Props. 6.4, 7.4);
  - that the brute-force grid LPs (`pfb.bounds`) lie inside the reduced bounds.
"""

import numpy as np

import pfb
from _style import COLOR, MUTED, plt, save

mu, Lx, Ly = 0.5, 0.1, 0.1
patch = pfb.Patch(Lx, Ly, N=256)
taxels = pfb.TaxelGrid(patch, 4, 4)

# Asymmetric pressure: two blobs of different weight.
nu = (pfb.gaussian(patch, (0.03, 0.035), 0.012, fN=1.5)
      + pfb.gaussian(patch, (0.075, 0.03), 0.008, fN=0.5)
      + pfb.gaussian(patch, (0.08, 0.08), 0.008, fN=0.4))
yT = pfb.measure_MT(nu, taxels)
y0, F = yT[:2], yT[2]
# print(nu, taxels)
from pfb.plot import plot_patch
fig, ax = plt.subplots(1, 1)
ax = plot_patch(nu, taxels)
save(fig, "00_patch.png")

def all_bounds(q):
    return dict(truth=pfb.support(nu, q, mu),
                M0=pfb.reduced_M0(patch, y0, q, mu),
                MF=pfb.reduced_MF(taxels, F, q, mu, n_c=5),
                MT=pfb.reduced_MT(taxels, yT, q, mu),
                cell=pfb.cellwise_MT(taxels, yT, q, mu),
                ref=pfb.certified_ref_MF(taxels, F, q, mu))  # c_hat = centre of C_MF


# --- sweep over theta ------------------------------------------------------------------
theta = np.linspace(0, np.pi, 31)
# Q = np.stack([np.cos(theta), 0 * theta, np.sin(theta) / Lx], axis=1)
# Q = np.stack([np.cos(theta), 0 * theta, np.sin(theta) / Lx], axis=1)
Q = np.stack([np.cos(theta), np.sin(theta), 1-np.sin(theta) / Lx], axis=1)
B = all_bounds(Q)

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
panels = [("M0", "$\\mathcal{M}_0$: total force + CoP", None),
          ("MF", "$\\mathcal{M}_F$: taxel forces", ("ref", "certified fixed-reference (Prop 7.4)")),
          ("MT", "$\\mathcal{M}_T$: taxel forces + CoP", ("cell", "cellwise (Prop 6.4)"))]
for ax, (key, title, extra) in zip(axes, panels):
    lo, up = B[key]
    ax.fill_between(theta, lo, up, color=COLOR[key], alpha=0.25, lw=0)
    ax.plot(theta, lo, color=COLOR[key], label="lower / upper bound")
    ax.plot(theta, up, color=COLOR[key])
    if extra:
        elo, eup = B[extra[0]]
        ax.plot(theta, elo, "--", color=COLOR[key], lw=1.2, label=extra[1])
        ax.plot(theta, eup, "--", color=COLOR[key], lw=1.2)
    ax.plot(theta, B["truth"], color=COLOR["truth"], lw=1.5, label="true $h_{W(\\nu^*)}$")
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlabel("$\\theta$  (0: force,  $\\pi/2$: torsion)")
    ax.set_xticks([0, np.pi / 2, np.pi], ["0", "$\\pi/2$", "$\\pi$"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=8, labelcolor=MUTED)
axes[0].set_ylabel("support value")
fig.suptitle(f"Support bounds, {taxels.nx}x{taxels.ny} taxels, $\\mu$={mu}", x=0.01, y=1.03, ha="left")
fig.tight_layout()
save(fig, "02_bounds_vs_truth.png")

# --- ordering checks on the sphere --------------------------------------------------------
tol = 1e-7
Qs = pfb.directions(30, torque_scale=1 / Lx)
S = all_bounds(Qs)
h = S["truth"]
checks = {
    "Prop 8.1  lo M0 <= lo MF <= lo MT <= h": (S["M0"][0] <= S["MF"][0] + tol) & (S["MF"][0] <= S["MT"][0] + tol)
                                              & (S["MT"][0] <= h + tol),
    "Prop 8.1  h <= up MT <= min(up M0, up MF)": (h <= S["MT"][1] + tol) & (S["MT"][1] <= np.minimum(S["M0"][1], S["MF"][1]) + tol),
    "Prop 6.4  cellwise brackets MT": (S["cell"][0] <= S["MT"][0] + tol) & (S["MT"][1] <= S["cell"][1] + tol),
    "Prop 7.4  certified brackets MF": (S["ref"][0] <= S["MF"][0] + tol) & (S["MF"][1] <= S["ref"][1] + tol),
}
# Brute force: the grid LPs are inner approximations, so they must lie inside the reduced bounds.
G = dict(M0=pfb.bounds_M0(patch, y0, Qs, mu), MT=pfb.bounds_MT(taxels, yT, Qs, mu),
         MF=pfb.bounds_MF(taxels, F, Qs, mu, n_c=5, polish=False))
for key, (glo, gup) in G.items():
    rlo, rup = S[key]
    checks[f"grid {key} inside reduced {key}  (max diff {max((glo - rlo).max(), (rup - gup).max()):.1e})"] = \
        (rlo <= glo + tol) & (gup <= rup + tol)
for name, ok in checks.items():
    print(f"{'OK ' if ok.all() else 'BAD'} {name}" + ("" if ok.all() else f"  <-- {np.count_nonzero(~ok)} directions"))

# Size of the gaps between nested bounds: how much each extra measurement buys.
print(f"mean width  M0: {np.mean(S['M0'][1] - S['M0'][0]):.4f}   MF: {np.mean(S['MF'][1] - S['MF'][0]):.4f}"
      f"   MT: {np.mean(S['MT'][1] - S['MT'][0]):.4f}   cellwise: {np.mean(S['cell'][1] - S['cell'][0]):.4f}")
