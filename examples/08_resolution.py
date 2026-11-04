"""F2: tightness of the bounds against taxel resolution, and the value of measuring the CoP.

Population of `pfb.population` (7 families x 3 fields, unit pad), k x k taxels, two directions
(see _study.py). Panels 1-2: bound / true support for M_T and M_F, median over the fields with the
interquartile band; the M_F upper curve is the sweep of Thm 7.3 (a value <= the exact bound), and
the dashed line is the certified Prop 7.4 upper bound. The distance between the M_F and M_T curves
is what measuring the CoP buys. Panel 3: gap (upper - lower) / (mu |b| f_N) against the max taxel
diameter, with the Prop 8.3 bounds (M_T: dmax, M_F: 2 dmax).
These are support values in fixed directions q; the capacity along a load (load factor) is in example 10.
Run with --quick for a fast version, --recompute to ignore the cache.
Figures: 08_resolution.png and $PFB_DOC_FIG/resolution.pdf.
"""

import numpy as np

from _style import COLOR, MUTED, plt, save
from _study import DIRECTIONS, MU, get_study, ratio
from pfb.plot import LABEL, plot_band

S, names = get_study(__doc__.splitlines()[0])
ks, dmax = S["ks"], S["dmax"]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, (iq, name) in zip(axes[:2], enumerate(names)):
    for key in ("MT", "MF"):
        plot_band(ax, ks, ratio(S, f"{key}_lo", iq), COLOR[key], LABEL[key])
        plot_band(ax, ks, ratio(S, f"{key}_up", iq), COLOR[key])
    plot_band(ax, ks, ratio(S, "ref_up", iq), COLOR["MF"], f"{LABEL['MF']} certified upper (Prop 7.4)", ls="--", marker=None)
    ax.axhline(1, color=COLOR["truth"], lw=1, label="true support")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [f"{k}x{k}" for k in ks])
    ax.set_ylim(0, 2.5)
    ax.set_xlabel("taxels")
    ax.set_title(name, loc="left", fontsize=10)
axes[0].set_ylabel("bound / true support")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8, labelcolor=MUTED,
           bbox_to_anchor=(0.36, -0.06))

ax = axes[2]
b = abs(DIRECTIONS[names[0]][2])  # first direction q = (0, 0, 1), |b| = 1
for key, factor in (("MT", 1), ("MF", 2)):
    gap = (S[f"{key}_up"][:, :, 0] - S[f"{key}_lo"][:, :, 0]) / (MU * b * S["fN"][:, None])
    plot_band(ax, dmax, gap, COLOR[key], f"{LABEL[key]} gap")
    ax.plot(dmax, factor * dmax, ":", color=COLOR[key], lw=1.2, label=f"{LABEL[key]} bound, Prop 8.3")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("max taxel diameter (pad sides)")
ax.set_ylabel("gap / ($\\mu |b| f_N$)")
ax.set_title("q = (0, 0, 1): gap vs resolution", loc="left", fontsize=10)
ax.legend(fontsize=8, labelcolor=MUTED, loc="lower right")
fig.suptitle(f"Bounds over {len(S['labels'])} contact-mechanics fields (median and interquartile range)",
             x=0.01, y=1.02, ha="left")
fig.tight_layout()
save(fig, "08_resolution.png", pdf_stem="resolution")

# --- table of medians --------------------------------------------------------------------------
for iq, name in enumerate(names):
    print(f"\n{name}: medians of bound / true")
    print(f"{'k':>6} | {'MT lo':>6} {'MT up':>6} | {'MF lo':>6} {'MF up':>6} {'ref up':>6} | {'CoP gain':>8}")
    for j, k in enumerate(ks):
        med = {key: np.median(ratio(S, key, iq)[:, j]) for key in ("MT_lo", "MT_up", "MF_lo", "MF_up", "ref_up")}
        width_T = ratio(S, "MT_up", iq)[:, j] - ratio(S, "MT_lo", iq)[:, j]
        width_F = ratio(S, "MF_up", iq)[:, j] - ratio(S, "MF_lo", iq)[:, j]
        gain = np.median(1 - width_T / width_F)
        print(f"{f'{k}x{k}':>6} | {med['MT_lo']:6.3f} {med['MT_up']:6.3f} | {med['MF_lo']:6.3f} {med['MF_up']:6.3f}"
              f" {med['ref_up']:6.3f} | {gain:8.1%}")
