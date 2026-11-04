"""Certified load factor (pfb.certify) over the field population: capacity, not support values.

For a load w about the pad centre o, the load factor s*(nu) = max {s : s w in W(nu)} is the capacity
along w (for w = (0, 0, 1), the maximal pure torque; this is inf_a h((a, 1)), not h((0, 0, 1))).
Left, middle: certified s*_lo / true s* for M_T and M_F against resolution (median and interquartile
range over the population), and the plug-in estimate / true s*. Right: solve time of one certificate.
Figures: 10_load_factor.png and $PFB_DOC_FIG/load_factor.pdf.
"""

import time

import numpy as np

import pfb
from _style import COLOR, MUTED, plt, save
from pfb.certify import PadTerm, load_factor_lower, load_factor_true
from pfb.plot import LABEL, plot_band

MU = 1.0
patch = pfb.Patch(1.0, 1.0, N=64)
fields = pfb.population(patch, n_per_family=3, seed=0)
o = np.array([0.5, 0.5])  # loads are expressed about the pad centre
LOADS = {"pure torque  w = (0, 0, 1)": np.array([0.0, 0.0, 1.0]),
         "force at lever 0.2  w = (1, 0, 0.2)": np.array([1.0, 0.0, 0.2])}
ks = [1, 2, 4, 8, 16, 32]

shape = (len(fields), len(ks), len(LOADS))
R = {key: np.zeros(shape) for key in ("MT", "MF", "plugin")}
times = {"MT": [[] for _ in ks], "MF": [[] for _ in ks]}
t0 = time.time()
for i, field in enumerate(fields):
    nu = field.measure
    m = nu.w > 0
    for j, k in enumerate(ks):
        tx = pfb.TaxelGrid(patch, k, k)
        F = tx.downsample(nu)
        est = pfb.plugin_uniform(tx, F)
        for l, w in enumerate(LOADS.values()):
            true = load_factor_true(patch.points[m], nu.w[m], w, MU, o)
            for key, cop in (("MT", nu.cop), ("MF", None)):
                t = time.perf_counter()
                R[key][i, j, l] = load_factor_lower(PadTerm.from_taxels(tx, F, cop=cop), w, MU, o) / true
                times[key][j].append(time.perf_counter() - t)
            R["plugin"][i, j, l] = load_factor_true(patch.points, est.w, w, MU, o) / true
    print(f"[{i + 1}/{len(fields)}] {field.label}  ({time.time() - t0:.0f} s)")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, (l, name) in zip(axes[:2], enumerate(LOADS)):
    plot_band(ax, ks, R["plugin"][:, :, l], "#9b9a96", "plug-in / true (not a bound)")
    for key in ("MT", "MF"):
        plot_band(ax, ks, R[key][:, :, l], COLOR[key], f"certified {LABEL[key]} / true")
    ax.axhline(1, color=COLOR["truth"], lw=1)
    ax.set_xscale("log", base=2); ax.set_xticks(ks, [f"{k}x{k}" for k in ks])
    ax.set_ylim(0, 1.6); ax.set_xlabel("taxels"); ax.set_title(name, loc="left", fontsize=10)
axes[0].set_ylabel("load factor / true load factor")
ax = axes[2]
for key in ("MT", "MF"):
    ax.plot(ks, [1e3 * np.median(t) for t in times[key]], "o-", color=COLOR[key], ms=4, label=LABEL[key])
ax.axhline(50, color=MUTED, ls=":", lw=1, label="20 Hz budget")
ax.set_xscale("log", base=2); ax.set_xticks(ks, [f"{k}x{k}" for k in ks]); ax.set_yscale("log")
ax.set_xlabel("taxels"); ax.set_ylabel("median solve time (ms)")
ax.set_title("one certificate (two SOCPs)", loc="left", fontsize=10)
ax.legend(fontsize=8, labelcolor=MUTED)
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8, labelcolor=MUTED,
           bbox_to_anchor=(0.36, -0.06))
fig.suptitle(f"Certified capacity along a load direction, {len(fields)} contact-mechanics fields", x=0.01, y=1.02, ha="left")
fig.tight_layout()
save(fig, "10_load_factor.png", pdf_stem="load_factor")

for l, name in enumerate(LOADS):
    print(f"\n{name}: medians")
    print(f"{'k':>6} | {'MT/true':>8} {'MF/true':>8} {'plug-in/true':>12} {'plug-in > 1':>11} | {'ms MT':>6} {'ms MF':>6}")
    for j, k in enumerate(ks):
        print(f"{f'{k}x{k}':>6} | {np.median(R['MT'][:, j, l]):8.3f} {np.median(R['MF'][:, j, l]):8.3f} "
              f"{np.median(R['plugin'][:, j, l]):12.3f} {np.mean(R['plugin'][:, j, l] > 1):11.0%} | "
              f"{1e3 * np.median(times['MT'][j]):6.1f} {1e3 * np.median(times['MF'][j]):6.1f}")
