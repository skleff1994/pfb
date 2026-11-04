"""F3: certification vs. plug-in estimation of the frictional capacity.

Same population study as 08 (cached). At each resolution, for each direction:
  grey   plug-in / true: the taxel forces spread uniformly over each taxel (`pfb.plugin_uniform`)
  green  certified M_T lower bound / true (Thm 6.1)
  orange certified M_F lower bound / true (Thm 7.2)
Values above 1 overestimate the capacity, i.e. predict that a load is held when it may slip.
The certified lower bounds are <= 1 by construction; the percentage above each grey box is the
share of fields for which the plug-in estimate is unsafe.
The comparison is on support values in fixed directions q; example 10 repeats it for load factors.
Run with --quick for a fast version, --recompute to ignore the cache.
Figures: 09_plugin.png and $PFB_DOC_FIG/plugin.pdf.
"""

import numpy as np
from matplotlib.patches import Patch as Swatch

from _style import COLOR, MUTED, plt, save
from _study import get_study, ratio
from pfb.plot import LABEL

S, names = get_study(__doc__.splitlines()[0])
ks = S["ks"]
SERIES = [("plugin", "#9b9a96", "plug-in (uniform per taxel)"),
          ("MT_lo", COLOR["MT"], f"certified {LABEL['MT']} lower bound"),
          ("MF_lo", COLOR["MF"], f"certified {LABEL['MF']} lower bound")]
x = np.arange(len(ks))
width = 0.25

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
for ax, (iq, name) in zip(axes, enumerate(names)):
    for s, (key, color, _) in enumerate(SERIES):
        R = ratio(S, key, iq)
        ax.boxplot([R[:, j] for j in range(len(ks))], positions=x + (s - 1) * width, widths=0.8 * width,
                   patch_artist=True, showfliers=True, manage_ticks=False,
                   boxprops=dict(facecolor=color, edgecolor=color, alpha=0.5),
                   medianprops=dict(color=color, lw=2), whiskerprops=dict(color=color),
                   capprops=dict(color=color), flierprops=dict(marker=".", markeredgecolor=color, ms=4))
    unsafe = (ratio(S, "plugin", iq) > 1).mean(axis=0)
    top = ratio(S, "plugin", iq).max(axis=0)
    for j in range(len(ks)):
        ax.text(x[j] - width, top[j] + 0.03, f"{unsafe[j]:.0%}", ha="center", fontsize=7.5, color=MUTED)
    ax.axhline(1, color=COLOR["truth"], lw=1)
    ax.set_xticks(x, [f"{k}x{k}" for k in ks])
    ax.set_xlabel("taxels")
    ax.set_title(name, loc="left", fontsize=10)
axes[0].set_ylabel("estimate / true support")
fig.legend([Swatch(color=c, alpha=0.5) for _, c, _ in SERIES], [lab for *_, lab in SERIES],
           loc="lower center", ncol=3, fontsize=8, labelcolor=MUTED, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("Plug-in estimates can overestimate the capacity; certified lower bounds cannot "
             "(% above each grey box: share of unsafe plug-in estimates)", x=0.01, y=1.02, ha="left", fontsize=10)
fig.tight_layout()
save(fig, "09_plugin.png", pdf_stem="plugin")

# --- table ---------------------------------------------------------------------------------------
for iq, name in enumerate(names):
    P = ratio(S, "plugin", iq)
    print(f"\n{name}")
    print(f"{'k':>6} | {'plug-in > true':>14} {'worst plug-in':>13} | {'median MT lo':>12} {'median MF lo':>12}")
    for j, k in enumerate(ks):
        print(f"{f'{k}x{k}':>6} | {np.mean(P[:, j] > 1):14.0%} {P[:, j].max():13.3f} | "
              f"{np.median(ratio(S, 'MT_lo', iq)[:, j]):12.3f} {np.median(ratio(S, 'MF_lo', iq)[:, j]):12.3f}")
