"""Figure S4: reliability of the S3 grip controller against sensing delay, with the S3 margins.

Rows: P, load previewed (the S3 loop; only the tactile state is stale) and M, load measured with the
same delay (no preview, the D1 situation). Columns 1-3: load rate 1x, 2x, 4x; silent-slip rate (a
slip the policy did not flag) against the sensing delay, one line per policy at its S3 margin.
Column 4: onset cause of every slip (grip.diagnose: the largest term of the capacity shortfall),
all rates and delays pooled.
Run after delay_study.py: pixi run -e sim sim-delay-plot [--quick]
Figures: examples/figures/s4_delay.png and $PFB_DOC_FIG/delay_envelope.pdf.
"""

import argparse

import numpy as np

from delay_study import CAUSES, DATA, KEYS, POLICY_MARGINS
from grip_study import outcomes
from viz import MUTED, STYLE, plt
from _style import save  # noqa: E402  (on the path through viz)

CAUSE_COLOR = {"model": "#9b9a96", "estimate": "#d03b3b", "load": "#e0a030", "state": "#7a5bd6", "actuator": "#2b7bba",
               "stale": "#f0cf8a"}
CAUSE_LABEL = {"model": "model: Coulomb holds, simulated friction slips", "estimate": "estimate optimistic",
               "load": "load grew since it was certified", "state": "contact changed since sensing",
               "actuator": "gripper behind the command",
               "stale": "load not yet measured at the last decision"}

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--quick", action="store_true")
a = ap.parse_args()
d = np.load(DATA / ("delay_study_quick.npz" if a.quick else "delay_study.npz"))
S = {k: d[k] for k in KEYS}
valid, success, infeasible, silent = outcomes(S)
rates, delays = sorted(set(S["rate"])), sorted(set(S["delay"]))
n_seeds = len(set(S["seed"]))

fig, axes = plt.subplots(2, len(rates) + 1, figsize=(16, 7.6), gridspec_kw=dict(width_ratios=[1] * len(rates) + [1.25]))
total = len(delays) * len(rates) * n_seeds  # episodes per policy and mode
for row, (pv, mode) in enumerate(((True, "P: load previewed"), (False, "M: load measured, no preview"))):
    for col, r in enumerate(rates):
        ax = axes[row, col]
        for p, m in POLICY_MARGINS:
            label, color, ls = STYLE[p]
            y = [silent[valid & (S["policy"] == p) & (S["rate"] == r) & (S["preview"] == pv) & (S["delay"] == dl)].mean()
                 for dl in delays]
            ax.plot(1e3 * np.array(delays), y, ls, color=color, lw=1.6, marker="o", ms=4, label=f"{label} (m = {m:g})")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([1e3 * dl for dl in delays])
        ax.set_xlabel("sensing delay (ms)")
        if col == 0:
            ax.set_ylabel(f"{mode}\nsilent-slip rate")
        ax.set_title(f"({'abcdefgh'[row * (len(rates) + 1) + col]}) {mode.split(':')[0]}, load rate {r:g}x "
                     f"(ramp {3 / r:g} s)", loc="left", fontsize=10)
    ax = axes[row, -1]
    left = np.zeros(len(POLICY_MARGINS))
    for c in CAUSES:
        n = np.array([np.sum(valid & (S["policy"] == p) & (S["preview"] == pv) & S["fail"] & (S["cause"] == c))
                      for p, _ in POLICY_MARGINS])
        ax.barh(range(len(n)), n, left=left, color=CAUSE_COLOR[c], edgecolor="white", linewidth=1.5,
                label=CAUSE_LABEL[c])
        left += n
    ax.set_yticks(range(len(POLICY_MARGINS)), [STYLE[p][0] for p, _ in POLICY_MARGINS], fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, total)  # same scale in both rows: the P row is nearly empty
    n_slips = int(np.sum(valid & (S["preview"] == pv) & S["fail"]))
    if n_slips < 10:  # too few to see at this scale: say it
        ax.text(0.97, 0.04, f"{n_slips} slip(s) of {total * len(POLICY_MARGINS)} episodes", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color=MUTED)
    ax.set_xlabel(f"slips by onset cause (of {total} episodes per policy)")
    ax.set_title(f"({'abcdefgh'[row * (len(rates) + 1) + len(rates)]}) why the slips began", loc="left", fontsize=10)
axes[0, -1].legend(fontsize=7.5, labelcolor=MUTED, loc="upper right")
axes[0, len(rates) - 1].legend(fontsize=7.5, labelcolor=MUTED, loc="upper left")
fig.suptitle(f"S4: sensing delay with the S3 margins (scenario A, {n_seeds} pad/load seeds, squeeze capped at the "
             "pad's working range)\nOverlapping lines: in P all policies sit at 0; in M the oracle, the plug-in and the "
             "F/T + mask baseline sit at 1 (except F/T + mask at 0 ms, 1x)", x=0.01, y=1.02, ha="left")
fig.tight_layout()
save(fig, "s4_delay_quick.png" if a.quick else "s4_delay.png", pdf_stem=None if a.quick else "delay_envelope")
