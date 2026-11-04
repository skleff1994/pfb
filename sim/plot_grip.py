"""Figure S3: closed-loop grip control in Drake, same controller with different capacity estimates.

(a, b) Scenarios A and B: silent-slip rate (a slip the policy did not flag) against the median squeeze
relative to the oracle at m = M_DYN, one curve per policy over its margin m (open marker: m = 0). The
squeeze is the median over successful episodes; margins without any success are drawn at the left edge.
(c) Outcomes at m = M_DYN, both scenarios: success, infeasible (the policy flagged that the load needs
more than the pad's force budget) and silent slip.
(d) Certified M_T: squeeze relative to the oracle against taxel resolution, on matched seeds (all
certified policies and the oracle succeed at m = M_DYN), with the infeasible rate of each resolution.
Run after grip_study.py: pixi run -e sim sim-grip-plot [--quick]
Figures: examples/figures/s3_grip.png and $PFB_DOC_FIG/grip_pareto.pdf.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from _style import COLOR, MUTED, plt, save  # noqa: E402,F401

from viz import OUTCOME, STYLE  # noqa: E402
from grip_study import CERTIFIED, DATA, KEYS, M_DYN, REF, matched, normalized_squeeze, outcomes  # noqa: E402


ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--quick", action="store_true")
a = ap.parse_args()
d = np.load(DATA / ("grip_study_v2_quick.npz" if a.quick else "grip_study_v2.npz"))
S = {k: d[k] for k in KEYS}
eff = normalized_squeeze(S)
valid, success, infeasible, silent = outcomes(S)

LEFT = 0.8  # x position of the margins without any success
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, sc, title in zip(axes[0], ("A", "B"), ("(a) A: monotone load (D1-like)", "(b) B: varying load direction")):
    for p, (label, color, ls) in STYLE.items():
        sel = valid & (S["scenario"] == sc) & (S["policy"] == p)
        ms = sorted(set(S["margin"][sel]))
        x = [np.median(eff[sel & (S["margin"] == m) & success]) if (sel & (S["margin"] == m) & success).any()
             else LEFT for m in ms]
        y = [silent[sel & (S["margin"] == m)].mean() for m in ms]
        ax.plot(x, y, ls, color=color, lw=1.5, label=label)
        ax.plot(x[1:], y[1:], "o", color=color, ms=4)
        ax.plot(x[:1], y[:1], "o", mfc="white", mec=color, ms=7, mew=1.5)
    ax.set_xscale("log")
    ax.set_xlim(LEFT * 0.9, None)
    ticks = [t for t in (0.8, 1, 1.5, 2, 3, 5, 8) if t <= ax.get_xlim()[1]]
    ax.set_xticks(ticks, ["no success" if t == LEFT else f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel(f"median squeeze / oracle squeeze (m = {M_DYN:g})")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title, loc="left", fontsize=10)
    ax.annotate("certified policies: no silent slip at any margin", (0.98, 0.07), xycoords="axes fraction",
                ha="right", fontsize=8, color=MUTED)
axes[0, 0].set_ylabel("silent-slip rate (slip > 0.2 mm, not flagged)")
axes[0, 1].legend(fontsize=7.5, labelcolor=MUTED, loc="upper right")

ax = axes[1, 0]
pols = list(STYLE)
at = valid & (S["margin"] == M_DYN)
rates = np.array([[o[at & (S["policy"] == p)].mean() for o in (success, infeasible, silent)] for p in pols])
left = np.zeros(len(pols))
for j, (name, color) in enumerate(OUTCOME):
    ax.barh(range(len(pols)), rates[:, j], left=left, color=color, edgecolor="white", linewidth=2, label=name)
    left += rates[:, j]
ax.set_yticks(range(len(pols)), [STYLE[p][0] for p in pols], fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("share of episodes (scenarios A and B)")
ax.grid(axis="y", visible=False)
ax.legend(fontsize=8, labelcolor=MUTED, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
ax.set_title(f"(c) outcomes at m = {M_DYN:g} (the oracle's dynamic margin)", loc="left", fontsize=10)

ax = axes[1, 1]
ks = [2, 4, 8, 16]
for sc, mk in (("A", "o"), ("B", "s")):
    cases = [REF] + [(p, M_DYN) for p in CERTIFIED]
    med, n = matched(S, eff, cases, sc)
    ax.plot(ks, med[1:5], mk + "-", color=COLOR["MT"], ms=6, label=f"certified $\\mathcal{{M}}_T$, scenario {sc} ({n} seeds)")
    for p, m_ in zip(("MF16", "MT16_onepad"), med[5:]):
        ax.plot([16], [m_], mk, color=STYLE[p][1], mfc="white" if p == "MF16" else STYLE[p][1], ms=7, mew=1.5)
ax.plot([], [], "o", color=COLOR["MF"], mfc="white", label="certified $\\mathcal{M}_F$ 16x16")
ax.plot([], [], "o", color=COLOR["MT"], label="certified, one pad sensed (16x16)")
for k in ks:
    sel = at & (S["policy"] == f"MT{k}")
    ax.annotate(f"infeasible {infeasible[sel].mean():.0%}", (k, 1.0), xytext=(0, -14), textcoords="offset points",
                ha="center", fontsize=7.5, color=MUTED)
ax.axhline(1, color=COLOR["truth"], lw=1, label=f"oracle (m = {M_DYN:g})")
ax.set_xscale("log", base=2)
ax.set_xticks(ks, [f"{k}x{k}" for k in ks])
ax.minorticks_off()
ax.set_xlabel("taxels per pad")
ax.set_ylabel("median squeeze / oracle squeeze")
ax.set_ylim(0.8, None)
ax.legend(fontsize=7.5, labelcolor=MUTED, loc="upper right")
ax.set_title(f"(d) certified squeeze vs resolution, matched seeds, m = {M_DYN:g}", loc="left", fontsize=10)

fig.suptitle("Closed-loop grip control in Drake: same controller, different capacity estimates "
             f"({len(set(S['seed']))} pad/load seeds per scenario, squeeze capped at the pad's working range)",
             x=0.01, y=1.0, ha="left")
fig.tight_layout()
save(fig, "s3_grip.png" if not a.quick else "s3_grip_quick.png", pdf_stem=None if a.quick else "grip_pareto")
