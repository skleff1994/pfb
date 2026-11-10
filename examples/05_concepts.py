"""Conceptual figures for the theory note, computed with pfb.

  measurements.pdf      what M_0, M_F and M_T know about the same contact
  vertex_reduction.pdf  lower bound = concentrate, upper bound = spread to the corners (lem:taxel_reduction)
  load_factor_concept.pdf  the load factors of the certificate: reach along a ray, tightest half-plane, decision vs measurement
  closure.pdf           ghost masses on taxel corners and the CoP-preserving push (boundary push, thm:combined_lower/upper);
                        not used by the theory note at the moment

Vector PDFs are written to $PFB_DOC_FIG (see _style.py), and PNG previews to examples/figures/.
"""

import numpy as np

import pfb
from _style import COLOR, MUTED, plt, save
from pfb.certify import PadTerm, load_factor_lower, load_factor_true, load_factor_upper

INK, GHOST = COLOR["truth"], "#e34948"
plt.rcParams.update({"font.size": 8, "axes.grid": False, "axes.titlesize": 8.5})

mu = 0.5
patch = pfb.Patch(1.0, 0.6, N=48)
nu = (pfb.gaussian(patch, (0.3, 0.35), 0.12, fN=1.5)
      + pfb.gaussian(patch, (0.75, 0.2), 0.08, fN=0.5))
n = patch.N + 1
X, Y = (patch.points[:, k].reshape(n, n) for k in (0, 1))
rho = nu.w / patch.quad
VMAX = rho.max()


def pressure(ax, values, alpha=1.0):
    ax.pcolormesh(X, Y, values.reshape(n, n), shading="nearest", cmap="Blues", vmin=0, vmax=VMAX,
                  alpha=alpha, rasterized=True)


def frame(ax, title):
    ax.set_aspect("equal")
    ax.set_xlim(-0.02, patch.Lx + 0.02); ax.set_ylim(-0.02, patch.Ly + 0.02)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left")


def grid_lines(ax, taxels, lw=0.6, color=MUTED):
    for lo, hi in zip(taxels.lo, taxels.hi):
        ax.add_patch(plt.Rectangle(lo, *(hi - lo), fill=False, ec=color, lw=lw))


def taxel_cells(ax, taxels, F):
    """Taxel image: each cell filled with its mean pressure F_i / |P_i|, on the same colour scale."""
    cmap = plt.get_cmap("Blues")
    for lo, hi, Fi in zip(taxels.lo, taxels.hi, F):
        area = np.prod(hi - lo)
        ax.add_patch(plt.Rectangle(lo, *(hi - lo), fc=cmap(Fi / area / VMAX), ec=MUTED, lw=0.6))


def cop_marker(ax, c, label="CoP $c$"):
    ax.plot(*c, "x", color=INK, ms=7, mew=2, label=label)


def taxel_barycenters(measure, taxels):
    F = taxels.downsample(measure)
    m = np.stack([np.bincount(taxels.index, weights=measure.w * measure.patch.points[:, k], minlength=taxels.r)
                  for k in (0, 1)], axis=1)
    return m / F[:, None]


# --- Fig. 1: what each measurement knows -------------------------------------------------
tx = pfb.TaxelGrid(patch, 4, 4)
F = tx.downsample(nu)
area = (patch.Lx / tx.nx) * (patch.Ly / tx.ny)
lo_c, hi_c = pfb.cop_set_MF(tx, F)

fig, axes = plt.subplots(1, 3, figsize=(7.5, 1.9))
ax = axes[0]
pressure(ax, rho, alpha=0.25)
ax.add_patch(plt.Rectangle((0, 0), patch.Lx, patch.Ly, fill=False, ec=INK, lw=1.4))
cop_marker(ax, nu.cop)
ax.text(0.03, 0.55, f"known: $f_N={nu.fN:.1f}$, $c$\nunknown: the rest", va="top", fontsize=7.5)
frame(ax, "$\\mathcal{M}_0$: total force + CoP")
for ax, with_cop in zip(axes[1:], (False, True)):
    taxel_cells(ax, tx, F)
    for i in range(tx.r):
        x, y = (tx.lo[i] + tx.hi[i]) / 2
        ax.text(x, y, f"{F[i]:.2f}", ha="center", va="center", fontsize=6,
                color="white" if F[i] / area > 0.55 * VMAX else INK)
    if with_cop:
        cop_marker(ax, nu.cop)
    else:
        ax.add_patch(plt.Rectangle(lo_c, *(hi_c - lo_c), fill=False, ec=COLOR["MF"], lw=1.4, ls="--"))
        ax.text(lo_c[0] - 0.01, lo_c[1], "CoP", color=COLOR["MF"], ha="right", va="top")
axes[1].set_title("$\\mathcal{M}_F$: taxel forces $F_i$, CoP unknown", loc="left")
axes[2].set_title("$\\mathcal{M}_T$: taxel forces $F_i$ + CoP", loc="left")
for ax in axes[1:]:
    frame(ax, ax.get_title(loc="left"))
fig.tight_layout(w_pad=0.8)
save(fig, "measurements.png", pdf_stem="measurements")

# --- Fig. 2: vertex reduction ---------------------------------------------------------------
tx = pfb.TaxelGrid(patch, 3, 3)
fN, m1, F = pfb.measure_MT(nu, tx)
c = m1 / fN
q = pfb.shift_twist([0.0, 0.0, 1.0], c)  # unit rotation about the CoP: g(x) = ||x - c||, a bowl centred at the CoP
h = pfb.support(nu, q, mu)
lo_val, pts = pfb.socp_lower_MT(tx, F, m1, q, mu, return_points=True)
up_val, lam = pfb.vertex_lp_MT(tx, F, m1, q, mu, return_weights=True)
V = pfb.taxel_vertices(tx)
bary = taxel_barycenters(nu, tx)
size = lambda m: 260 * m / fN  # marker area proportional to force

fig, axes = plt.subplots(1, 4, figsize=(7.5, 1.95), gridspec_kw=dict(width_ratios=[0.8, 1, 1, 1]))
# (a) one-dimensional picture of the three values at the same barycentre m
ax = axes[0]
g = lambda x: np.sqrt(0.03 + (x - 0.42) ** 2)
x = np.linspace(0, 1, 200)
x1, x2 = 0.2, 0.75
m = (x1 + x2) / 2
chord = lambda t: (1 - t) * g(0) + t * g(1)
ax.plot(x, g(x), color=INK, lw=1.4)
ax.plot([0, 1], [g(0), g(1)], color=COLOR["MT"], lw=1.2)
ax.plot([x1, x2], [g(x1), g(x2)], color=MUTED, lw=0.8, ls=":")
ax.plot([m] * 3, [g(m), (g(x1) + g(x2)) / 2, chord(m)], color=MUTED, lw=0.6)
ax.plot(m, chord(m), "o", color=COLOR["MT"], ms=4)
ax.plot(m, (g(x1) + g(x2)) / 2, "o", color=INK, ms=4)
ax.plot(m, g(m), "o", color=COLOR["MT"], mfc="white", ms=4)
ax.plot([x1, x2], [0, 0], "o", color=INK, ms=3, clip_on=False)
ax.text(m - 0.03, chord(m) + 0.04, "spread to $v_{i1},v_{i2}$", ha="right", va="bottom", fontsize=6.5, color=COLOR["MT"])
ax.text(m + 0.03, (g(x1) + g(x2)) / 2, "actual", va="center", fontsize=6.5)
ax.text(m + 0.03, g(m) - 0.02, "concentrate at $u_i/F_i$", va="top", fontsize=6.5, color=COLOR["MT"])
ax.set_xticks([0, x1, m, x2, 1], ["$v_{i1}$", "$x$", "$u_i/F_i$", "$x'$", "$v_{i2}$"])
ax.set_yticks([]); ax.set_ylim(0, 0.75)
ax.spines["left"].set_visible(False)
ax.set_title("(a) convex $g$ on a taxel", loc="left")
# (b) setup
ax = axes[1]
pressure(ax, rho)
grid_lines(ax, tx)
ax.contour(X, Y, np.hypot(X - c[0], Y - c[1]), levels=6, colors=INK, linewidths=0.4, alpha=0.6)
cop_marker(ax, c)
frame(ax, "(b) $\\nu^\\star$ and level sets of $g$")
ax.text(0.5, -0.06, f"true $h={h:.3f}$ (rotation about the CoP)", ha="center", va="top", transform=ax.transAxes, fontsize=6.5)
# (c) lower bound: one point per taxel, pulled toward the CoP
ax = axes[2]
pressure(ax, rho, alpha=0.2)
grid_lines(ax, tx)
for b0, p1, Fi in zip(bary, pts, F):
    if np.linalg.norm(p1 - b0) > 1e-3:
        ax.annotate("", p1, b0, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.6))
ax.scatter(*pts.T, s=size(F), color=COLOR["MT"], ec="white", lw=0.5, zorder=3)
cop_marker(ax, c)
frame(ax, "(c) lower bound: concentrate")
ax.text(0.5, -0.06, f"$\\underline{{h}}={lo_val:.3f}$, one point per taxel", ha="center", va="top",
        transform=ax.transAxes, fontsize=6.5)
# (d) upper bound: weights on the corners, pushed away from the CoP
ax = axes[3]
pressure(ax, rho, alpha=0.2)
grid_lines(ax, tx)
keep = lam > 1e-9 * fN
ax.scatter(*V[keep].T, s=size(lam[keep]), color=COLOR["MT"], ec="white", lw=0.5, zorder=3, clip_on=False)
cop_marker(ax, c)
frame(ax, "(d) upper bound: spread to corners")
ax.text(0.5, -0.06, f"$\\overline{{h}}={up_val:.3f}$, {keep.sum()} nonzero weights $\\leq r+2={tx.r + 2}$", ha="center", va="top",
        transform=ax.transAxes, fontsize=6.5)
fig.tight_layout(w_pad=0.6)
save(fig, "vertex_reduction.png", pdf_stem="vertex_reduction")

# --- closure figure: ghost masses (boundary push) ---------------------------------------------
tx = pfb.TaxelGrid(patch, 2, 1)
fN, m1, F = pfb.measure_MT(nu, tx)
c = m1 / fN
q = np.array([0.0, 1.0, 1.0]) / np.sqrt(2)  # twist about the CoP; rotation centre p = c + J a / b, left of the patch
print("Fig 3: rotation centre p =", c + pfb.J @ q[:2] / q[2])
up_val, lam = pfb.vertex_lp_MT(tx, F, m1, pfb.shift_twist(q, c), mu, return_weights=True)
V = pfb.taxel_vertices(tx)
# A corner belongs to its own taxel unless it lies on an open (right or top, non-last) side.
ix, iy = np.arange(tx.r) % tx.nx, np.arange(tx.r) // tx.nx
own = np.ones((tx.r, 4), bool)
own[:, [1, 2]] &= (ix == tx.nx - 1)[:, None]  # corners with x = hi_x
own[:, [2, 3]] &= (iy == tx.ny - 1)[:, None]  # corners with y = hi_y
keep = lam > 1e-9 * fN
p_int = taxel_barycenters(nu, tx)  # interior points with sum F_i p_i = m_1 (local CoPs)
eps = 0.25
T = (1 - eps) * V + eps * p_int[:, None, :]
val_eps = mu * np.sum(lam * pfb.integrand(T.reshape(-1, 2), q, c).reshape(tx.r, 4))
cop_eps = np.einsum("ik,ikd->d", lam, T) / fN
bound = eps * abs(q[2]) * mu * F @ tx.diam


def taxels_with_open_sides(ax):
    for i in range(tx.r):
        (x0, y0), (x1, y1) = tx.lo[i], tx.hi[i]
        right_open, top_open = ix[i] < tx.nx - 1, iy[i] < tx.ny - 1
        ax.plot([x0, x1], [y0, y0], color=INK, lw=1)
        ax.plot([x0, x0], [y0, y1], color=INK, lw=1)
        ax.plot([x1, x1], [y0, y1], color=INK, lw=1, ls=(0, (3, 2)) if right_open else "-", alpha=0.6 if right_open else 1)
        ax.plot([x0, x1], [y1, y1], color=INK, lw=1, ls=(0, (3, 2)) if top_open else "-")
        ax.text((x0 + x1) / 2, y1 + 0.03, f"$P_{i + 1}$, $F_{i + 1}={F[i]:.2f}$", ha="center", fontsize=7)
    x_shared = tx.hi[0, 0]
    ax.text(x_shared, -0.03, f"edge $x_1={x_shared:.1f}$ belongs to $P_2$", ha="center", va="top", fontsize=6.5, color=MUTED)


fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.3))
for ax in axes:
    pressure(ax, rho, alpha=0.2)
    taxels_with_open_sides(ax)
    cop_marker(ax, c)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.08, 1.08); ax.set_ylim(-0.08, 0.72)
    for s in ax.spines.values():
        s.set_visible(False)
ax = axes[0]
ok, gh = keep & own, keep & ~own
ax.scatter(*V[ok].T, s=size(lam[ok]), color=COLOR["MT"], ec="white", lw=0.5, zorder=3, clip_on=False)
ax.scatter(*V[gh].T, s=size(lam[gh]), facecolor="none", ec=GHOST, lw=1.4, zorder=3, clip_on=False)
for v in V[gh]:
    ax.annotate("ghost: counted in $F_1$,\nread by $P_2$", v, v + np.array([0.08, 0.1 if v[1] < 0.3 else -0.16]),
                fontsize=6.5, color=GHOST, arrowprops=dict(arrowstyle="->", color=GHOST, lw=0.6))
ax.set_title(f"(a) vertex LP optimum, value $\\overline{{h}}={up_val:.4f}$", loc="left")
ax = axes[1]
for i, k in zip(*np.nonzero(keep)):
    ax.annotate("", T[i, k], V[i, k], arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7))
ax.scatter(*T[keep].T, s=size(lam[keep]), color=COLOR["MT"], ec="white", lw=0.5, zorder=3)
ax.plot(*p_int.T, "+", color=COLOR["MT"], ms=7, mew=1.5)
for i, pi in enumerate(p_int):
    ax.text(pi[0] + 0.02, pi[1] + 0.02, f"$p_{i + 1}$", color=COLOR["MT"], fontsize=7)
ax.set_title(f"(b) slide toward $p_i$ ($\\varepsilon={eps}$): value {val_eps:.4f}", loc="left")
ax.text(0.5, -0.13, f"CoP unchanged: $|c_\\varepsilon-c|={np.linalg.norm(cop_eps - c):.0e}$;   "
        f"$|\\Delta|={abs(up_val - val_eps):.4f}\\leq\\varepsilon|b|\\mu\\sum_iF_i\\,\\mathrm{{diam}}P_i={bound:.4f}$",
        ha="center", va="top", transform=ax.transAxes, fontsize=6.5)
fig.tight_layout(w_pad=0.5)
save(fig, "closure.png", pdf_stem="closure")


# --- Fig. 4: the load factors (prop:load_factor) ------------------------------------------------
# A load w = (f_x, 0, tau) about the pad centre o is a ray in the 3D wrench space; the plane f_y = 0 slices
# the wrench sets, so the load factor s(w) is the distance from the origin to the boundary of the slice along w.
# Plot coordinates: (f_x, tau / ell) with a fixed lever ell, so that both axes are forces.
UNDEC = "#c9c8c3"
HOLD, SLIP, CUT = COLOR["MT"], GHOST, "#8a4fd6"  # decision zones; supporting half-planes
o = np.array([patch.Lx / 2, patch.Ly / 2])
ell = 0.4
tx4, tx16 = pfb.TaxelGrid(patch, 4, 4), pfb.TaxelGrid(patch, 16, 16)
fN, m1, F4 = pfb.measure_MT(nu, tx4)
F16, c = tx16.downsample(nu), m1 / fN
pad4 = PadTerm.from_taxels(tx4, F4, cop=c)
atoms = nu.w > 0
w_dir = lambda t: np.array([np.cos(t), 0.0, ell * np.sin(t)])  # (f_x, f_y, tau), plotted at (f_x, tau / ell)
true_factor = lambda w: load_factor_true(patch.points[atoms], nu.w[atoms], w, mu, o)

# (a) radial functions: s_lo (guaranteed set X-), true s* and s_up (possible set X+) in 72 directions of the plane
ts = np.linspace(0, 2 * np.pi, 73)
S = np.array([(load_factor_lower(pad4, w_dir(t), mu, o), true_factor(w_dir(t)), load_factor_upper(pad4, w_dir(t), mu, o))
              for t in ts])
assert np.all(S[:, 0] <= S[:, 1] + 1e-5) and np.all(S[:, 1] <= S[:, 2] + 1e-5), "s_lo <= s* <= s_up violated"
print(f"Fig 4: s_lo/s* in [{(S[:, 0] / S[:, 1]).min():.2f}, {(S[:, 0] / S[:, 1]).max():.2f}], "
      f"s_up/s* in [{(S[:, 2] / S[:, 1]).min():.2f}, {(S[:, 2] / S[:, 1]).max():.2f}] over the plane")
slice_pts = [np.c_[S[:, k] * np.cos(ts), S[:, k] * np.sin(ts)] for k in range(3)]  # X-, truth, X+

t0 = np.radians(50)  # the load w0 used in all panels
w0, d0 = w_dir(t0), np.array([np.cos(t0), np.sin(t0)])
s_lo, q_lo = load_factor_lower(pad4, w0, mu, o, return_q=True)
s_up, q_up = load_factor_upper(pad4, w0, mu, o, return_q=True)
s_true = true_factor(w0)


def supporting_line(ax, q, h, **kw):
    """Line <q, (f_x, 0, tau)> = h in the plot plane: a half-plane that contains the slice."""
    n = np.array([q[0], q[2] * ell])
    p0, d = h * n / (n @ n), np.array([-n[1], n[0]]) / np.linalg.norm(n)
    ax.plot(*np.c_[p0 - 4 * d, p0 + 4 * d], **kw)


def slices(ax, alpha=1.0, lw=1.4, labels=False):
    """True set (grey), possible set X+ (solid) and guaranteed set X- (dashed) in the plane f_y = 0."""
    ax.fill(*slice_pts[1].T, color="#e4e3df", alpha=alpha, zorder=0)
    ax.plot(*slice_pts[2].T, color=INK, lw=lw, alpha=alpha)
    ax.plot(*slice_pts[0].T, color=INK, lw=lw, alpha=alpha, ls="--")
    lim = 1.1 * np.abs(np.r_[slice_pts[2], slice_pts[1]]).max()
    ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color=MUTED, lw=0.4); ax.axvline(0, color=MUTED, lw=0.4)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("$f_x$"); ax.set_ylabel("$\\tau/L_0$")
    if labels:  # direct labels on the curves
        k1, k2 = (np.argmin(abs(ts - np.radians(d))) for d in (140, 270))
        ax.text(*(slice_pts[2][k1] + [-0.05, 0.04]), "$\\mathcal{X}^+$", ha="right", va="bottom", fontsize=7.5)
        ax.text(*(slice_pts[0][k2] * 0.62), "$\\mathcal{X}^-$", ha="center", va="center", fontsize=7.5)

fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.2), gridspec_kw=dict(width_ratios=[1.15, 1.15, 1]))
ax = axes[0]
slices(ax, labels=True)
for a0, a1, col in ((0, s_lo, HOLD), (s_lo, s_up, UNDEC), (s_up, 1.45 * s_up, SLIP)):  # hold / undecided / slip zones
    ax.plot(*np.c_[a0 * d0, a1 * d0], color=col, lw=3, solid_capstyle="butt", zorder=3)
ax.plot(*(s_lo * d0), "o", color=HOLD, mec=INK, mew=0.6, ms=5, zorder=4)
ax.plot(*(s_true * d0), "o", color=INK, ms=5, zorder=4)
ax.plot(*(s_up * d0), "o", color="white", mec=INK, ms=5, zorder=4)
perp = np.array([d0[1], -d0[0]])  # to the right of the ray
for s_mark, lab in ((s_lo, "$\\underline{s}$"), (s_true, "$s^\\star$"), (s_up, "$\\overline{s}$")):
    ax.text(*(s_mark * d0 - 0.11 * perp), lab, fontsize=8, ha="right", va="center")
for sc, lab in ((0.6 * s_lo, "A hold"), (0.5 * (s_lo + s_true), "B undecided"), (1.3 * s_up, "C slip")):
    ax.plot(*(sc * d0), "D", color=INK, ms=3, zorder=4)
    ax.text(*(sc * d0 + 0.07 * perp), lab, fontsize=6.5, ha="left", va="center")
ax.set_title("(a) ray view: load factor = reach", loc="left")

# (b) the same load factors as the tightest supporting half-plane: s_up = min over <q, w0> = 1 of h_up(q)
ax = axes[1]
slices(ax, alpha=0.45, lw=0.9)
ax.plot(*np.c_[0 * d0, 1.45 * s_up * d0], color=MUTED, lw=1)
n0 = np.array([w0[2], 0.0, -w0[0]]) / np.hypot(w0[2], w0[0])  # keeps <q, w0> = 1 when added to q
for step in (-0.5, -0.25, 0.25, 0.5):  # other directions: each half-plane contains X+ but cuts the ray farther out
    q = q_up + step * np.linalg.norm(q_up) * n0
    q = q / (q @ w0)
    h = float(np.atleast_1d(pfb.reduced_MT(tx4, (fN, m1, F4), pfb.shift_twist(q, o), mu)[1])[0])
    supporting_line(ax, q, h, color=CUT, lw=0.7, alpha=0.8)
    ax.plot(*(h * d0), "x", color=CUT, ms=3.5, mew=0.8, zorder=4)
supporting_line(ax, q_up, s_up, color=CUT, lw=1.6)
supporting_line(ax, q_lo, s_lo, color=CUT, lw=1.6, ls="--")
ax.plot(*(s_up * d0), "o", color="white", mec=INK, ms=5, zorder=4)
ax.plot(*(s_lo * d0), "o", color=HOLD, mec=INK, mew=0.6, ms=5, zorder=4)
ax.set_title("(b) dual view: tightest half-plane", loc="left")

# (c) the decision band shrinks with the measurement (information order of the operators)
ax = axes[2]
pads = [("$\\mathcal{M}_0$", PadTerm.from_taxels(pfb.TaxelGrid(patch, 1, 1), [fN], cop=c)),
        ("$\\mathcal{M}_F$ 4x4", PadTerm.from_taxels(tx4, F4)),
        ("$\\mathcal{M}_T$ 4x4", pad4),
        ("$\\mathcal{M}_T$ 16x16", PadTerm.from_taxels(tx16, F16, cop=c))]
xmax = 3.0
for r, (name, pad) in enumerate(pads):
    lo, up = (f(pad, w0, mu, o) / s_true for f in (load_factor_lower, load_factor_upper))
    y = len(pads) - 1 - r
    for a0, a1, col in ((0, lo, HOLD), (lo, min(up, xmax), UNDEC), (min(up, xmax), xmax, SLIP)):
        ax.barh(y, a1 - a0, left=a0, height=0.55, color=col, lw=0)
ax.axvline(1, color=INK, lw=1)
ax.text(1.03, len(pads) - 0.45, "true capacity", fontsize=6.5, va="bottom")
ax.set_yticks(range(len(pads)), [n for n, _ in pads][::-1]); ax.set_xlim(0, xmax); ax.set_ylim(-0.5, len(pads) - 0.2)
ax.set_xlabel("demanded load / true capacity $s^\\star$")
ax.set_title("(c) decision vs measurement", loc="left")

from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch as Swatch  # noqa: E402
handles = [Swatch(fc="#e4e3df", label="true wrench set $\\mathcal{X}(\\nu^\\star)$"),
           Line2D([], [], color=INK, label="possible set $\\mathcal{X}^+$ (solid)"),
           Line2D([], [], color=INK, ls="--", label="guaranteed set $\\mathcal{X}^-$ (dashed)"),
           Line2D([], [], color=CUT, label="half-plane of $\\mathcal{X}^+$, tightest one thick"),
           Line2D([], [], color=CUT, ls="--", label="half-plane of $\\mathcal{X}^-$"),
           Line2D([], [], marker="x", color=CUT, ls="", ms=4, label="cut of the ray by a half-plane"),
           Line2D([], [], marker="o", color=HOLD, mec=INK, mew=0.6, ls="", ms=5, label="$\\underline{s}$ (certified)"),
           Line2D([], [], marker="o", color=INK, ls="", ms=5, label="$s^\\star$ (true)"),
           Line2D([], [], marker="o", color="white", mec=INK, ls="", ms=5, label="$\\overline{s}$ (upper)"),
           Swatch(fc=HOLD, label="hold"), Swatch(fc=UNDEC, label="undecided"), Swatch(fc=SLIP, label="slip")]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.5, columnspacing=1.4, handlelength=1.6)
fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=0.6)
save(fig, "load_factor_concept.png", pdf_stem="load_factor_concept")
