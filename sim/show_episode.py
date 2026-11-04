"""Watch one S3 grip episode: Meshcat 3D replay, a dashboard figure and an animated dashboard.

Runs grip.episode with the same force budget as the study (the seed's pad working range) and shows
    Meshcat (live, and out/<name>.html)  the bar between the two pads, the hydroelastic contact force
                                         (red) and moment (blue) arrows of each pad, and the applied load
                                         force (blue shaft, drawn in front of the grasp)
    out/<name>.png                       dashboard at the key moment (first slip, else flag, else peak load)
    out/<name>.mp4                       the same dashboard over the whole episode (real time, 20 fps)
Dashboard: (a) applied demand D against the policy's estimate s_hat and the true capacity s_true of
the current contact (a sound estimate stays below s_true); (b) measured, commanded and requested
squeeze with the force budget; (c) slip at the pad faces; (d, e) the two pads seen from +y, Drake's
contact faces filled with their pressure, the policy's taxel grid, the CoP and, for a certificate,
its worst-case point loads (the compatible pressure that attains s_hat).
Run: pixi run -e sim sim-show [--seed 0] [--policy MT16] [--margin 0.05] [--scenario A]
                              [--delay 0 (ms)] [--rate 1] [--no-preview] [--no-meshcat] [--no-video] [--no-wait]
"""

import argparse
import time

import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

import grip
import viz
from grip_study import squeeze_limits
from policies import POLICIES, certificate

plt = viz.plt
C = {"D": "#0b0b0b", "s_hat": viz.COLOR["MT"], "s_true": "#7a5bd6", "budget": "#d03b3b", "flag": "#e0a030"}


def taxel_k(policy):
    """Taxels per side drawn on (pad 1, pad 2) for a policy; None where the policy uses no taxels."""
    if policy == "MT16_onepad":
        return 16, 1
    if policy == "uniform":
        return 1, 1
    for prefix in ("MT", "MF", "plugin"):
        if policy.startswith(prefix):
            k = int(policy[len(prefix):])
            return k, k
    return None, None


def outcome(r):
    """success / infeasible / silent slip, with the definitions of grip_study.outcomes."""
    flagged = not np.isnan(r["t_flag"])
    silent = r["fail"] and not (r["t_flag"] < r["t_slip"])
    return "silent slip" if silent else "infeasible (flagged)" if flagged else "success"


def key_frame(r):
    """Index of the control step to show in the static figure: first slip, else first flag, else peak demand."""
    lg = r["log"]
    for t in (r["t_slip"], r["t_flag"]):
        if not np.isnan(t):
            return int(np.argmin(np.abs(lg["t"] - t)))
    return int(np.argmax(lg["D_now"]))


def worst_cases(r, policy):
    """Certificate's worst-case point loads per control step (lists of per-pad arrays, or None)."""
    lg = r["log"]
    return [certificate(policy, st, w, grip.MU)[1:] for st, w in zip(lg["state"], lg["w"])]


class Dashboard:
    """Figure with the time series (top) and the two pad maps (bottom); draw(i) shows control step i."""

    def __init__(self, r, policy, worst):
        self.r, self.lg, self.policy, self.worst = r, r["log"], policy, worst
        self.k = taxel_k(policy)
        # one colour scale for the whole episode, so the maps can be compared over time
        self.vmax = 1.2e-3 * max((pad.forces / pad.areas).max() for st in self.lg["state"] for pad in st)
        # zoom on the contact (fixed over the episode): its extent plus 2 mm, at most the footprint
        extent = max(np.abs(pad.triangles).max() for st in self.lg["state"] for pad in st)
        self.view = min(extent + 0.002, grip.GRID / 2)
        self.fig = plt.figure(figsize=(14, 8.2))
        gs = self.fig.add_gridspec(2, 3, height_ratios=(1, 1.25), hspace=0.32, wspace=0.32,
                                   left=0.05, right=0.98, top=0.95, bottom=0.07)  # video frames are not cropped
        self.ax_load, self.ax_sq, self.ax_slip = (self.fig.add_subplot(gs[0, j]) for j in range(3))
        self.ax_pads = [self.fig.add_subplot(gs[1, j]) for j in range(2)]
        self.ax_info = self.fig.add_subplot(gs[1, 2])
        self._series()
        self._info()
        self.cursors = [ax.axvline(0, color=viz.MUTED, lw=1, ls=":") for ax in (self.ax_load, self.ax_sq, self.ax_slip)]
        self.colorbar = None

    def _series(self):
        """Time series of the whole episode (drawn once; a cursor marks the current step)."""
        lg, r = self.lg, self.r
        t = lg["t"]
        ax = self.ax_load
        ax.plot(t, lg["D_now"], color=C["D"], lw=1.5, label="demand D (applied)")
        ax.plot(t, lg["D"], color=C["D"], lw=1, ls="--", label="demand used by the command (" + ("lead" if self.r["preview"] else "measured, delayed") + ")")
        ax.plot(t, lg["s_true"], color=C["s_true"], lw=1.5, label="true capacity s_true")
        if self.policy != "oracle":
            ax.plot(t, lg["s_hat"], color=C["s_hat"], lw=1.5, label=f"policy estimate s_hat ({self.policy})")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("load along w (load factor)")
        ax.set_ylim(0, None)
        ax.legend(fontsize=7, labelcolor=viz.MUTED, loc="upper left")
        ax.set_title("(a) demand and capacity of the current contact", loc="left", fontsize=10)

        ax = self.ax_sq
        ax.plot(t, lg["f_meas"], color=C["D"], lw=1.5, label="measured squeeze")
        ax.plot(t, lg["f_cmd"], color=C["s_hat"], lw=1.2, label="command")
        req = np.where(np.isfinite(lg["f_req"]), lg["f_req"], np.nan)
        ax.plot(t, np.minimum(req, 1.5 * r["f_max"]), color=C["s_hat"], lw=1, ls="--", label="requested (before clipping)")
        ax.axhline(r["f_max"], color=C["budget"], lw=1, label=f"budget {r['f_max']:.0f} N")
        if not np.isnan(r["t_flag"]):
            ax.axvline(r["t_flag"], color=C["flag"], lw=1.5, label="flag: cannot certify within budget")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("squeeze per pad (N)")
        ax.set_ylim(0, 1.6 * r["f_max"] if np.any(lg["f_req"] > r["f_max"]) else 1.15 * max(lg["f_cmd"].max(), lg["f_meas"].max()))
        ax.legend(fontsize=7, labelcolor=viz.MUTED, loc="upper left")
        ax.set_title("(b) squeeze", loc="left", fontsize=10)

        ax = self.ax_slip
        ax.plot(t, 1e3 * lg["slip"], color=C["D"], lw=1.5)
        ax.axhline(1e3 * grip.SLIP_FAIL, color=C["budget"], lw=1, label=f"failure threshold {1e3 * grip.SLIP_FAIL:g} mm")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("slip at the pad faces (mm)")
        ax.set_ylim(0, max(1.3 * grip.SLIP_FAIL, 1.1 * lg["slip"].max()) * 1e3)
        ax.legend(fontsize=7, labelcolor=viz.MUTED, loc="upper left")
        ax.set_title("(c) slip", loc="left", fontsize=10)

    def _info(self):
        """Text panel: the seed, the policy and the outcome."""
        r, spec = self.r, grip.sample_episode(self.r["seed"])
        lever = "pure torque" if np.isinf(spec.lever) else f"lever {spec.lever:+.1f} x rho"
        lines = [f"seed {r['seed']} ({spec.family} pads), scenario {r['scenario']}",
                 f"load: theta = {np.rad2deg(spec.theta):.0f} deg, {lever}",
                 f"rho = {1e3 * r['rho']:.1f} mm, D_max = {r['d_max']:.3f}",
                 "", f"policy {self.policy}, margin m = {r['margin']:g}", f"force budget {r['f_max']:.0f} N",
                 f"sensing delay {1e3 * r['delay']:g} ms, load rate {r['rate']:g}x, "
                 + ("load previewed" if r["preview"] else "load measured"),
                 "", f"outcome: {outcome(r)}", f"max slip {1e3 * r['slip']:.3f} mm",
                 *([f"slip onset at {r['t_onset']:.2f} s: {r['cause']} (capacity/demand {r['onset_ratio']:.2f})"]
                   if r["cause"] else []),
                 f"mean squeeze {r['f_mean']:.1f} N, peak {r['f_peak']:.1f} N",
                 f"median decision time {1e3 * r['solve']:.0f} ms"]
        self.ax_info.axis("off")
        self.ax_info.text(0, 1, "\n".join(lines), va="top", fontsize=9, family="monospace", transform=self.ax_info.transAxes)
        self.ax_info.text(0, 0.02, "maps: contact faces filled with pressure, seen from +y;\n"
                          "blue grid: taxels; red +: CoP; orange circles: worst-case\n"
                          "point loads of the certificate (area ~ force)", va="bottom", fontsize=7.5,
                          color=viz.MUTED, transform=self.ax_info.transAxes)

    def draw(self, i):
        """Show control step i: redraw the two pad maps and move the cursors."""
        st, t = self.lg["state"][i], self.lg["t"][i]
        worst, wf = self.worst[i] if self.worst is not None else (None, None)
        img = None
        for j, (ax, pad) in enumerate(zip(self.ax_pads, st)):
            ax.clear()
            img = viz.draw_pad(ax, pad, grip.GRID / 2, k=self.k[j], worst=None if worst is None else worst[j],
                               worst_forces=None if wf is None else wf[j], vmax=self.vmax, view=self.view)
            ax.set_title(f"({'de'[j]}) pad {j + 1} ({'-' if j == 0 else '+'}y), t = {t:.2f} s, "
                         f"{pad.forces.sum():.1f} N", loc="left", fontsize=10)
        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(img, ax=self.ax_pads, label="pressure (kPa)", shrink=0.85)
        for c in self.cursors:
            c.set_xdata([t, t])
        return []


def main():
    """Parse the arguments, run the episode, then save the replay, the dashboard and the video."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default="MT16", choices=list(POLICIES))
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--scenario", default="A", choices=["A", "B"])
    ap.add_argument("--delay", type=float, default=0.0, help="sensing delay (ms, a multiple of 5; S4)")
    ap.add_argument("--rate", type=float, default=1.0, help="load-rate factor (S4)")
    ap.add_argument("--no-preview", action="store_true", help="load measured with the delay instead of previewed (S4)")
    ap.add_argument("--no-meshcat", action="store_true", help="no 3D viewer")
    ap.add_argument("--no-video", action="store_true", help="skip the animated dashboard")
    ap.add_argument("--no-wait", action="store_true", help="exit at the end instead of keeping the viewer alive")
    a = ap.parse_args()
    name = f"episode_{a.scenario}_s{a.seed}_{a.policy}_m{a.margin:g}"
    if a.delay or a.rate != 1 or a.no_preview:
        name += f"_d{a.delay:g}ms_r{a.rate:g}" + ("_measured" if a.no_preview else "")
    mc = None if a.no_meshcat else viz.start_meshcat()
    f_max = squeeze_limits([a.seed])[a.seed]
    tic = time.time()
    r = grip.episode(a.seed, a.policy, a.margin, a.scenario, record=True, f_max=f_max, meshcat=mc,
                     delay=a.delay / 1e3, rate=a.rate, preview=not a.no_preview)
    print(f"episode: {outcome(r)}, max slip {1e3 * r['slip']:.3f} mm, mean squeeze {r['f_mean']:.1f} N "
          f"(budget {f_max:.0f} N), {time.time() - tic:.0f} s")
    if mc is not None:
        viz.save_html(mc, r["scene"].visualizer, name + ".html")
    worst = worst_cases(r, a.policy)
    dash = Dashboard(r, a.policy, worst)
    dash.draw(key_frame(r))
    viz.OUT.mkdir(exist_ok=True)
    dash.fig.savefig(viz.OUT / f"{name}.png")
    print(f"saved {viz.OUT / name}.png")
    if not a.no_video:
        anim = FuncAnimation(dash.fig, dash.draw, frames=len(r["log"]["t"]), blit=False)
        fps = round(1 / grip.DT_CTRL)  # one frame per control step: real time
        try:
            anim.save(viz.OUT / f"{name}.mp4", writer=FFMpegWriter(fps=fps), dpi=90)
            print(f"saved {viz.OUT / name}.mp4")
        except (FileNotFoundError, RuntimeError):  # no ffmpeg: animated GIF instead
            anim.save(viz.OUT / f"{name}.gif", writer=PillowWriter(fps=fps), dpi=70)
            print(f"saved {viz.OUT / name}.gif")
    if mc is not None and not a.no_wait:
        viz.wait(mc)


if __name__ == "__main__":
    main()
