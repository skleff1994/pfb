"""H3: friction coefficient from pure-shear slides (hardware protocol, step H3).

    pixi run -e hw python -m hw.h3_mu SESSION... --out h3.json

Slides are recorded by panda_ws apps/planar_friction (`friction.py h3`). Each one is bracketed by events:
trial_start (its parameters in `info`), slip (the online gross-slip declaration; absent if the slide
ended without slip) and slide_end (emitted after the HOLD that stops the slide, plus 0.5 s).
The ATI wrench (sensor_wrench_corrected, at the gel-surface origin, as in h2_pilot) gives f_t along the
slide direction and f_n = |F_y|. Per slide:
    mu_kinetic  median |f_t| / f_n between the slip declaration and the HOLD (gross sliding: the
                procedure continues post_slip_travel past the declaration);
    mu_static   peak |f_t| / f_n before the HOLD (elastomer static friction can exceed kinetic).
Slides without a slip event are listed and excluded. The value used by the certificate is the minimum over
the operating range of the per-(f_n, speed) medians, with the spread.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .h2_pilot import surface_wrenches
from .io import Session

HOLD_SETTLE = 0.5   # [s] the procedure waits this long after the HOLD before emitting slide_end
MIN_FN = 0.2        # [N] samples below are not in contact


def slide_mu(t, W, direction_deg, t_start, t_slip, t_end):
    """(mu_static, mu_kinetic) of one slide from surface wrenches W (N, 6) at times t; mu_kinetic is NaN
    without a slip time."""
    d = np.radians(direction_deg)
    ft = np.abs(np.cos(d) * W[:, 0] + np.sin(d) * W[:, 2])
    fn = np.abs(W[:, 1])
    hold = t_end - HOLD_SETTLE
    ok = fn > MIN_FN
    run = ok & (t >= t_start) & (t <= hold)
    mu_static = float(np.max(ft[run] / fn[run])) if run.any() else float("nan")
    if t_slip is None:
        return mu_static, float("nan")
    gross = ok & (t >= t_slip) & (t <= hold)
    return mu_static, float(np.median(ft[gross] / fn[gross])) if gross.any() else float("nan")


def slides(session):
    """Per slide: trial parameters and the start / slip / end times, from the events."""
    out, cur = [], None
    for e in session.events():
        info = e.get("info") if isinstance(e.get("info"), dict) else {}
        if e["label"] == "trial_start" and info.get("test") == "h3":
            cur = {"params": info, "t_start": e["stamp"], "t_slip": None}
        elif cur is not None and e["label"] == "slip":
            cur["t_slip"] = e["stamp"]
        elif cur is not None and e["label"] == "slide_end":
            cur["t_end"] = e["stamp"]
            out.append(cur)
            cur = None
    return out


def analyse(session):
    t, W = surface_wrenches(session)
    rows = []
    for s in slides(session):
        p = s["params"]
        ms, mk = slide_mu(t, W, p["direction"], s["t_start"], s["t_slip"], s["t_end"])
        rows.append({"session": session.dir.name, "t_start": s["t_start"], "slipped": s["t_slip"] is not None,
                     "mu_static": ms, "mu_kinetic": mk,
                     **{k: p[k] for k in ("normal_force", "direction", "speed", "repeat", "indenter", "position") if k in p}})
    return rows


def summarize(rows):
    """Per (normal force, speed): median and quartiles of mu_kinetic; mu_min over the range; per-session drift
    (last minus first slipped slide)."""
    groups = defaultdict(list)
    for r in rows:
        if r["slipped"] and np.isfinite(r["mu_kinetic"]):
            groups[(r["normal_force"], r["speed"])].append(r["mu_kinetic"])
    table = [{"normal_force": f, "speed": v, "n": len(m), "median": float(np.median(m)),
              "q25": float(np.percentile(m, 25)), "q75": float(np.percentile(m, 75))}
             for (f, v), m in sorted(groups.items())]
    drift = {}
    for name in sorted({r["session"] for r in rows}):
        mk = [r["mu_kinetic"] for r in sorted(rows, key=lambda r: r["t_start"])
              if r["session"] == name and r["slipped"] and np.isfinite(r["mu_kinetic"])]
        if len(mk) >= 2:
            drift[name] = mk[-1] - mk[0]
    all_mk = [m for ms in groups.values() for m in ms]
    return {"by_load": table,
            "mu_min": min(g["median"] for g in table) if table else None,
            "spread_std": float(np.std(all_mk)) if all_mk else None,
            "drift_last_minus_first": drift,
            "excluded_no_slip": sum(not r["slipped"] for r in rows)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = [r for s in args.sessions for r in analyse(Session(s))]
    summary = summarize(rows)
    Path(args.out).write_text(json.dumps({"slides": rows, "summary": summary}, indent=2))
    for g in summary["by_load"]:
        print(f"f_n {g['normal_force']:4.1f} N  v {1e3 * g['speed']:.2f} mm/s  n {g['n']:2d}  "
              f"mu {g['median']:.3f} [{g['q25']:.3f}, {g['q75']:.3f}]")
    print(f"mu_min {summary['mu_min']}  spread {summary['spread_std']}  "
          f"excluded (no slip) {summary['excluded_no_slip']} -> {args.out}")


if __name__ == "__main__":
    main()
