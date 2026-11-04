"""H2 pilot: image registration and depth -> pressure fits from recorded press sessions.

    # 1. single-boss presses: fit the image -> surface registration
    pixi run -e hw python -m hw.h2_pilot registration SESSION... --out reg.json
    # 2. multi-boss presses: boss forces from the ATI, depth from the images, fit the maps
    pixi run -e hw python -m hw.h2_pilot fit SESSION... --registration reg.json --n-bosses 3 --out h2_fit

Dwell windows come from the legacy shear-calibration state (E1a normal-force staircase: the last
--hold-tail seconds of each HOLD state) or from operator events (--event-label dwell_end).
The ATI wrench is /ati_ft_sensor/sensor_wrench_corrected (sensor frame, no CoP shift), expressed
here at the gel-surface origin with the static transform from /tf_static. Depth is reconstructed
offline (hw.depth) and averaged over the frames of each window. Needs GS_SDK_PATH (see hw.depth).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from . import boss_solve, frames, pressure_fit
from .depth import OfflineReconstructor
from .io import Session

ATI_TOPIC = "/ati_ft_sensor/sensor_wrench_corrected"
CONTACT_FRAME = "gelsight_contact_surface"
SHEAR_STATE_TOPIC = "/shear_calib/state"
HOLD_STATES = {4, 6, 8, 12}  # ShearTiltCalibrator HOLD_POSITIVE, HOLD_NEGATIVE, HOLD_ZERO, HOLD_SEGMENT


def dwell_windows(session, event_label=None, hold_tail=1.0):
    """[(t0, t1)] analysis windows: the last hold_tail seconds of each hold or before each event."""
    if event_label:
        return [(e["stamp"] - hold_tail, e["stamp"]) for e in session.events() if e["label"] == event_label]
    t, state = session.scalar(SHEAR_STATE_TOPIC)
    hold = np.isin(np.rint(state).astype(int), list(HOLD_STATES))
    windows, start = [], None
    for k in range(len(t)):
        if hold[k] and start is None:
            start = k
        if start is not None and (not hold[k] or k == len(t) - 1):
            end = k - 1 if not hold[k] else k
            if t[end] - t[start] >= hold_tail:
                windows.append((t[end] - hold_tail, t[end]))
            start = None
    return windows


def surface_wrenches(session, ati_frame="ati"):
    """ATI wrench series at the gel-surface origin, in contact-frame coordinates."""
    t, W = session.wrench(ATI_TOPIC)
    R, p = session.static_transform(CONTACT_FRAME, ati_frame)
    return t, frames.transform_wrench(W, R, p)


def window_data(session, windows, recon, ati_frame="ati"):
    """Per window: mean surface wrench, mean depth map and mask (majority over the frames), frame count."""
    t_w, W = surface_wrenches(session, ati_frame)
    acc = [{"depth": None, "mask": None, "n": 0} for _ in windows]
    starts = np.array([w[0] for w in windows])
    for t, _, img in session.images(t0=starts.min() if len(starts) else None, t1=max(w[1] for w in windows)):
        for k, (t0, t1) in enumerate(windows):
            if t0 <= t <= t1:
                s = recon.surface(img)
                a = acc[k]
                a["depth"] = s.depth_mm.astype(np.float64) if a["depth"] is None else a["depth"] + s.depth_mm
                a["mask"] = s.mask.astype(int) if a["mask"] is None else a["mask"] + s.mask
                a["n"] += 1
    out = []
    for (t0, t1), a in zip(windows, acc):
        m = (t_w >= t0) & (t_w <= t1)
        if a["n"] == 0 or not m.any():
            continue
        out.append({"t0": t0, "t1": t1, "wrench": W[m].mean(axis=0), "wrench_std": W[m].std(axis=0),
                    "depth": a["depth"] / a["n"], "mask": a["mask"] * 2 >= a["n"], "n_frames": a["n"]})
    return out


def check_sign(data):
    fy = np.array([d["wrench"][1] for d in data])
    if np.median(fy) > 0:
        print("WARNING: compressive presses give F_y > 0 at the surface; check the frame convention")


def run_registration(sessions, args):
    px, pts = [], []
    for s in sessions:
        recon = OfflineReconstructor(s, device=args.device)
        data = window_data(s, dwell_windows(s, args.event_label, args.hold_tail), recon, args.ati_frame)
        check_sign(data)
        for d in data:
            regions = pressure_fit.label_regions(d["mask"], args.min_px)
            if not regions:
                continue
            region, centroid, _ = regions[0]
            y_c = -1e-3 * float(np.median(d["depth"][region]))  # contact plane: indentation below the surface
            px.append(centroid)
            pts.append(boss_solve.cop(d["wrench"], y_c=y_c))
    M, rms = boss_solve.fit_registration(px, pts)
    out = {"M": M.tolist(), "rms_m": rms, "n_presses": len(px), "pixels": np.asarray(px).tolist(),
           "cop_m": np.asarray(pts).tolist()}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"registration from {len(px)} presses: rms {1e3 * rms:.3f} mm -> {args.out}")


def run_fit(sessions, args):
    reg = json.loads(Path(args.registration).read_text())
    M = np.asarray(reg["M"])
    samples, rows = [], []
    for s in sessions:
        recon = OfflineReconstructor(s, device=args.device)
        data = window_data(s, dwell_windows(s, args.event_label, args.hold_tail), recon, args.ati_frame)
        check_sign(data)
        for d in data:
            regions = pressure_fit.label_regions(d["mask"], args.min_px)[: args.n_bosses]
            if len(regions) < args.n_bosses:
                continue
            masks = [r for r, _, _ in regions]
            positions = boss_solve.apply_registration(M, [c for _, c, _ in regions])
            y_c = -1e-3 * float(np.median(d["depth"][np.any(masks, axis=0)]))
            sol = boss_solve.solve(positions, d["wrench"], y_c=y_c)
            F = -sol.F  # compression positive
            samples.append(pressure_fit.Sample(d["depth"], masks, F, recon.mm_per_px,
                                               {"session": s.dir.name, "t0": d["t0"], "t1": d["t1"]}))
            rows.append({"session": s.dir.name, "t0": d["t0"], "t1": d["t1"], "n_frames": d["n_frames"],
                         "F_boss": F.tolist(), "positions_m": positions.tolist(), "y_c_m": y_c,
                         "residual_N": sol.residual, "cond": sol.cond, "wrench": d["wrench"].tolist(),
                         "wrench_std": d["wrench_std"].tolist()})
    if len(samples) < 2:
        raise SystemExit(f"only {len(samples)} usable windows")
    results = {}
    train, test = samples[0::2], samples[1::2]
    for name, fitter in pressure_fit.FITTERS.items():
        fit_all = fitter(samples)
        fit_tr = fitter(train)
        test_res = np.concatenate([fit_tr.predict(x) - x.F for x in test])
        meas = np.concatenate([x.F for x in test])
        results[name] = {"params": fit_all.params, "rms_N": fit_all.rms, "rel_rms": fit_all.rel_rms,
                         "holdout_rms_N": float(np.sqrt(np.mean(test_res ** 2))),
                         "holdout_rel_rms": float(np.sqrt(np.mean(test_res ** 2)) / np.sqrt(np.mean(meas ** 2)))}
        print(f"{name:8s} rel rms {fit_all.rel_rms:.3f} (holdout {results[name]['holdout_rel_rms']:.3f})  {fit_all.params}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "windows.json").write_text(json.dumps(rows, indent=2))
    (out / "fits.json").write_text(json.dumps({"registration": args.registration, "n_samples": len(samples),
                                               "models": results}, indent=2))
    np.savez_compressed(out / "samples.npz", depth=np.stack([x.depth_mm for x in samples]),
                        regions=np.stack([np.stack(x.regions) for x in samples]),
                        F=np.stack([x.F for x in samples]), mm_per_px=samples[0].mm_per_px)
    print(f"{len(samples)} windows -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["registration", "fit"])
    p.add_argument("sessions", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--registration", help="registration JSON (mode fit)")
    p.add_argument("--n-bosses", type=int, default=3)
    p.add_argument("--event-label", default=None, help="use operator events instead of the shear-calibration state")
    p.add_argument("--hold-tail", type=float, default=1.0, help="analysis window at the end of each dwell [s]")
    p.add_argument("--min-px", type=int, default=50, help="smallest contact region kept [px]")
    p.add_argument("--ati-frame", default="ati")
    p.add_argument("--device", default="cpu")
    p.add_argument("--allow-unsynced", action="store_true",
                   help="accept sessions without clock synchronization (dwell averages only; never for latency)")
    args = p.parse_args()
    sessions = [Session(d) for d in args.sessions]
    for s in sessions:  # dwell windows pair RT-host ATI data with GPU-host images
        s.require_synchronized(args.allow_unsynced)
    if args.mode == "registration":
        run_registration(sessions, args)
    else:
        if not args.registration:
            p.error("--registration is required for mode fit")
        run_fit(sessions, args)


if __name__ == "__main__":
    main()
