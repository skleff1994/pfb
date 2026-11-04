"""H1: camera delay d_cam from a recorded press/tap session (hardware protocol, step H1).

    pixi run python -m hw.h1_latency SESSION [--fn-threshold 10] [--img-threshold 10]

Compares the ATI normal force (RT-host stamps) with the image activity against the background
(GPU-host grab stamps), with the session's d_cam forced to 0 so the raw grab delay is measured.
Requires a clock-synchronized session - there is deliberately no override: an unsynchronized
latency number is meaningless.
"""

import argparse
import json

import numpy as np

from . import frames, latency
from .h2_pilot import ATI_TOPIC, CONTACT_FRAME
from .io import Session


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session")
    p.add_argument("--baseline", type=float, nargs=2, default=None,
                   help="unloaded interval [t0 t1] (s, absolute stamps); default: first 2 s of the ATI series")
    p.add_argument("--fn-threshold", type=float, default=10.0, help="force onset, in baseline noise units")
    p.add_argument("--img-threshold", type=float, default=10.0, help="image onset, in baseline noise units")
    p.add_argument("--ati-frame", default="ati")
    p.add_argument("--lag-window", type=float, nargs=2, default=(0.3, 3.0), metavar=("PRE", "POST"),
                   help="lag scan is restricted to [onset-PRE, onset+POST] around each force onset; "
                        "POST must span ramp + hold + release + d_cam, or truncating the release "
                        "biases the lag (the image trails the force, so a cut is asymmetric)")
    p.add_argument("--out", default=None, help="write the result as JSON")
    args = p.parse_args()

    s = Session(args.session)
    s.require_synchronized()

    t_f, W = s.wrench(ATI_TOPIC)
    R, t = s.static_transform(CONTACT_FRAME, args.ati_frame)
    fn = frames.transform_wrench(W, R, t)[:, 1]
    t_i, act = latency.image_activity(s.images(d_cam=0.0), s.background())
    base = args.baseline or (t_f[0], t_f[0] + 2.0)

    zf = latency.baseline_normalize(t_f, fn, base)
    zi = latency.baseline_normalize(t_i, act, base)
    on_f = latency.rising_crossings(t_f, zf, args.fn_threshold)
    on_i = latency.rising_crossings(t_i, zi, args.img_threshold)
    delays = latency.pair_onsets(on_f, on_i)
    # Only the press intervals: a whole-recording rank correlation is swamped by the unloaded
    # baseline and returns the argmax of noise (see latency.estimate_lag).
    pre, post = args.lag_window
    windows = [(o - pre, o + post) for o in on_f] or None
    lag, _, rho = latency.estimate_lag(t_i, act, t_f, np.abs(fn), windows=windows)

    frame_dt = float(np.median(np.diff(t_i)))
    result = {
        "session": s.dir.name,
        "d_cam_lag_scan_s": lag, "lag_scan_max_rho": float(np.nanmax(rho)),
        "lag_scan_windows": len(windows) if windows else 0,
        "onset_delays_s": delays.tolist(), "n_force_onsets": int(len(on_f)), "n_paired": int(len(delays)),
        "onset_delay_median_s": float(np.median(delays)) if len(delays) else None,
        "onset_delay_iqr_s": (np.percentile(delays, [25, 75]).tolist() if len(delays) else None),
        "frame_period_s": frame_dt,
        "frame_gaps": int(np.sum(np.diff(t_i) > 1.5 * frame_dt)),
        # The two estimators must agree; the runbook's gate. Reported so it is not eyeballed.
        "lag_scan_vs_onset_median_s": (abs(lag - float(np.median(delays))) if len(delays) else None),
        "chrony": {k: s.manifest["timing"].get(k) for k in ("chrony_start", "chrony_end")},
    }
    print(json.dumps({k: v for k, v in result.items() if k not in ("onset_delays_s", "chrony")}, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
