"""H1: camera delay d_cam and measurement ages.

The GelSight image is stamped at grab time on the GPU host; the ATI at receipt on the RT host; both
clocks are chrony-synchronized. A press appears in the ATI force at its physical time and in the
images d_cam later (exposure, USB, decode), so

    d_cam = image onset (grab stamps) - force onset (ATI stamps)

Two estimators, to be compared on the H1 recording (about 20 slow presses and sharp taps):
    onsets         per-press threshold crossings, interpolated; a per-press distribution. Linear
                   interpolation between frames is only valid when the activity ramps over several
                   frames: for a step-like tap that completes within one frame, the interpolated
                   image onset is biased EARLY by up to one frame period (40 ms). Use slow ramps
                   (>= 0.3 s) for onsets; for taps, rely on estimate_lag.
    estimate_lag   one lag for the recording: the lag maximizing the rank correlation between image
                   activity and the delayed force (any monotone force -> activity relation). Pass
                   `windows` covering the presses; over a whole sparse recording the correlation is
                   swamped by the unloaded baseline (see the note on estimate_lag).
"""

import numpy as np
from scipy import stats


def rising_crossings(t, x, threshold, refractory=0.5):
    """Times where x crosses threshold upwards (linear interpolation), at least `refractory` s apart."""
    t, x = np.asarray(t, float), np.asarray(x, float)
    i = np.nonzero((x[:-1] < threshold) & (x[1:] >= threshold))[0]
    tc = t[i] + (threshold - x[i]) * (t[i + 1] - t[i]) / (x[i + 1] - x[i])
    keep, last = [], -np.inf
    for c in tc:
        if c - last >= refractory:
            keep.append(c)
            last = c
    return np.asarray(keep)


def baseline_normalize(t, x, baseline_window):
    """|x - median over the baseline window| / its MAD-based noise level, so thresholds are in noise units."""
    t, x = np.asarray(t, float), np.asarray(x, float)
    m = (t >= baseline_window[0]) & (t <= baseline_window[1])
    med = np.median(x[m])
    noise = 1.4826 * np.median(np.abs(x[m] - med)) or 1e-12
    return np.abs(x - med) / noise


def image_activity(images, background_bgr, roi=None):
    """Mean absolute difference to the background per frame: (t, activity). images yields (t, seq, rgb)."""
    bg = background_bgr[..., ::-1].astype(np.float32)  # to RGB, the recorded encoding
    if roi is not None:
        bg = bg[roi]
    t, a = [], []
    for ti, _, img in images:
        im = img.astype(np.float32)
        if roi is not None:
            im = im[roi]
        t.append(ti)
        a.append(np.abs(im - bg).mean())
    return np.asarray(t), np.asarray(a)


def pair_onsets(t_force, t_image, max_lag=0.3):
    """Delays image - force for each force onset that has an image onset within [0, max_lag]."""
    t_image = np.asarray(t_image)
    out = []
    for tf in t_force:
        c = t_image[(t_image >= tf - 0.02) & (t_image <= tf + max_lag)]
        if len(c):
            out.append(c[0] - tf)
    return np.asarray(out)


def estimate_lag(t_image, activity, t_force, force, lags=np.arange(-0.05, 0.25, 0.001), windows=None):
    """Lag d maximizing Spearman correlation between activity(t) and force(t - d). Returns (best, lags, rho).

    `windows` is an iterable of (t0, t1) intervals (in the image time base) restricting which samples
    are correlated. Pass the press intervals: under the H1 protocol (about 20 presses with >= 2 s
    unloaded between them) only ~25% of frames carry signal, and a rank correlation over the whole
    recording is dominated by baseline noise. It then peaks at a near-zero - commonly negative -
    value and the returned lag is the argmax of noise rather than a delay. Measured on a real
    session: whole recording gave rho_max = -0.048 at 0.144 s, the same data restricted to the
    press windows gave rho_max = +0.788 at 0.114 s, agreeing with the onset median to 10 ms.
    """
    t_image, activity = np.asarray(t_image, float), np.asarray(activity, float)
    if windows is not None:
        keep = np.zeros(len(t_image), bool)
        for t0, t1 in windows:
            keep |= (t_image >= t0) & (t_image <= t1)
        if keep.sum() < 3:
            raise ValueError("windows select %d image samples, too few to correlate" % int(keep.sum()))
        t_image, activity = t_image[keep], activity[keep]
    rho = np.array([stats.spearmanr(activity, np.interp(t_image - d, t_force, force)).statistic for d in lags])
    return float(lags[np.nanargmax(rho)]), lags, rho


def measurement_age(session, topic):
    """Age at the recorder = bag receive time - header stamp [s], per message (processing + transport)."""
    ages = [t_bag - (m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec) for t_bag, m in session._messages(topic)]
    return np.asarray(ages)
