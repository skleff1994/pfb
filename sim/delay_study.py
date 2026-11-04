"""S4 study: how much sensing delay the S3 grip controller tolerates, with the margins fixed in S3.

Same scene, controller and force budget as S3 (grip.episode), scenario A (the D1-like ramp). Varied:
    delay    sensing delay of the tactile state and the gripper force (0-200 ms)
    rate     load-rate factor (the ramp lasts 3 / rate s)
    preview  P: the load profile is known ahead (the S3 loop; only the contact state is stale)
             M: the load is measured with the same delay, no preview (D1: a load growing in an unknown way)
Policies at their S3 margins, not retuned: the point is transfer. The oracle and the certificates use
the oracle's dynamic margin M_DYN; the nominal estimators use the margin tuned on the S3 training
seeds (grip_study, split by seed).
Outcomes as in S3 (grip_study.outcomes). Effort is the mean squeeze relative to the oracle at
delay 0 with preview, same seed and rate. Each slip is diagnosed at its onset (grip.diagnose): its
capacity shortfall splits into estimate, load, state and actuator terms; the cause is the largest
(or model, when the ideal Coulomb capacity still exceeds the load).
Run: pixi run -e sim sim-delay [--quick] [--recompute]; the figure is made by plot_delay.py.
"""

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

import grip
from grip_study import DATA, KEYS as S3_KEYS, M_DYN, outcomes, squeeze_limits

POLICY_MARGINS = (("oracle", M_DYN), ("MT8", M_DYN), ("MT16", M_DYN), ("MT16_onepad", M_DYN),
                  ("plugin16", 0.1), ("mask_uniform", 0.25))  # S3: m_dyn, and the margins tuned on A seeds < 15
DELAYS = (0.0, 0.025, 0.05, 0.1, 0.2)  # s: 30-60 fps tactile frames + processing, up to a slow reconstruction
RATES = (1.0, 2.0, 4.0)
MODES = (True, False)  # preview (P), measured load (M)
KEYS = S3_KEYS + ("delay", "preview", "rate", "t_onset", "onset_ratio", "cause") + tuple(f"dlog_{k}" for k in grip.CAUSE_TERMS + ("slack",))
CAUSES = ("model",) + grip.CAUSE_TERMS + ("stale",)  # stale: no load measured yet at the last decision


def jobs(seeds, limits, delays=DELAYS, rates=RATES):
    """Episode arguments (seed, policy, margin, delay, rate, preview, f_max) of the study."""
    return [(seed, p, m, d, r, pv, limits[seed]) for seed in seeds for p, m in POLICY_MARGINS
            for d in delays for r in rates for pv in MODES]


def _run(job):
    """One scenario-A episode from a job tuple (module level, so the process pool can pickle it)."""
    seed, policy, margin, delay, rate, preview, f_max = job
    return grip.episode(seed, policy, margin, "A", f_max=f_max, delay=delay, rate=rate, preview=preview)


def run(seeds, cache, recompute=False, **grid):
    """All episodes as a dict of arrays (KEYS), from the cache or computed on all cores (see grip_study.run)."""
    if cache.exists() and not recompute:
        d = np.load(cache)
        return {k: d[k] for k in KEYS}
    todo = jobs(seeds, squeeze_limits(seeds), **grid)
    # longest first: dome seeds (most faces), slow loads (longest episodes)
    todo.sort(key=lambda j: (grip.sample_episode(j[0]).family != "dome", j[4]))
    rows, tic = [], time.time()
    with Pool(os.cpu_count()) as pool:
        for i, r in enumerate(pool.imap_unordered(_run, todo, chunksize=1)):
            rows.append(r)
            if (i + 1) % 100 == 0 or i + 1 == len(todo):
                print(f"  {i + 1}/{len(todo)} episodes, {time.time() - tic:.0f} s", flush=True)
    S = {k: np.array([r[k] for r in rows]) for k in KEYS}
    DATA.mkdir(exist_ok=True)
    np.savez(cache, **S)
    return S


def effort(S):
    """f_mean relative to the oracle at delay 0 with preview, same seed and rate (NaN if missing)."""
    ref = {(s, r): f for s, r, p, d, pv, f in zip(S["seed"], S["rate"], S["policy"], S["delay"], S["preview"],
                                                   S["f_mean"]) if p == "oracle" and d == 0 and pv}
    return np.array([f / ref.get((s, r), np.nan) for s, r, f in zip(S["seed"], S["rate"], S["f_mean"])])


def envelope(S, silent, valid, policy, rate, preview, tol=0.0):
    """Largest delay (s) up to which the silent-slip rate stays <= tol at every delay (-1 if not even at 0)."""
    best = -1.0
    for d in sorted(set(S["delay"])):
        sel = valid & (S["policy"] == policy) & (S["rate"] == rate) & (S["preview"] == preview) & (S["delay"] == d)
        if silent[sel].mean() > tol:
            break
        best = d
    return best


def table(S):
    """Print, per mode and rate: success / infeasible / silent % and median effort per policy and delay;
    the envelope of each policy; and the onset causes of the slips per mode and policy."""
    eff = effort(S)
    valid, success, infeasible, silent = outcomes(S)
    print(f"rejected episodes (side walls or contact outside the taxels > 0.5%): {np.sum(~valid)} of {len(valid)}")
    delays = sorted(set(S["delay"]))
    for pv in MODES:
        name = "P: load previewed (S3 loop), tactile state delayed" if pv else "M: load measured with the delay, no preview"
        for r in sorted(set(S["rate"])):
            print(f"\n{name}; load rate {r:g}x (ramp {3 / r:g} s): success / infeasible / silent slip (%) and median "
                  "effort of the successes (squeeze / oracle at delay 0, preview)")
            print("| policy (margin) | " + " | ".join(f"{1e3 * d:g} ms" for d in delays) + " | silent-free up to |")
            print("|---|" + "---|" * (len(delays) + 1))
            for p, m in POLICY_MARGINS:
                cells = []
                for d in delays:
                    sel = valid & (S["policy"] == p) & (S["rate"] == r) & (S["preview"] == pv) & (S["delay"] == d)
                    ok = sel & success
                    cells.append(f"{ok.sum() / sel.sum():.0%}/{infeasible[sel].mean():.0%}/{silent[sel].mean():.0%} "
                                 + (f"{np.nanmedian(eff[ok]):.2f}" if ok.any() else "-"))
                env = envelope(S, silent, valid, p, r, pv)
                print(f"| {p} ({m:g}) | " + " | ".join(cells) + " | "
                      + ("none" if env < 0 else f"{1e3 * env:g} ms") + " |")
        print(f"\n{name}: onset causes of the slips (all rates and delays)")
        for p, _ in POLICY_MARGINS:
            sel = valid & (S["policy"] == p) & (S["preview"] == pv) & S["fail"]
            counts = {c: int(np.sum(S["cause"][sel] == c)) for c in CAUSES}
            if not sel.any():
                print(f"  {p}: no slip")
                continue
            terms = ", ".join(f"{k} {np.nanmedian(S[f'dlog_{k}'][sel]):+.2f}" for k in grip.CAUSE_TERMS + ("slack",))
            print(f"  {p}: {sel.sum()} slips, " + ", ".join(f"{c} {n}" for c, n in counts.items())
                  + f"; median capacity / demand at onset {np.nanmedian(S['onset_ratio'][sel]):.2f}; "
                  f"median log terms: {terms}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="6 seeds (2 per pad family)")
    ap.add_argument("--recompute", action="store_true", help="ignore the cached study")
    a = ap.parse_args()
    seeds = range(6) if a.quick else range(30)
    cache = DATA / ("delay_study_quick.npz" if a.quick else "delay_study.npz")
    table(run(seeds, cache, a.recompute))
