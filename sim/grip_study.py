"""S3 study: every capacity estimator in the same grip controller, over pad/load seeds and margins.

Episodes (grip.episode) run in a process pool and are cached in sim/data/. The squeeze is capped at
the pad's working range of each seed (grip.squeeze_limit, the force budget), so every episode stays in
the valid range of the contact model. Outcomes of an episode:
    success      no slip and no flag
    infeasible   the policy requested more than the budget (flag) before any slip: by its own estimate
                 it cannot carry the load, and a real controller would stop or replan
    silent slip  slip (> grip.SLIP_FAIL) without an earlier flag
Effort is the mean squeeze relative to the oracle at M_DYN (same seed and scenario), over successful
episodes. M_DYN = 0.05 is the dynamic margin the oracle itself needs (control period, gripper lag).
Reported per scenario: outcomes and median effort per policy and margin; effort on matched seeds (all
compared policies succeed); and, for the nominal estimators, the margin tuned on half of scenario A
and evaluated on the other half and on scenario B, then tuned on the flat pads of A and evaluated on
the curved pads (shift of the contact geometry instead of the load). Decision times are measured under the parallel
load of the pool (about 4x the serial times).
Run: pixi run -e sim sim-grip [--quick] [--recompute]; the figure is made by plot_grip.py.
"""

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import grip
from policies import CERTIFIED_POLICIES, POLICIES

DATA = Path(__file__).parent / "data"
KEYS = ("seed", "family", "policy", "margin", "scenario", "d_max", "rho", "fail", "slip", "f_mean", "f_peak",
        "solve", "walls", "outside", "saturated", "completed", "f_max", "t_flag", "t_slip")
ORACLE_MARGINS = (0.0, 0.05, 0.1)
M_DYN = 0.05
REF = ("oracle", M_DYN)  # normalizer of the squeeze
CERTIFIED = CERTIFIED_POLICIES
NOMINAL = ("plugin4", "plugin16", "mask_uniform", "uniform")
TRAIN = 15  # seeds < TRAIN of scenario A tune the margins of the nominal estimators


def jobs(seeds, margins, limits, scenarios=("A", "B")):
    """Episode arguments (seed, policy, margin, scenario, f_max) of the study.

    The oracle runs only at ORACLE_MARGINS (it is the reference, not a candidate); every other policy
    runs at every margin. limits: force budget per seed (squeeze_limits).
    """
    out = []
    for sc in scenarios:
        for seed in seeds:
            out += [(seed, "oracle", m, sc, limits[seed]) for m in ORACLE_MARGINS]
            out += [(seed, p, m, sc, limits[seed]) for p in POLICIES[1:] for m in margins]
    return out


def _run(job):
    """One episode from a job tuple (a module-level function, so the process pool can pickle it)."""
    seed, policy, margin, scenario, f_max = job
    return grip.episode(seed, policy, margin, scenario, f_max=f_max)


def run(seeds, margins, cache, recompute=False):
    """All the episodes of the study as a dict of arrays (keys KEYS, one entry per episode).

    Loaded from the .npz cache if it exists (the cache does not record seeds and margins: delete it,
    or pass recompute, after changing them). Otherwise the episodes run on all cores, one per task.
    """
    if cache.exists() and not recompute:
        d = np.load(cache)
        return {k: d[k] for k in KEYS}
    todo = jobs(seeds, margins, squeeze_limits(seeds))
    rows, tic = [], time.time()
    # longest first (dome seeds: most contact faces; uniform/plugin4: largest plug-in measures) so that
    # no long episode starts last and keeps one core busy alone at the end
    todo.sort(key=lambda j: (grip.sample_episode(j[0]).family != "dome", j[1] not in ("uniform", "plugin4")))
    with Pool(os.cpu_count()) as pool:
        for i, r in enumerate(pool.imap_unordered(_run, todo, chunksize=1)):
            rows.append(r)
            if (i + 1) % 50 == 0 or i + 1 == len(todo):
                print(f"  {i + 1}/{len(todo)} episodes, {time.time() - tic:.0f} s", flush=True)
    S = {k: np.array([r[k] for r in rows]) for k in KEYS}
    DATA.mkdir(exist_ok=True)
    np.savez(cache, **S)
    return S


def squeeze_limits(seeds):
    """grip.squeeze_limit per seed (cached in sim/data/squeeze_limits.npz)."""
    path = DATA / "squeeze_limits.npz"
    known = {}
    if path.exists():
        d = np.load(path)
        known = dict(zip(d["seeds"].tolist(), d["limits"].tolist()))
    todo = [s for s in seeds if s not in known]
    if todo:
        with Pool(os.cpu_count()) as pool:
            known.update(zip(todo, pool.map(grip.squeeze_limit, todo)))
        DATA.mkdir(exist_ok=True)
        np.savez(path, seeds=np.array(list(known)), limits=np.array(list(known.values())))
    return {s: known[s] for s in seeds}


def normalized_squeeze(S):
    """f_mean / f_mean of the reference oracle episode with the same seed and scenario."""
    ref = {(sc, s): f for sc, s, p, m, f in zip(S["scenario"], S["seed"], S["policy"], S["margin"], S["f_mean"])
           if (p, m) == REF}
    return np.array([f / ref[(sc, s)] for sc, s, f in zip(S["scenario"], S["seed"], S["f_mean"])])


def outcomes(S):
    """Boolean arrays (valid, success, infeasible, silent) per episode."""
    valid = (S["walls"] <= 0.005) & (S["outside"] <= 0.005)
    flagged = ~np.isnan(S["t_flag"])
    silent = S["fail"] & ~(S["t_flag"] < S["t_slip"])  # nan compares False: a slip without flag is silent
    return valid, ~S["fail"] & ~flagged, flagged & ~silent, silent


def matched(S, eff, cases, sc, seeds=None):
    """Median effort of each (policy, margin) in cases over the seeds where all of them succeed."""
    valid, success, _, _ = outcomes(S)
    ok = None
    for p, m in cases:
        sel = valid & success & (S["scenario"] == sc) & (S["policy"] == p) & (S["margin"] == m)
        s = set(S["seed"][sel].tolist())
        ok = s if ok is None else ok & s
    if seeds is not None:
        ok &= set(seeds)
    out = []
    for p, m in cases:
        sel = (S["scenario"] == sc) & (S["policy"] == p) & (S["margin"] == m) & np.isin(S["seed"], list(ok))
        out.append(np.median(eff[sel]) if sel.any() else np.nan)
    return out, len(ok)


def tuned_margin(S, p, train):
    """Smallest margin without any slip on the training episodes (boolean mask; None if none)."""
    valid = outcomes(S)[0]
    for m in sorted(set(S["margin"][S["policy"] == p])):
        sel = valid & train & (S["policy"] == p) & (S["margin"] == m)
        if sel.any() and not S["fail"][sel].any():
            return m
    return None


def splits(S):
    """(name, training mask, test masks per scenario) for tuning the nominal margins."""
    A, held = S["scenario"] == "A", S["seed"] >= TRAIN
    curved = S["family"] != "flat"
    return ((f"seeds < {TRAIN} of A", A & ~held, {"A": held, "B": np.ones(len(A), bool)}),
            ("flat pads of A", A & ~curved, {"A": curved, "B": curved}))


def table(S):
    """Print the study tables (markdown): outcomes and effort per scenario, policy and margin, the
    certified silent slips, silent slips per pad family at M_DYN, matched-seed effort, and the
    tuned-margin transfer of the nominal estimators for each split of `splits`."""
    eff = normalized_squeeze(S)
    valid, success, infeasible, silent = outcomes(S)
    print(f"rejected episodes (side walls or contact outside the taxels > 0.5%): {np.sum(~valid)} of {len(valid)}")
    print(f"force budget (pad working range) per seed: {S['f_max'].min():.0f}-{S['f_max'].max():.0f} N")
    for sc in ("A", "B"):
        print(f"\nscenario {sc}: success / infeasible (flagged) / silent slip (%) and median effort of the successes "
              f"(squeeze / oracle at m = {M_DYN:g})")
        margins = sorted(set(S["margin"][S["policy"] != "oracle"]))
        print("| policy | " + " | ".join(f"m = {m:g}" for m in margins) + " | decision (ms) |")
        print("|---|" + "---|" * (len(margins) + 1))
        for p in POLICIES:
            cells = []
            for m in (ORACLE_MARGINS if p == "oracle" else margins):
                sel = valid & (S["scenario"] == sc) & (S["policy"] == p) & (S["margin"] == m)
                if not sel.any():
                    cells.append("")
                    continue
                ok = sel & success
                cells.append(f"{ok.sum() / sel.sum():.0%}/{infeasible[sel].mean():.0%}/{silent[sel].mean():.0%} "
                             + (f"{np.median(eff[ok]):.2f}" if ok.any() else "-"))
            sel = valid & (S["scenario"] == sc) & (S["policy"] == p)
            name = p + (" (m = " + ", ".join(f"{m:g}" for m in ORACLE_MARGINS) + ")" if p == "oracle" else "")
            print(f"| {name} | " + " | ".join(cells) + f" | {1e3 * np.median(S['solve'][sel]):.0f} |")
        sel = valid & (S["scenario"] == sc) & np.isin(S["policy"], CERTIFIED)
        print(f"  certified policies, all margins: {silent[sel].sum()} silent slips, "
              f"{(infeasible[sel] & S['fail'][sel]).sum()} slips after a flag, of {sel.sum()} episodes")
        for fam in grip.FAMILIES:
            f = valid & (S["scenario"] == sc) & (S["margin"] == M_DYN) & (S["family"] == fam)
            print(f"  {fam}, m = {M_DYN:g}: silent slips per policy: " + ", ".join(
                f"{p} {silent[f & (S['policy'] == p)].mean():.0%}" for p in POLICIES[1:]))
        cases = [REF] + [(p, M_DYN) for p in CERTIFIED]
        med, n = matched(S, eff, cases, sc)
        print(f"  matched seeds at m = {M_DYN:g} (all certified policies succeed, {n} seeds): "
              + ", ".join(f"{p} {v:.2f}" for (p, _), v in zip(cases, med)))

    for name, train, tests in splits(S):
        print(f"\nnominal estimators: margin tuned on {name} (smallest margin without slip), evaluated on the other "
              f"episodes; slip % / infeasible % / effort (matched with MT16 at m = {M_DYN:g})")
        for p in NOMINAL:
            m = tuned_margin(S, p, train)
            if m is None:
                print(f"  {p}: no margin without slip on the training episodes")
                continue
            cells = []
            for sc, test in tests.items():
                sel = valid & test & (S["scenario"] == sc) & (S["policy"] == p) & (S["margin"] == m)
                (e_p, e_c), n = matched(S, eff, [(p, m), ("MT16", M_DYN)], sc, S["seed"][sel].tolist())
                cells.append(f"{sc}: {S['fail'][sel].mean():.0%} / {infeasible[sel].mean():.0%} / {e_p:.2f} "
                             f"vs MT16 {e_c:.2f} ({n} seeds)")
            print(f"  {p}, tuned m = {m:g}: " + "; ".join(cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="6 seeds (2 per pad family), margins 0, 0.05, 0.25, 1")
    ap.add_argument("--recompute", action="store_true", help="ignore the cached study")
    a = ap.parse_args()
    seeds, margins = (range(6), (0.0, M_DYN, 0.25, 1.0)) if a.quick else (range(30), (0.0, M_DYN, 0.1, 0.25, 0.5, 1.0))
    cache = DATA / ("grip_study_v2_quick.npz" if a.quick else "grip_study_v2.npz")
    table(run(seeds, margins, cache, a.recompute))
