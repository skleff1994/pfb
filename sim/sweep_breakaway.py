"""Preconditions of the simulation study, on the bar break-away experiment (bar_breakaway.py).

For each set of numerical parameters: observed break-away factor s_obs vs the ideal Coulomb load factor
s_true of the simulated contact, the certified 16x16 M_T factor (both pads) relative to s_obs, the
side-wall force share, and the time of one two-pad certificate. Prints a markdown table.
Run: pixi run -e sim sim-sweep
"""

import time

import numpy as np

import bar_breakaway as bb
import pfb
from pfb.certify import PadTerm, load_factor_lower

BASE = dict(dt=1e-3, approx="lagged", hint=1e-3, stiction=1e-6, rate=2.0, v_slip=1e-3, modulus=5e5)
CASES = [
    ("reference (lagged, stiction 1e-6)", {}),
    ("time step 2.5e-4", dict(dt=2.5e-4)),
    ("ramp rate 0.5 N/s", dict(rate=0.5)),
    ("pad mesh hint 0.5 mm", dict(hint=5e-4)),
    ("slip threshold 0.3 mm/s", dict(v_slip=3e-4)),
    ("stiffer pads (modulus 2e6)", dict(modulus=2e6)),
    ("stiction 1e-5", dict(stiction=1e-5)),
    ("stiction 1e-4 (Drake default)", dict(stiction=1e-4)),
    ("similar, stiction 1e-6", dict(approx="similar")),
    ("sap (automatic regularization)", dict(approx="sap")),
]

print("| case | s_obs (N) | s_true (N) | s_obs / s_true | certified M_T 16x16 / s_obs | side walls |")
print("|---|---|---|---|---|---|")
for name, kw in CASES:
    r = bb.run(**{**BASE, **kw}, ks=(16,), verbose=False)
    print(f"| {name} | {r['s_obs']:.3f} | {r['s_true']:.3f} | {r['s_obs'] / r['s_true']:.3f} | "
          f"{r['rows'][0][1] / r['s_obs']:.3f} | {r['walls']:.2%} |")

# Time of one certificate for the two pads (M_T, 16x16), as the controller would solve it.
patch = pfb.Patch(bb.PAD[2], bb.PAD[0], N=64)
tx = pfb.TaxelGrid(patch, 16, 16)
offset = -np.array([bb.PAD[2], bb.PAD[0]]) / 2
nu = pfb.hertz(patch, (0.008, 0.010), 0.008)
pads = [PadTerm.from_taxels(tx, tx.downsample(nu), cop=nu.cop, t=offset)] * 2
w = np.array([1.0, 0.0, -bb.LEVER])
times = []
for _ in range(10):
    t = time.perf_counter()
    load_factor_lower(pads, w, bb.MU)
    times.append(time.perf_counter() - t)
print(f"\ntwo-pad M_T certificate, 16x16 taxels: median {1e3 * np.median(times):.0f} ms per decision")
