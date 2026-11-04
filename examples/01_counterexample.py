"""Example 4.8: (f_N, CoP) does not determine W(nu).

On the segment [0, L], a Dirac f_N at the midpoint and two Diracs f_N/2 at the endpoints have
the same total force and CoP, but different torsional support h(0, 0, 1).
"""

import numpy as np

import pfb

L, fN, mu = 1.0, 2.0, 0.5
seg = pfb.Patch(L, 0.0, N=20)
nu1 = pfb.from_diracs(seg, [(L / 2, 0)], [fN])
nu2 = pfb.from_diracs(seg, [(0, 0), (L, 0)], [fN / 2, fN / 2])

print("M0(nu1) =", pfb.measure_M0(nu1))
print("M0(nu2) =", pfb.measure_M0(nu2))
q = np.array([0.0, 0.0, 1.0])
print(f"h_W(nu1)(0,0,1) = {pfb.support(nu1, q, mu):.4f}   (expected 0)")
print(f"h_W(nu2)(0,0,1) = {pfb.support(nu2, q, mu):.4f}   (expected mu fN L/2 = {mu * fN * L / 2:.4f})")

# The M_0 bounds bracket both: lower = mu fN ||a|| = 0, upper = the most torsion any
# measure with this (f_N, c) can support.
lo, up = pfb.bounds_M0(seg, pfb.measure_M0(nu1), q, mu)
print(f"M_0 bounds in pure torsion: [{lo:.4f}, {up:.4f}]")
