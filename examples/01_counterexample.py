"""Segment example: (f_N, CoP) does not determine the admissible wrench set.

On the segment [0, L], a Dirac f_N at the midpoint and two Diracs f_N/2 at the endpoints have
the same total force and CoP, but different support for the unit rotation about the CoP.
"""

import numpy as np

import pfb

L, fN, mu = 1.0, 2.0, 0.5
seg = pfb.Patch(L, 0.0, N=20)
nu1 = pfb.from_diracs(seg, [(L / 2, 0)], [fN])
nu2 = pfb.from_diracs(seg, [(0, 0), (L, 0)], [fN / 2, fN / 2])

print("M0(nu1) =", pfb.measure_M0(nu1))
print("M0(nu2) =", pfb.measure_M0(nu2))
q = pfb.shift_twist([0.0, 0.0, 1.0], nu1.cop)  # unit rotation about the CoP, written about the origin
print(f"h(nu1)(q_c) = {pfb.support(nu1, q, mu):.4f}   (expected 0)")
print(f"h(nu2)(q_c) = {pfb.support(nu2, q, mu):.4f}   (expected mu fN L/2 = {mu * fN * L / 2:.4f})")

# The M_0 bounds bracket both: lower = mu ||f_N a + bJ m_1|| = 0 for a rotation about the CoP, upper = the most
# torsion about the CoP that any measure with this (f_N, m_1) can support.
lo, up = pfb.bounds_M0(seg, pfb.measure_M0(nu1), q, mu)
print(f"M_0 bounds, rotation about the CoP: [{lo:.4f}, {up:.4f}]")
