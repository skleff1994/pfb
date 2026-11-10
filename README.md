# pfb: planar friction bounds

Small library to check numerically the results of *Torsional friction in planar contact*
(the statement names in brackets below, e.g. [thm:MF], refer to the companion theory note).
You define a normal force measure on a planar patch, downsample it into taxels, and compute bounds on the support
function of the admissible tangential wrench set X(ν) under the measurement models M_0 (total force and CoP),
M_F (taxel forces) and M_T (both).

**Conventions.** Wrenches w = (f_1, f_2, τ) and twists q = (a_1, a_2, b) are written about a fixed point: the origin of
the patch coordinates, unless a reference point `o` is passed. The support is h(q) = μ ∫ ‖a + bJx‖ dν, the maximal
power friction can dissipate under the twist q. With a fixed reference point, every bound is a generalized moment
problem (linear in the measure). `pfb.shift_twist(q, p)` converts a twist given about a point p (e.g. a rotation about
the CoP) to the origin. Measurements use homogeneous coordinates: y_0 = (f_N, m_1) with m_1 = ∫ x dν = f_N c.

## Install

```
pixi install           # numpy, scipy, matplotlib, cvxpy (conda-forge)
pixi run test          # one check per statement
pixi run examples      # run all example scripts; figures go to examples/figures/
pixi shell             # shell inside the environment
```

## Quick start

```python
import pfb

mu = 0.5
P  = pfb.Patch(1.0, 0.6, N=48)                    # patch [0,1]x[0,0.6], 49x49 grid nodes
nu = pfb.gaussian(P, (0.3, 0.35), 0.12, fN=2.0)   # "true" normal force measure
tx = pfb.TaxelGrid(P, 4, 4)                       # 4x4 taxels
q  = pfb.directions(30)                           # twists q = (a1, a2, b) about the origin, shape (30, 3)
q  = pfb.shift_twist(q, (0.5, 0.3))               # ... or about the pad centre

h = pfb.support(nu, q, mu)                                     # true support h_X(nu)(q)
lo0, up0 = pfb.reduced_M0(P, pfb.measure_M0(nu), q, mu)        # total force + first moment (CoP)
loT, upT = pfb.reduced_MT(tx, pfb.measure_MT(nu, tx), q, mu)   # taxel forces + CoP
loF, upF = pfb.reduced_MF(tx, pfb.measure_MF(nu, tx), q, mu)   # taxel forces (closed form)

# Same bounds by brute force on the support grid (independent check of the theorems):
glo, gup = pfb.bounds_MT(tx, pfb.measure_MT(nu, tx), q, mu)    # loT <= glo <= h <= gup <= upT
```

Certifying a load (does the contact carry w, surely, surely not, or undecided?):

```python
from pfb.certify import PadTerm
pad = PadTerm.from_taxels(tx, tx.downsample(nu), cop=nu.cop)   # omit cop for M_F
w = [0.0, 0.0, 0.05]                                           # pure torque about o
verdict, s_lo, s_up = pfb.classify(pad, w, mu, o=(0.5, 0.3))   # "hold" if s_lo >= 1, "slip" if s_up < 1
```

Plots:

```python
import pfb.plot as pp
fig, B = pp.overview(nu, tx, mu, o=(0.5, 0.3))   # patch | 3D wrench set | support and bounds | projection
```

Contact-mechanics fields and population studies:

```python
P  = pfb.Patch(1.0, 1.0, N=64)                          # unit pad
nu = pfb.hertz(P, center=(0.4, 0.5), radius=0.3)        # also xydas_kao, flat_punch, ramp, line_contact, asperities
fields = pfb.population(P, n_per_family=3, seed=0)      # 7 families x 3 fields (list of pfb.Field)
S = pfb.resolution_study(fields, ks=[1, 2, 4, 8], Q=[[0, 0, 1]], ref="cop", cache="study.npz")
S["MT_lo"][:, :, 0] / S["truth"][:, None, 0]            # certified M_T lower bound / true (rotation about the CoP)
```

## How it works

There are two independent ways to compute each bound.

**Exact finite-dimensional forms (`pfb/reduced.py`, the default),** solved on the continuous taxel geometry:
- **M_0:** closed-form lower bound μ‖f_N a + bJ m_1‖ [thm:coarse_lower] and patch-vertex LP upper bound
  [cor:coarse_upper_polygon], also available as a maximum over the vertex triangles containing the CoP [rem:coarse_triangles]. This is M_T with a
  single taxel [rem:MT_special].
- **M_F:** closed forms μ Σ_i F_i min / max of ‖a + bJx‖ over each taxel [thm:MF]; no solver.
- **M_T:** SOCP lower bound in the local first moments [thm:combined_lower] and corner LP upper bound [thm:combined_upper].

The SOCPs use cvxpy (Clarabel) and the LPs use `scipy.optimize.linprog` (HiGHS). The values are exact, as infima and suprema over compatible measures, for the half-open
taxels of a uniform grid. Point loads on an edge shared by two taxels are approached by moving them slightly into the taxel
[thm:combined_lower, thm:combined_upper].

**Grid LPs (`pfb/bounds.py`).** These are brute force:
- A measure is a set of nonnegative weights on the nodes of the patch grid (boundary included), and each GMP becomes an LP over those weights.
- The true measure lives on the same grid, so `lower <= h_true <= upper` holds exactly in the discrete problem.
- The discrete bounds are inner approximations of the continuous ones (about one grid spacing), so
  `reduced lower <= grid lower <= h_true <= grid upper <= reduced upper`.
- Comparing the two methods tests the theorems numerically.

Every bound function takes `q` of shape `(3,)` or `(m, 3)` and returns `(lower, upper)`.

## API

### Geometry (`pfb/geometry.py`)
| | |
|---|---|
| `Patch(Lx, Ly=0, N=48)` | Rectangle `[0,Lx]x[0,Ly]` with an `(N+1)^2` node grid. `Ly=0` gives a segment. |
| `TaxelGrid(patch, nx, ny=1)` | Partition into `nx x ny` half-open rectangular taxels (`N` must be divisible by `nx`, `ny`). Provides `.downsample(nu)`, `.refine(k)`, `.lo`, `.hi`, `.diam`. |
| `directions(n, torque_scale=1)` | Directions spread over the half sphere (b >= 0); `torque_scale` multiplies b. |
| `shift_twist(q, p)` | Coordinates about the origin of the twist given by q about the point p (Varignon) [lem:varignon]. |
| `J` | The skew-symmetric matrix with x × y = (Jx)ᵀy. |

### Measures (`pfb/measure.py`)
| | |
|---|---|
| `Measure(patch, w)` | Weights on the grid nodes. Provides `.fN`, `.m1`, `.cop`, `.atoms`; supports `+` and `*`. |
| `from_pressure(patch, rho, fN=None)` | From a density `rho(x, y)`. |
| `from_diracs(patch, points, forces)` | Point loads; each point must be a grid node. |
| `uniform`, `gaussian`, `random_measure` | Ready-made measures. |
| `support(nu, q, mu, o=None)` | Support function about o (default origin) [prop:support]. |
| `gamma_star`, `wrench`, `support_points` | Optimal friction field, resultant wrench, and the points of X_o(ν) that attain the support. |

### Exact bounds (`pfb/reduced.py`)
| | |
|---|---|
| `reduced_M0(patch, y0, q, mu)` | Closed-form lower bound, and the patch-vertex LP. |
| `coarse_upper_triangles(patch, y0, q, mu)` | The M_0 upper bound as the maximum over the vertex triangles containing the CoP. |
| `reduced_MF(taxels, F, q, mu)` | Closed-form M_F bounds. |
| `reduced_MT(taxels, yT, q, mu)` | SOCP lower bound and corner LP upper bound. |
| `vertex_lp_MT(taxels, F, m1, q, mu, return_weights)` | Corner LP for one q; the weights have at most r+2 nonzeros. |
| `socp_lower_MT(taxels, F, m1, q, mu)` | The M_T SOCP (`return_points=True` also returns the minimizing point loads c_i). |
| `taxel_vertices(taxels)` | Corners of the taxel closures, shape `(r, 4, 2)`. |

### Pressure fields (`pfb/fields.py`)
Each constructor returns a `Measure` with total force `fN` (default 1). The part of a contact outside the pad is clipped (partial contact).
| | |
|---|---|
| `hertz(patch, center, radius)` | Sphere on the pad, p ∝ √(1 − r²/a²). |
| `xydas_kao(patch, center, radius, k)` | Soft-finger family p ∝ (1 − (r/a)^k)^(1/k); k = 2 is Hertz, large k a uniform disk. |
| `flat_punch(patch, center, radius)` | Rigid flat punch, p ∝ (1 − r²/a²)^(−1/2), load on the rim (singularity capped at one grid spacing). |
| `ramp(patch, center, radius, tilt, angle)` | Linear ramp over a disk, a simple tilted-contact model; tilt > 1 lifts off part of the disk. |
| `line_contact(patch, point, angle, half_width)` | Cylinder on the pad, a 2D Hertz strip. |
| `asperities(patch, centers, radii, forces)` | Sum of Hertz contacts. |
| `population(patch, n_per_family=3, seed=0)` | Fixed random draws of the 7 families in `FAMILIES` (`partial` = Hertz or ramp clipped by the pad edge). Returns `Field(family, label, measure)` objects. |

### Plug-in estimate and studies (`pfb/estimate.py`, `pfb/study.py`)
| | |
|---|---|
| `plugin_uniform(taxels, F)` | Plug-in estimate of the measure: each F_i spread uniformly over its taxel. Its support is the plug-in capacity (not a bound). |
| `resolution_study(fields, ks, Q, mu, ref="cop", cache)` | Every bound (M_0, M_T, M_F) and the plug-in, for each field, each k x k grid and each twist of Q given about `ref` (`"cop"`: the true CoP of each field; a point; or None for the origin). Returns arrays of shape (n_fields, n_k, n_q); optional `.npz` cache, reused only when the parameters match. |

### Load factors (`pfb/certify.py`)
For a load w = (f_1, f_2, τ) about a point o, the load factor s* = max{s : s·w ∈ X_o(ν)} is the capacity along w (for w = (0, 0, 1), the maximal pure torque) [prop:load_factor].
| | |
|---|---|
| `PadTerm.from_taxels(taxels, F, cop=None, R=None, t=None)` | One pad: taxel boxes and forces, optional measured CoP (M_T; M_F without it), optional map `x = R x_pad + t` to a common frame. A single taxel covering the pad gives M_0 (with CoP) or an unsensed pad (without). |
| `load_factor_lower(pads, w, mu, o)` | Exact certified load factor s_lo (two SOCPs), for one pad or a list (Minkowski sum). `s_lo >= 1`: every compatible pressure carries w. `return_worst=True` gives a compatible measure attaining it. |
| `load_factor_upper(pads, w, mu, o)` | Exact upper load factor s_up (one SOCP). `s_up < 1`: no compatible pressure carries w, slip is certain. `return_best=True` gives a most favourable compatible measure (corner weights). |
| `classify(pads, w, mu, o)` | `("hold" \| "slip" \| "undecided", s_lo, s_up)`. |
| `load_factor_true(points, forces, w, mu, o)` | Exact s* for a known discrete measure (the truth, or a plug-in estimate). |

### Grid bounds (`pfb/bounds.py`)
| | |
|---|---|
| `measure_M0`, `measure_MF`, `measure_MT` | Measurements y_0 = (fN, m1), y_F = F, y_T = (fN, m1, F). |
| `coarse_lower(y0, q, mu)` | Closed-form coarse lower bound μ‖f_N a + bJ m_1‖. |
| `bounds_M0(patch, y0, q, mu, lower="exact")` | Closed-form lower bound (or `"lp"`), and the grid LP upper bound. |
| `extremal_M0(patch, y0, q, mu)` | Value and weights of the optimal grid measure (to check the at most 3 atoms of the upper bound). |
| `bounds_MF(taxels, F, q, mu)`, `bounds_MT(taxels, yT, q, mu)` | Grid LPs for the M_F and M_T GMPs. |
| `cellwise_g(taxels, q, o=None)` | Per-taxel min and max of ‖a + bJ(x − o)‖ over each taxel closure (closed form). |
| `cop_set_MF(taxels, F)` | CoPs compatible with the taxel forces, the rectangle Σ_i (F_i/f_N) P_i (for plots). |
| `gmp(points, q, A_eq, b_eq, mu, sense)` | Generic LP solver used by the functions above, also usable for new measurement operators. |

### Plots (`pfb/plot.py`)
| | |
|---|---|
| `plot_patch(nu, taxels)` | Pressure map, taxels, CoP, compatible CoPs. |
| `plot_wrench_set(nu, mu, bounds=..., o=None)` | X_o(ν) in 3D, plus optional guaranteed (X⁻) and possible (X⁺) sets. |
| `plot_support(theta, B)` | Support function and bounds against the direction angle. |
| `plot_projection(Q, plane, B)` | Sets in the plane `"fx-fy"`, `"fx-tau"` or `"fy-tau"`. |
| `plane_directions`, `all_bounds`, `overview` | Helpers: directions in a plane, all bounds at once (`method="reduced"` or `"grid"`, twists about `o`), one summary figure. |
| `plot_band(ax, x, Y, color, label)` | Median over the samples (first axis of Y) with the interquartile band, for population statistics. |

The sets are built from the support values sampled over a finite set of directions,
`{w : <q_i, w> <= h(q_i)}`:
- From the true support or an upper bound, the set in a plane is the projection of X or X⁺.
- From a lower bound, the set in a plane contains the projection of X⁻.

## Examples (`examples/`)

| pixi task | script | shows |
|---|---|---|
| `counterexample` | `01_counterexample.py` | Segment example: same (fN, CoP), different support for the rotation about the CoP. |
| `bounds` | `02_bounds_vs_truth.py` | True support vs every bound (twists about the pad centre). Checks the orderings, and that the grid bounds lie inside the exact ones. |
| `refinement` | `03_refinement.py` | Gap vs taxel size against the bound μ\|b\| f_N max diam and the first-order rate, up to 32x32 taxels; gain from the CoP (M_T vs M_F); contraction of the compatible CoPs. |
| `visualize` | `04_visualize.py` | Overview figure, M_T sets under refinement, segment counterexample. |
| `concepts` | `05_concepts.py` | Figures for the theory note (PDF to `$PFB_DOC_FIG`): what M_0/M_F/M_T observe, concentrate vs spread, and ghost masses on taxel corners (`closure.pdf`, not used at the moment). |
| `fields` | `06_fields.py` | Gallery of the pressure families (`fields.pdf`). |
| `wrench-sets` | `07_wrench_sets.py` | F1: guaranteed (X⁻) and possible (X⁺) sets for M_0, M_F, M_T, torques about the true CoP, partial tilted contact, 4x4 and 16x16 taxels (`wrench_sets.pdf`). |
| `resolution` | `08_resolution.py` | F2: bound / true over the population vs resolution (twists about the true CoP), value of the CoP (M_T vs M_F), gaps vs the bound (`resolution.pdf`). |
| `plugin` | `09_plugin.py` | F3: plug-in / true vs certified lower bound / true; share of unsafe plug-in estimates (`plugin.pdf`). |
| `load-factor` | `10_load_factor.py` | Certified and upper load factors / true (capacity along a load, not a support value) vs resolution, plug-in for comparison, and certificate solve times (`load_factor.pdf`). |

Examples 08 and 09 share one population study (`examples/_study.py`). The first run computes it and caches it in
`examples/data/` (git-ignored); later runs reload it. Both accept `--quick` (1 field per family, k ≤ 16) and `--recompute`.
The figures for the theory note and the report are written as PNG to `examples/figures/` and as PDF to the directory
`$PFB_DOC_FIG` (default: `examples/figures/pdf/`; see `examples/_style.py`).

## Tests

- `tests/test_propositions.py` checks the statements with the grid LPs and the closed forms: the support function, the
  change of reference point, linearity, the segment example, the coarse bounds (closed form, at most 3 atoms, triangle
  form), the exact guaranteed set of M_0, convexity of the possible set, monotonicity along dyadic grid chains, the
  orderings between M_0, M_F and M_T, the gap bounds, and the first-order rate of tightening.
- `tests/test_reductions.py` checks the exact forms against the grid LPs, including their convergence as N grows (half-open
  taxels), the M_F closed forms against sampling, the r+2 sparsity, M_0 as single-taxel M_T, the compatibility of the
  minimizing point loads, and taxels without force.
- `tests/test_certify.py` checks the load factors: the certified one is attained at its binding direction, below finite
  direction grids, equal to a brute-force minimum over point loads, sound, ordered and tighter under refinement; the upper
  one brackets the truth, its primal and dual SOCPs agree and are attained by a corner measure, it tightens under
  refinement and with the CoP; both are sound for two mirrored pads, and `classify` detects certain slip.
- `tests/test_fields.py` checks the pressure families (valid measures, Xydas–Kao limits, clipping), the plug-in estimate
  (it reproduces the taxel forces), and a small `resolution_study` (orderings of all bounds, cache reuse).

If a test fails, either the code has a bug or a statement of the theory note is false.
