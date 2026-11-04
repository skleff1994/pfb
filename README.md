# pfb: planar friction bounds

Small library to check numerically the results of *Torsional friction in planar contact*
(the theory note, available on request; theorem numbers below refer to it). You define a normal force measure on a planar patch,
downsample it into taxels, and compute bounds on the support function of the admissible
tangential wrench set W(ν) under the hypotheses M_0, M_F and M_T.

## Install

```
pixi install           # numpy, scipy, matplotlib, cvxpy (conda-forge)
pixi run test          # one check per proposition
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
q  = pfb.directions(30)                           # directions q = (a1, a2, b), shape (30, 3)

h = pfb.support(nu, q, mu)                                     # true support h_W(nu)(q)
lo0, up0 = pfb.reduced_M0(P, pfb.measure_M0(nu), q, mu)        # total force + CoP
loT, upT = pfb.reduced_MT(tx, pfb.measure_MT(nu, tx), q, mu)   # taxel forces + CoP
loF, upF = pfb.reduced_MF(tx, pfb.measure_MF(nu, tx), q, mu)   # taxel forces

# Same bounds by brute force on the support grid (independent check of the theorems):
glo, gup = pfb.bounds_MT(tx, pfb.measure_MT(nu, tx), q, mu)    # loT <= glo <= h <= gup <= upT
```

Plots:

```python
import pfb.plot as pp
fig, B = pp.overview(nu, tx, mu)   # patch | 3D W(nu) | support and bounds | projection
```

Contact-mechanics fields and population studies:

```python
P  = pfb.Patch(1.0, 1.0, N=64)                          # unit pad
nu = pfb.hertz(P, center=(0.4, 0.5), radius=0.3)        # also xydas_kao, flat_punch, ramp, line_contact, asperities
fields = pfb.population(P, n_per_family=3, seed=0)      # 7 families x 3 fields (list of pfb.Field)
S = pfb.resolution_study(fields, ks=[1, 2, 4, 8], Q=[[0, 0, 1]], cache="study.npz")
S["MT_lo"][:, :, 0] / S["truth"][:, None, 0]            # certified M_T lower bound / true, per field and resolution
```

## How it works

There are two independent ways to compute each bound.

**Reduced formulations (`pfb/reduced.py`, the default).** These are the finite-dimensional problems
of the draft, solved on the continuous taxel geometry:
- **M_0:** closed-form lower bound and vertex LP upper bound (Thm 4.3, Cor 4.6). This is M_T with a single taxel (Rem 6.3).
- **M_T:** SOCP lower bound (Thm 6.1) and vertex LP upper bound (Thm 6.2).
- **M_F:** a single convex SOCP for the lower bound (Thm 7.2). The upper bound is the maximum over
  the CoP c in C_MF of the M_T vertex LP (Prop 7.1, Thm 7.3).

The SOCPs use cvxpy (Clarabel) and the LPs use `scipy.optimize.linprog` (HiGHS). The reduced formulas are exact
for the taxel model of Definition 5.2, where a load on an edge shared by two taxels may be counted by either of them. The grid LPs use strict
half-open cells instead. The closure model contains the strict one, so the reduced bounds are conservative for it (Remark 5.9);
for rectangular taxels the two gave the same values in all tests.

**Grid LPs (`pfb/bounds.py`).** These are brute force:
- A measure is a set of nonnegative weights on the nodes of the patch grid (boundary included), and each GMP becomes an LP over those weights.
- The true measure lives on the same grid, so `lower <= h_true <= upper` holds exactly in the discrete problem.
- The discrete bounds are inner approximations of the continuous ones (about one grid spacing), so
  `reduced lower <= grid lower <= h_true <= grid upper <= reduced upper`.
- Comparing the two methods tests the theorems numerically.

**The search over the CoP.** For the M_F upper bound (both methods) and the grid M_F lower bound, c is searched over C_MF with
`outer_search`: an `n_c x n_c` grid, then a Nelder–Mead refinement (`polish=True`). This search is approximate.
The reduced M_F lower bound is global, since it is a convex program.

Every bound function takes `q` of shape `(3,)` or `(m, 3)` and returns `(lower, upper)`.

## API

### Geometry (`pfb/geometry.py`)
| | |
|---|---|
| `Patch(Lx, Ly=0, N=48)` | Rectangle `[0,Lx]x[0,Ly]` with an `(N+1)^2` node grid. `Ly=0` gives a segment. |
| `TaxelGrid(patch, nx, ny=1)` | Partition into `nx x ny` rectangular taxels (`N` must be divisible by `nx`, `ny`). Provides `.downsample(nu)`, `.refine(k)`, `.lo`, `.hi`, `.diam`. |
| `directions(n, torque_scale=1)` | Directions spread over the half sphere (b >= 0); `torque_scale` multiplies b. |
| `J` | The skew-symmetric matrix with x × y = (Jx)ᵀy. |

### Measures (`pfb/measure.py`)
| | |
|---|---|
| `Measure(patch, w)` | Weights on the grid nodes. Provides `.fN`, `.m1`, `.cop`, `.atoms`; supports `+` and `*`. |
| `from_pressure(patch, rho, fN=None)` | From a density `rho(x, y)`. |
| `from_diracs(patch, points, forces)` | Point loads; each point must be a grid node. |
| `uniform`, `gaussian`, `random_measure` | Ready-made measures. |
| `support(nu, q, mu, c=None)` | Support function, Prop 3.2. |
| `gamma_star`, `wrench`, `support_points` | Optimal friction field, resultant wrench, and the points of W(ν) that attain the support. |

### Reduced bounds (`pfb/reduced.py`)
| | |
|---|---|
| `reduced_M0(patch, y0, q, mu)` | Thm 4.3 closed form, and the Cor 4.6 vertex LP. |
| `reduced_MT(taxels, yT, q, mu)` | Thm 6.1 SOCP lower bound, and the Thm 6.2 vertex LP upper bound. |
| `reduced_MF(taxels, F, q, mu, n_c=7, polish=True)` | Thm 7.2 joint SOCP lower bound, and Thm 7.3 (max over c of the vertex LP). |
| `certified_ref_MF(taxels, F, q, mu, c_hat=None)` | Prop 7.4: fixed-reference bounds corrected by μ\|b\| f_N δ(ĉ). By default ĉ is the centre of C_MF. |
| `vertex_lp_MT(taxels, F, c, q, mu, return_weights)` | Inner LP of Thm 6.2 for one q and c. The weights have at most r+2 nonzeros. |
| `socp_lower_MT(taxels, F, c, q, mu)`, `socp_lower_MF(taxels, F, q, mu)` | The SOCPs of Thms 6.1 and 7.2 (`return_points=True` also returns the minimizing points c_i). |
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
| `resolution_study(fields, ks, Q, mu, mf_kwargs, cache)` | Every bound (M_0, M_T, M_F, Prop 7.4) and the plug-in, for each field, each k x k grid and each direction. Returns arrays of shape (n_fields, n_k, n_q); optional `.npz` cache, reused only when the parameters match. The `MF_up` entry is the sweep of Thm 7.3 (a value ≤ the exact bound). |

### Certified load factor (`pfb/certify.py`)
For a load w = (f_1, f_2, τ) about a point o, the load factor s* = max{s : s·w ∈ W} is the capacity along w (for w = (0, 0, 1), the maximal pure torque).
| | |
|---|---|
| `PadTerm.from_taxels(taxels, F, cop=None, R=None, t=None)` | One pad: taxel boxes and forces, optional measured CoP (M_T; M_F without it), optional map `x = R x_pad + t` to a common frame. A single taxel covering the pad gives M_0 (with CoP) or an unsensed pad (without). |
| `load_factor_lower(pads, w, mu, o)` | Exact certified load factor s*_lo (two SOCPs), for one pad or a list (Minkowski sum). `s*_lo >= 1` certifies that every compatible pressure carries w. |
| `load_factor_true(points, forces, w, mu, o)` | Exact s* for a known discrete measure (the truth, or a plug-in estimate). |

### Grid bounds (`pfb/bounds.py`)
| | |
|---|---|
| `measure_M0`, `measure_MF`, `measure_MT` | Measurements y_0 = (fN, c), y_F = F, y_T = (fN, c, F). |
| `bounds_M0(patch, y0, q, mu, lower="exact")` | Thm 4.3 closed-form lower bound (or `"lp"`), and the grid LP upper bound. |
| `extremal_M0(patch, y0, q, mu)` | Value and weights of the optimal grid measure (to check the at most 3 atoms of Thm 4.5). |
| `bounds_MT(taxels, yT, q, mu)` | Grid LPs for the M_T GMPs. |
| `cellwise_MT(taxels, yT, q, mu)` | Closed-form cellwise bounds, Prop 6.4. |
| `bounds_MF(taxels, F, q, mu, n_c=7, polish=True)` | Search over c in C_MF of the grid LPs, Prop 7.1. |
| `fixed_ref_MF(taxels, F, c_hat, q, mu)` | Uncorrected fixed-reference approximation. This is not a bound; use `certified_ref_MF`. |
| `cop_set_MF(taxels, F)` | Compatible CoP set C_MF of Prop 5.6, a rectangle `(lo, hi)`. |
| `outer_search(f, lo_c, hi_c, sense, n_c, polish)` | Grid search plus Nelder–Mead over the C_MF rectangle. |
| `gmp(points, q, c, A_eq, b_eq, mu, sense)` | Generic LP solver used by the functions above, also usable for new measurement operators. |

### Plots (`pfb/plot.py`)
| | |
|---|---|
| `plot_patch(nu, taxels)` | Pressure map, taxels, CoP, C_MF. |
| `plot_wrench_set(nu, mu, bounds=...)` | W(ν) in 3D, plus optional inner (W⁻) and outer (W⁺) sets. |
| `plot_support(theta, B)` | Support function and bounds against the direction angle. |
| `plot_projection(Q, plane, B)` | Sets in the plane `"fx-fy"`, `"fx-tau"` or `"fy-tau"`. |
| `plane_directions`, `all_bounds`, `overview` | Helpers: directions in a plane, all bounds at once (`method="reduced"` or `"grid"`), one summary figure. |
| `plot_band(ax, x, Y, color, label)` | Median over the samples (first axis of Y) with the interquartile band, for population statistics. |

The sets are built from the support values sampled over a finite set of directions,
`{w : <q_i, w> <= h(q_i)}`:
- From the true support or an upper bound, the set in a plane is the projection of W or W⁺.
- From a lower bound, the set in a plane contains the projection of W⁻.

## Examples (`examples/`)

| pixi task | script | shows |
|---|---|---|
| `counterexample` | `01_counterexample.py` | Example 4.8: same (fN, c), different torsional support. |
| `bounds` | `02_bounds_vs_truth.py` | True support vs every bound. Checks Props 6.4, 7.4 and 8.1, and that the grid bounds lie inside the reduced ones. |
| `refinement` | `03_refinement.py` | Gap vs taxel size against Prop 8.3, up to 32x32 taxels; gain from the CoP (M_T vs M_F); contraction of C_MF. |
| `visualize` | `04_visualize.py` | Overview figure, M_T sets under refinement, segment counterexample. |
| `concepts` | `05_concepts.py` | Figures for the draft (PDF to `$PFB_DOC_FIG`): what M_0/M_F/M_T observe, vertex reduction (concentrate vs spread), and ghost masses with the closure fix (`closure.pdf`, not used by the draft at the moment). |
| `fields` | `06_fields.py` | Gallery of the pressure families (`fields.pdf`). |
| `wrench-sets` | `07_wrench_sets.py` | F1: guaranteed (W⁻) and possible (W⁺) sets for M_0, M_F, M_T, partial tilted contact, 4x4 and 16x16 taxels (`wrench_sets.pdf`). |
| `resolution` | `08_resolution.py` | F2: bound / true over the population vs resolution, value of the CoP (M_T vs M_F), gaps vs Prop 8.3 (`resolution.pdf`). |
| `plugin` | `09_plugin.py` | F3: plug-in / true vs certified lower bound / true; share of unsafe plug-in estimates (`plugin.pdf`). |
| `load-factor` | `10_load_factor.py` | Certified load factor / true (capacity along a load, not a support value) vs resolution, plug-in for comparison, and certificate solve times (`load_factor.pdf`). |

Examples 08 and 09 share one population study (`examples/_study.py`). The first run computes it (about 6–8 min) and caches it in
`examples/data/` (git-ignored); later runs reload it. Both accept `--quick` (1 field per family, k ≤ 16, about 1 min) and `--recompute`.
The figures for the draft are written as PNG to `examples/figures/` and as PDF to the directory `$PFB_DOC_FIG`
(default: `examples/figures/pdf/`; see `examples/_style.py`).

## Tests

- `tests/test_propositions.py` checks the propositions with the grid LPs: Prop 3.2, Example 4.8, Thms 4.3 and 4.5,
  Prop 5.6, the refinement results, Prop 8.1 and Prop 6.4.
- `tests/test_reductions.py` checks the reduced formulations against the grid LPs, including their convergence as N grows.
  It covers Thms 6.1, 6.2 (r+2 sparsity), 7.2 and 7.3, Rem 6.3, Props 7.1, 7.4, 8.1 and 8.3, Remark 5.9 (numerically), and the extreme-point CoP case (its corollary is commented out in the draft).

- `tests/test_certify.py` checks the certified load factor: attained at its binding direction, below finite direction grids, equal to a brute-force minimum over point loads, sound (≤ truth), ordered (unsensed ≤ M_0, M_F ≤ M_T), tighter under refinement, and sound for two mirrored pads.
- `tests/test_fields.py` checks the pressure families (valid measures, Xydas–Kao limits, clipping), the plug-in estimate
  (it reproduces the taxel forces), and a small `resolution_study` (orderings of all bounds, cache reuse).

If a test fails, either the code has a bug or a statement of the draft is false.
