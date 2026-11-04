"""Capacity estimators of the S3 controller: the load factor s_hat of the current contact along w.

Input: the contact state of `sensing.contact_state` (one PadContact per pad, sorted -y first: face points
u = (z, x) in the common frame, normal forces, side-wall share, face areas) and the demanded load w = (f_z, f_x, tau_y) about
o = 0. Every pad has a GRID x GRID taxel footprint centred at o; the taxel forces are the face forces
binned by `sensing.taxel_forces`, so a policy sees only what a k x k tactile sensor would report.

    oracle         exact load factor of Drake's faces (both pads), pfb.load_factor_true
    MT<k>          certified, taxel forces and CoP on both pads, pfb.load_factor_lower
    MF<k>          certified, taxel forces only on both pads
    MT16_onepad    certified, pad 1 sensed (16x16 + CoP), pad 2 unsensed (total force, unknown CoP)
    plugin<k>      plug-in: taxel forces spread uniformly over their taxels (pfb.plugin_uniform)
    mask_uniform   F/T + contact mask: total force spread uniformly over the true contact area (the classical
                   uniform-pressure limit surface on the measured patch, e.g. a GelSight contact mask)
    uniform        F/T-only engineering baseline: total force spread uniformly over the footprint

Why these baselines: the plug-ins use the same taxel data as the certificates but a point estimate of
the pressure instead of the worst case, so they isolate the value of certification. mask_uniform is
what a vision-based sensor user would build without a pressure model (it knows the contact area
exactly). uniform is the F/T-only rule of thumb. All of them, and the oracle, compute the exact load
factor of their estimated measure, so the only difference between policies is the measure.
"""

import numpy as np

import pfb
from pfb.certify import PadTerm, load_factor_lower, load_factor_true
from sensing import GRID, taxel_forces

PATCH = pfb.Patch(GRID, GRID, N=48)  # support grid of the plug-in estimates (48 x 48 nodes per pad)
OFFSET = -np.array([GRID, GRID]) / 2  # common-frame position of the patch corner (taxel grid centred at o = 0)
_TAXELS = {k: pfb.TaxelGrid(PATCH, k, k) for k in (1, 2, 4, 8, 16)}
POLICIES = ("oracle", "MT2", "MT4", "MT8", "MT16", "MF16", "MT16_onepad", "plugin4", "plugin16", "mask_uniform",
            "uniform")
CERTIFIED_POLICIES = ("MT2", "MT4", "MT8", "MT16", "MF16", "MT16_onepad")


def _merge(state, forces=None):
    """Points (n, 2) and forces (n,) of all pads stacked; forces[i] replaces the forces of pad i if given."""
    pts = np.vstack([pad.points for pad in state])
    frc = np.concatenate([pad.forces for pad in state] if forces is None else forces)
    return pts, frc


def oracle(state, w, mu):
    """Exact load factor of the simulated contact: Drake's face loads of both pads, one measure."""
    return load_factor_true(*_merge(state), w, mu)


def mask_uniform(state, w, mu):
    """Load factor if each pad's total force were spread uniformly over its true contact area.

    The face forces are replaced by F_pad * A_f / A_pad, i.e. uniform pressure on the measured patch.
    This is exact for a uniform field (the flat pads here), optimistic for a Hertz-like (dome) contact,
    whose pressure is concentrated at the centre where it resists little torsion, and conservative for
    a rim-loaded (flat-punch) contact.
    """
    return load_factor_true(*_merge(state, [pad.forces.sum() * pad.areas / pad.areas.sum() for pad in state]), w, mu)


def _term(points, forces, k, cop):
    """PadTerm of one pad seen by a k x k sensor: taxel forces, and the measured CoP if cop (M_T) else none (M_F).

    The taxels live in pad coordinates [0, GRID]^2; t = OFFSET maps them to the common frame, so the
    CoP is shifted by -OFFSET into pad coordinates.
    """
    tx = _TAXELS[k]
    c = (forces @ points / forces.sum()) - OFFSET if cop else None
    return PadTerm.from_taxels(tx, taxel_forces(points, forces, tx, OFFSET), cop=c, t=OFFSET)


def _plugin(state, w, mu, k):
    """Load factor of the plug-in estimate: each taxel force spread uniformly over its taxel, both pads.

    k = 1 is the F/T-only baseline (one taxel: the total force uniform over the whole footprint).
    """
    tx = _TAXELS[k]
    pts, frc = [], []
    for pad in state:
        nu = pfb.plugin_uniform(tx, taxel_forces(pad.points, pad.forces, tx, OFFSET))
        pts.append(PATCH.points + OFFSET); frc.append(nu.w)
    return load_factor_true(np.vstack(pts), np.concatenate(frc), w, mu)


def _terms(policy, state):
    """PadTerms of a certified policy: both pads at k x k (MT<k>, MF<k>), or pad 1 sensed and pad 2 unsensed.

    The unsensed pad is a single taxel covering the footprint with the measured total force (the
    gripper's force sensor) and no CoP: the M_0 information, the weakest certificate for that pad.
    """
    if policy == "MT16_onepad":
        pad1, pad2 = state
        plain = PadTerm.from_taxels(_TAXELS[1], [pad2.forces.sum()], t=OFFSET)
        return [_term(pad1.points, pad1.forces, 16, True), plain]
    model, k = policy[:2], int(policy[2:])
    return [_term(pad.points, pad.forces, k, model == "MT") for pad in state]


def estimate(policy, state, w, mu):
    """Load factor s_hat of the contact state along w for a policy of POLICIES (see the module docstring)."""
    if policy == "oracle":
        return oracle(state, w, mu)
    if policy == "mask_uniform":
        return mask_uniform(state, w, mu)
    if policy == "uniform":
        return _plugin(state, w, mu, 1)
    if policy.startswith("plugin"):
        return _plugin(state, w, mu, int(policy[6:]))
    return load_factor_lower(_terms(policy, state), w, mu)


def certificate(policy, state, w, mu):
    """Certified load factor and its worst case, for the viewer: (s, worst, worst_forces).

    worst is a list (one per pad) of point loads (r, 2) in the common frame with forces worst_forces:
    the compatible measure whose exact load factor is s (pfb.certify, return_worst=True). None for the
    non-certified policies, or when the binding direction has no torque part (b = 0): the bound
    mu f_N ||a|| is then the same for every compatible measure, and any of them is a worst case.
    """
    if policy not in CERTIFIED_POLICIES:
        return estimate(policy, state, w, mu), None, None
    terms = _terms(policy, state)
    s, worst = load_factor_lower(terms, w, mu, return_worst=True)
    return s, worst, None if worst is None else [T.F for T in terms]
