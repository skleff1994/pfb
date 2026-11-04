"""Minimal experiment: break-away of a bar squeezed between two flat compliant pads (S2 preconditions).

A rigid bar is held between two compliant-hydroelastic pads welded to the world (fixed grip, no
gravity). A force along world z, applied in the bar's mid-plane at a lever arm d along x, is ramped
slowly. It creates a shear and a torque about the grip axis (world y), i.e. a fixed wrench direction w.
The break-away factor s_obs is the force at which the bar starts to slide. It is compared with
    s_true   load factor of the simulated contact (both pads), pfb.load_factor_true on Drake's faces
    s_lo     certified load factors from k x k taxels on both pads (M_T, M_F) or on one pad (other unsensed)
Common frame: grasp-plane coordinates u = (z, x), normal +y, torques about +y, reference o = 0.
Run: pixi run -e sim python sim/bar_breakaway.py [--dt 1e-3] [--approx lagged] [--hint 1e-3] [--rate 2]
"""

import argparse
import time

import numpy as np
from pydrake.all import (AddCompliantHydroelasticProperties, AddContactMaterial, AddMultibodyPlant,
                         AddRigidHydroelasticProperties, Box, CoulombFriction, DiagramBuilder,
                         ExternallyAppliedSpatialForce, MultibodyPlantConfig, ProximityProperties,
                         RigidTransform, Simulator, SpatialForce, SpatialInertia, UnitInertia)

import pfb
from pfb.certify import PadTerm, load_factor_lower, load_factor_true
from sensing import contact_state, taxel_forces

MU = 0.8
PAD = np.array([0.020, 0.004, 0.016])   # pad size along world x, y (thickness), z (m)
BAR = np.array([0.150, 0.020, 0.030])   # bar size along x, y (grip axis), z
LEVER = 0.03                            # lever arm of the applied force along x (m)
SQUEEZE = 1e-4                          # overlap of each pad with the bar (m)


def build(dt, approx, hint, modulus, stiction):
    """Diagram of the fixed grip: two flat pads welded to the world, SQUEEZE into a free bar (no gravity).

    dt: physics step (s); approx: discrete contact solver; hint: pad mesh resolution hint (m), the bar
    uses hint / 2; modulus: hydroelastic modulus of the pads (Pa); stiction: stiction tolerance (m/s).
    The squeeze is set by the geometric overlap (welded pads), so the normal force is fixed during
    the ramp and the break-away load measures the friction capacity alone.
    Returns (diagram, plant, bar body).
    """
    builder = DiagramBuilder()
    config = MultibodyPlantConfig(time_step=dt, discrete_contact_approximation=approx, stiction_tolerance=stiction,
                                  contact_model="hydroelastic_with_fallback", contact_surface_representation="triangle")
    plant, _ = AddMultibodyPlant(config, builder)
    plant.mutable_gravity_field().set_gravity_vector([0, 0, 0])
    pads = []
    for side in (-1, 1):  # pad centre at y = side * (bar half-thickness + pad half-thickness - squeeze)
        props = ProximityProperties()
        AddCompliantHydroelasticProperties(resolution_hint=hint, hydroelastic_modulus=modulus, properties=props)
        AddContactMaterial(dissipation=5.0, friction=CoulombFriction(MU, MU), properties=props)
        X = RigidTransform([0, side * (BAR[1] / 2 + PAD[1] / 2 - SQUEEZE), 0])
        plant.RegisterCollisionGeometry(plant.world_body(), X, Box(*PAD), f"pad{side}", props)
        pads.append(X)
    bar = plant.AddRigidBody("bar", SpatialInertia(0.2, np.zeros(3), UnitInertia.SolidBox(*BAR)))
    props = ProximityProperties()
    AddRigidHydroelasticProperties(resolution_hint=hint / 2, properties=props)
    AddContactMaterial(friction=CoulombFriction(MU, MU), properties=props)
    plant.RegisterCollisionGeometry(bar, RigidTransform(), Box(*BAR), "bar", props)
    plant.Finalize()
    diagram = builder.Build()
    return diagram, plant, bar


def run(dt=1e-3, approx="lagged", hint=1e-3, modulus=5e5, stiction=1e-4, rate=2.0, t_max=40.0, v_slip=1e-3,
        ks=(1, 2, 4, 8, 16), verbose=True):
    """Ramp the load until break-away; compare it with the true and the certified load factors.

    rate: load ramp (N/s), meant to be quasi-static (sweep_breakaway.py compares it with 0.5 N/s). v_slip: break-away is the first 5 ms chunk where the bar's speed (translation plus
    LEVER x rotation rate) exceeds it (m/s); t_max stops a run that never breaks away. s_true and the
    certificates are evaluated on the contact state of the last chunk before break-away ("last stick"):
    the squeeze is fixed, but the pressure redistributes slightly under the tangential load.
    ks: taxel resolutions k x k of the certificates.
    Returns dict(s_obs, s_true, s_true0 (no tangential load), walls, rows); rows are (k, M_T both pads,
    M_F both pads, M_T on pad 1 with pad 2 unsensed).
    """
    diagram, plant, bar = build(dt, approx, hint, modulus, stiction)
    sim = Simulator(diagram)
    ctx = sim.get_mutable_context()
    pctx = plant.GetMyMutableContextFromRoot(ctx)
    port = plant.get_applied_spatial_force_input_port()
    w = np.array([1.0, 0.0, -LEVER])  # unit force along z at x = LEVER: u-force (1, 0), torque about +y = -LEVER

    def apply(F):
        """Apply the force F (N) along world z at (LEVER, 0, 0) in the bar frame."""
        f = ExternallyAppliedSpatialForce()
        f.body_index = bar.index()
        f.p_BoBq_B = [LEVER, 0, 0]
        f.F_Bq_W = SpatialForce([0, 0, 0], [0, 0, F])
        port.FixValue(pctx, [f])

    apply(0.0)
    sim.AdvanceTo(0.5)  # settle the squeeze
    state0 = contact_state(plant, pctx)
    t0, chunk, last, wall = sim.get_context().get_time(), 5e-3, None, time.time()
    while True:
        t = sim.get_context().get_time()
        F = rate * (t - t0)
        apply(F)
        sim.AdvanceTo(t + chunk)
        V = plant.EvalBodySpatialVelocityInWorld(pctx, bar)
        speed = np.linalg.norm(V.translational()) + LEVER * np.linalg.norm(V.rotational())
        if speed > v_slip or t - t0 > t_max:
            break
        last = (F, contact_state(plant, pctx))
    s_obs = last[0]

    pts = np.vstack([pad.points for pad in last[1]])
    frc = np.concatenate([pad.forces for pad in last[1]])
    s_true = load_factor_true(pts, frc, w, MU)
    s_true0 = load_factor_true(np.vstack([pad.points for pad in state0]), np.concatenate([pad.forces for pad in state0]), w, MU)
    walls = max(pad.walls for pad in last[1])

    patch = pfb.Patch(PAD[2], PAD[0], N=64)   # taxel grid in u = (z, x) over the pad footprint
    offset = -np.array([PAD[2], PAD[0]]) / 2
    rows = []
    for k in ks:
        tx = pfb.TaxelGrid(patch, k, k)
        terms_T, terms_F = [], []
        for p, f, *_ in last[1]:
            Fi = taxel_forces(p, f, tx, offset)
            cop = (f @ p / f.sum()) - offset
            terms_T.append(PadTerm.from_taxels(tx, Fi, cop=cop, t=offset))
            terms_F.append(PadTerm.from_taxels(tx, Fi, t=offset))
        one = pfb.TaxelGrid(patch, 1, 1)
        plain = PadTerm.from_taxels(one, [last[1][1].forces.sum()], t=offset)  # second pad unsensed, unknown CoP
        rows.append((k, load_factor_lower(terms_T, w, MU), load_factor_lower(terms_F, w, MU),
                     load_factor_lower([terms_T[0], plain], w, MU)))
    if verbose:
        print(f"dt={dt} approx={approx} hint={hint} stiction={stiction} rate={rate} N/s  (wall {time.time() - wall:.0f} s)")
        print(f"  normal force per pad: {[round(pad.forces.sum(), 3) for pad in last[1]]} N   faces: {[len(pad.forces) for pad in last[1]]}"
              f"   side-wall share: {walls:.2%}")
        print(f"  break-away s_obs = {s_obs:.4f} N   s_true (last stick) = {s_true:.4f}   s_true (no load) = {s_true0:.4f}"
              f"   s_obs / s_true = {s_obs / s_true:.4f}")
        print(f"  {'k':>5} | {'M_T both':>9} {'M_F both':>9} {'M_T + plain':>11}   (certified load factor / s_obs)")
        for k, sT, sF, sP in rows:
            print(f"  {f'{k}x{k}':>5} | {sT / s_obs:9.3f} {sF / s_obs:9.3f} {sP / s_obs:11.3f}")
    return dict(s_obs=s_obs, s_true=s_true, s_true0=s_true0, walls=walls, rows=rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--approx", default="lagged", choices=["lagged", "similar", "sap", "tamsi"])
    ap.add_argument("--hint", type=float, default=1e-3, help="pad resolution hint (m)")
    ap.add_argument("--stiction", type=float, default=1e-4, help="stiction tolerance (m/s)")
    ap.add_argument("--rate", type=float, default=2.0, help="force ramp rate (N/s)")
    a = ap.parse_args()
    run(a.dt, a.approx, a.hint, stiction=a.stiction, rate=a.rate)
