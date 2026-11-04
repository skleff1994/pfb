"""Smoke test of the Drake pipeline (S1): a rigid box resting on a compliant pad.

Checks that the face forces read from the contact surface add up to the weight and to Drake's own
resultant F_Ac_W, and prints the true pure-torque capacity of the simulated contact.
Shows the block settling into the gel in Meshcat (live, and saved as out/smoke.html), and the pad
pressure with the CoP in out/smoke.png.
Run: pixi run -e sim sim-smoke [--no-meshcat] [--no-wait]

Why these settings: SAP with the triangle contact surface is Drake's default discrete hydroelastic
pipeline; the pad is compliant (modulus 2e5 Pa, a soft gel) and the block rigid, so the contact
surface lies on the block face and its faces are the "point contacts" read by sensing.pad_measure.
"""

import argparse

import numpy as np
from matplotlib.patches import Rectangle
from pydrake.all import (AddCompliantHydroelasticProperties, AddContactMaterial, AddMultibodyPlant,
                         AddRigidHydroelasticProperties, Box, CoulombFriction, DiagramBuilder,
                         MultibodyPlantConfig, ProximityProperties, RigidTransform, Simulator,
                         SpatialInertia, UnitInertia)

import viz
from pfb.certify import load_factor_true
from sensing import pad_measure, support

MU, MASS = 0.8, 0.5
PAD = np.array([0.02, 0.016, 0.004])     # pad size (m): GelSight-like sensing area, 4 mm gel
OBJ = np.array([0.012, 0.012, 0.02])     # rigid block, 12 x 12 mm footprint
PLACE = np.array([0.001, -0.002])        # block centre in the pad plane (m), off-centre to test the CoP


def build(meshcat=None):
    """Diagram with the pad welded to the world (top face at z = 0) and a free block; optional Meshcat."""
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlant(
        MultibodyPlantConfig(time_step=1e-3, discrete_contact_approximation="sap",
                             contact_model="hydroelastic_with_fallback", contact_surface_representation="triangle"),
        builder)
    pad_props = ProximityProperties()
    AddCompliantHydroelasticProperties(resolution_hint=0.001, hydroelastic_modulus=2e5, properties=pad_props)
    AddContactMaterial(dissipation=5.0, friction=CoulombFriction(MU, MU), properties=pad_props)
    X_WP = RigidTransform([0, 0, -PAD[2] / 2])  # pad top surface at z = 0
    plant.RegisterCollisionGeometry(plant.world_body(), X_WP, Box(*PAD), "pad", pad_props)
    block = plant.AddRigidBody("block", SpatialInertia(MASS, np.zeros(3), UnitInertia.SolidBox(*OBJ)))
    obj_props = ProximityProperties()
    AddRigidHydroelasticProperties(resolution_hint=0.0005, properties=obj_props)
    AddContactMaterial(friction=CoulombFriction(MU, MU), properties=obj_props)
    plant.RegisterCollisionGeometry(block, RigidTransform(), Box(*OBJ), "block", obj_props)
    plant.Finalize()
    vis = viz.add_visualizers(builder, plant, scene_graph, meshcat) if meshcat is not None else None
    return builder.Build(), plant, block, X_WP, vis


def figure(pad, checks):
    """Pad pressure map with the block footprint, the placement point and the CoP (out/smoke.png).

    pad: PadContact in (pad y, pad x) order, because draw_pad plots column 1 horizontally.
    """
    fig, ax = viz.plt.subplots(figsize=(6.4, 4.6))
    half = PAD[0] / 2
    img = viz.draw_pad(ax, pad, half)
    mm = 1e3
    ax.add_patch(Rectangle(mm * (PLACE - OBJ[:2] / 2), mm * OBJ[0], mm * OBJ[1], fill=False, ls="--",
                           color="#2b7bba", label="block footprint"))
    ax.plot(*(mm * PLACE), "x", color="#2b7bba", ms=8, mew=2, label="block centre")
    ax.set_ylim(-mm * PAD[1] / 2, mm * PAD[1] / 2)
    ax.set_xlabel("pad x (mm)")
    ax.set_ylabel("pad y (mm)")
    fig.colorbar(img, ax=ax, label="pressure (kPa)", shrink=0.8)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.25, 1.0))
    ax.set_title("S1 smoke test: block resting on a compliant pad", loc="left", fontsize=10)
    fig.text(0.02, -0.06, "\n".join(checks), fontsize=8, family="monospace", va="top")
    viz.OUT.mkdir(exist_ok=True)
    fig.savefig(viz.OUT / "smoke.png")
    print(f"saved {viz.OUT / 'smoke.png'}")


def main(meshcat=True, wait=True):
    """Settle the block for 2 s, print the checks, save the figure and (with meshcat) the replay."""
    mc =viz.start_meshcat() if meshcat else None
    diagram, plant, block, X_WP, vis = build(mc)
    sim = Simulator(diagram)
    ctx = sim.get_mutable_context()
    plant_ctx = plant.GetMyMutableContextFromRoot(ctx)
    # start 0.1 mm into the gel (no drop impact), then let it settle
    plant.SetFreeBodyPose(plant_ctx, block, RigidTransform([*PLACE, OBJ[2] / 2 - 1e-4]))
    if vis is not None:
        vis.StartRecording()
        sim.set_target_realtime_rate(1.0)  # watchable live: the 2 s settle plays in about 2 s
    sim.AdvanceTo(2.0)

    results = plant.get_contact_results_output_port().Eval(plant_ctx)
    info = results.hydroelastic_contact_info(0)
    pad = pad_measure(info.contact_surface(), X_WP, triangles=True)
    x_P, F = pad.points, pad.forces
    weight = MASS * 9.81
    cop = F @ x_P / F.sum()
    tau_max = load_factor_true(x_P, F, [0, 0, 1], MU, o=cop)  # max pure torque = inf_a h((a, 1))
    checks = [
        f"hydroelastic contacts: {results.num_hydroelastic_contacts()}   pad faces: {len(F)}",
        f"sum of face normal forces = {F.sum():.4f} N   Drake resultant = {abs(info.F_Ac_W().translational()[2]):.4f} N"
        f"   weight = {weight:.4f} N",
        f"CoP (pad frame, mm) = {np.round(1e3 * cop, 3)}   (block placed at {1e3 * PLACE} mm)",
        f"true pure-torque capacity = {1e3 * tau_max:.3f} N mm   (support h(0,0,1) = "
        f"{1e3 * support(x_P, F, [0, 0, 1], MU):.3f} N mm; uniform 12 mm square: {1e3 * MU * weight * 0.012 * 0.3826:.3f} N mm)",
    ]
    print("\n".join(checks))
    figure(pad._replace(points=x_P[:, [1, 0]], triangles=pad.triangles[..., [1, 0]]), checks)
    if vis is not None:
        viz.save_html(mc, vis, "smoke.html")
        if wait:
            viz.wait(mc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-meshcat", action="store_true", help="no 3D viewer (figure and checks only)")
    ap.add_argument("--no-wait", action="store_true", help="exit at the end instead of keeping the viewer alive")
    a = ap.parse_args()
    main(not a.no_meshcat, not a.no_wait)
