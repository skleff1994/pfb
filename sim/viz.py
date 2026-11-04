"""Visualization helpers for the Drake scenes: Meshcat 3D view and matplotlib pressure maps.

Meshcat (Drake's browser viewer) shows the bodies and the hydroelastic contact force and moment
arrows, live at the URL printed by `start_meshcat`. The same run is recorded and saved as a
standalone HTML replay (time slider, no Drake needed to open it). The scenes register collision
geometry only, so the viewer shows the proximity role, which is exactly what the contact model sees.

The matplotlib side draws what a tactile pad reports: the pressure field of Drake's contact faces
(the actual triangles, filled with their pressure), the k x k taxel grid of a policy, the measured CoP and the certificate's
worst-case point loads (pfb.certify.load_factor_lower, return_worst=True). All maps use grasp-plane
coordinates seen from +y: x horizontal, z vertical, in mm (the common frame of sensing.py is
u = (z, x), so the columns are swapped for display).
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
from pydrake.all import (ContactVisualizer, ContactVisualizerParams, Cylinder, MeshcatVisualizer,
                         MeshcatVisualizerParams, Rgba, RigidTransform, Role, RotationMatrix, StartMeshcat)

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from _style import COLOR, MUTED, plt  # noqa: E402,F401  shared figure style (Agg backend)

OUT = Path(__file__).parent / "out"  # generated HTML replays, figures and videos (git-ignored)
NEWTONS_PER_METER = 1000.0  # arrow scale: 10 N -> 1 cm, so the 1-80 N squeezes fit the 15 cm bar
NEWTON_METERS_PER_METER = 5.0  # 0.01 N m -> 2 mm
LOAD_PATH = "/load"
_MESHCAT = None


def start_meshcat():
    """The Meshcat server of this process, started on first use (its URL is printed by Drake)."""
    global _MESHCAT
    if _MESHCAT is None:
        _MESHCAT = StartMeshcat()
    return _MESHCAT


def add_visualizers(builder, plant, scene_graph, meshcat, publish_period=1 / 30):
    """Add the geometry and contact-force visualizers to a diagram before it is built.

    The geometry is drawn from the proximity role (the scenes have no illustration geometry), with an
    alpha slider so the bar can be seen through the pads. The contact visualizer draws, per hydroelastic
    contact, the resultant force (red) and moment (blue) at the centroid of the contact surface: these
    are the grasp reactions, which balance the applied load in quasi-static holding.
    Returns the MeshcatVisualizer, which also controls the recording.
    """
    vis = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, MeshcatVisualizerParams(
        role=Role.kProximity, prefix="proximity", publish_period=publish_period, enable_alpha_slider=True,
        initial_alpha_slider_value=0.6, default_color=Rgba(0.75, 0.72, 0.65, 1.0)))
    ContactVisualizer.AddToBuilder(builder, plant, meshcat, ContactVisualizerParams(
        publish_period=publish_period, newtons_per_meter=NEWTONS_PER_METER,
        newton_meters_per_meter=NEWTON_METERS_PER_METER, radius=0.0008))
    return vis


def init_load_arrow(meshcat, color=Rgba(0.1, 0.45, 0.85, 1.0)):
    """Create the arrow shaft of the applied external force (a unit-length cylinder, scaled per frame)."""
    meshcat.SetObject(LOAD_PATH + "/shaft", Cylinder(0.0012, 1.0), color)
    meshcat.SetProperty(LOAD_PATH, "visible", False)


def set_load_arrow(meshcat, origin, force, time=None):
    """Draw the external force (world frame, N) as a shaft from `origin`, NEWTONS_PER_METER scale.

    Meshcat recordings animate transforms and properties, not geometry, so the length is a scale of a
    fixed unit cylinder (along its z axis, centred) and the direction is a rotation taking z to the force.
    """
    n = np.linalg.norm(force)
    meshcat.SetProperty(LOAD_PATH, "visible", bool(n > 1e-6), time)
    if n <= 1e-6:
        return
    z = np.asarray(force) / n
    R = RotationMatrix.MakeFromOneVector(z, 2)  # columns: (x, y, z) with z along the force
    length = n / NEWTONS_PER_METER
    meshcat.SetTransform(LOAD_PATH, RigidTransform(R, np.asarray(origin)), time)
    meshcat.SetTransform(LOAD_PATH + "/shaft", RigidTransform([0, 0, length / 2]), time)
    meshcat.SetProperty(LOAD_PATH + "/shaft", "scale", [1.0, 1.0, length], time)


def save_html(meshcat, visualizer, name):
    """Stop the recording, publish it to the live viewer and save the standalone replay as OUT/name."""
    visualizer.StopRecording()
    visualizer.PublishRecording()
    OUT.mkdir(exist_ok=True)
    path = OUT / name
    path.write_text(meshcat.StaticHtml())
    print(f"saved {path} (open in a browser: replay with the Animations panel)")
    return path


def wait(meshcat):
    """Keep the process, hence the live Meshcat server, alive until the user presses Enter."""
    input(f"Meshcat live at {meshcat.web_url()}  (replay with Animations > play). Press Enter to exit. ")


def draw_pressure(ax, pad, vmax=None, cmap="Greys"):
    """Drake's contact faces of one pad, each filled with its mean pressure F_f / A_f (kPa).

    pad: a sensing.PadContact with triangles (common frame u = (z, x), m). Drawing the actual faces,
    rather than interpolating between centroids, shows exactly the measure the policies see: one
    point load per face (Drake's field is piecewise linear, so F_f / A_f is its mean over the face).
    vmax: top of the colour scale (kPa); default 1.5 x the peak, so that the overlays stay readable.
    Returns the collection (for a colour bar).
    """
    p = 1e-3 * pad.forces / pad.areas
    polys = 1e3 * pad.triangles[..., ::-1]  # (x, z) in mm for display
    pc = PolyCollection(polys, array=p, cmap=cmap, edgecolors="face", linewidths=0.2)
    pc.set_clim(0, 1.5 * p.max() if vmax is None else vmax)
    ax.add_collection(pc)
    return pc


def draw_pad(ax, pad, half, k=None, worst=None, worst_forces=None, vmax=None, cmap="Greys", view=None):
    """Draw one pad: pressure field, k x k taxel grid, measured CoP and the certificate's worst-case loads.

    pad: a sensing.PadContact with triangles. The footprint is the square [-half, half]^2 (m)
    centred at o = 0, as in policies.py. worst: (r, 2) point loads in the common frame (one per
    taxel) with forces worst_forces, the compatible measure that attains the certified load factor
    (drawn as orange circles, area ~ force). view: half-width of the shown square (m), default the
    footprint (zoom in on small contacts). Returns the pressure artist (for a colour bar).
    """
    mm = 1e3 * half
    lim = mm if view is None else 1e3 * view
    points, forces = pad.points, pad.forces
    img = draw_pressure(ax, pad, vmax, cmap)
    if k is not None:
        for v in np.linspace(-mm, mm, k + 1):
            ax.axvline(v, color="#2b7bba", lw=0.6, alpha=0.8)
            ax.axhline(v, color="#2b7bba", lw=0.6, alpha=0.8)
    if forces.sum() > 0:
        c = 1e3 * forces @ points / forces.sum()
        ax.plot(c[1], c[0], "+", color="#d03b3b", ms=10, mew=2, label="CoP")
    if worst is not None:
        F = np.asarray(worst_forces, float)
        keep = F > 1e-9
        size = 120 * F[keep] / max(F.max(), 1e-9)
        ax.scatter(1e3 * worst[keep, 1], 1e3 * worst[keep, 0], s=size, facecolor="none", edgecolor="#e08a00", lw=1.2,
                   label="worst case (certificate)")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    return img


# --- figure style shared by the study figures (plot_grip.py, plot_delay.py) ---------------------

def shade(color, t):
    """Blend color with white: t = 1 is the color, t = 0 is white."""
    return tuple(1 - t * (1 - c) for c in to_rgb(color))


STYLE = {  # policy: (label, color, line style)
    "oracle": ("oracle", COLOR["truth"], "-"),
    "MT2": ("certified $\\mathcal{M}_T$ 2x2", shade(COLOR["MT"], 0.35), "-"),
    "MT4": ("certified $\\mathcal{M}_T$ 4x4", shade(COLOR["MT"], 0.55), "-"),
    "MT8": ("certified $\\mathcal{M}_T$ 8x8", shade(COLOR["MT"], 0.78), "-"),
    "MT16": ("certified $\\mathcal{M}_T$ 16x16", COLOR["MT"], "-"),
    "MF16": ("certified $\\mathcal{M}_F$ 16x16", COLOR["MF"], "-"),
    "MT16_onepad": ("certified, one pad sensed", COLOR["MT"], "--"),
    "plugin4": ("plug-in 4x4", "#7a5bd6", "-"),
    "plugin16": ("plug-in 16x16", "#7a5bd6", "--"),
    "mask_uniform": ("F/T + contact mask", "#b8860b", "-"),
    "uniform": ("uniform over pad (F/T only)", COLOR["M0"], "-"),
}
OUTCOME = (("success", "#c9c8c3"), ("infeasible (flagged)", "#e0a030"), ("silent slip", "#d03b3b"))
