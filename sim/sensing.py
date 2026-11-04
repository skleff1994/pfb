"""From a Drake hydroelastic contact surface to the measure used by pfb.

In Drake's discrete solvers each face f of the contact surface acts as one Coulomb point contact
at its centroid x_f, with normal force A_f p_f along the face normal (Masterjohn et al., RA-L 2022,
Sec. IV). For the planar model we keep the faces parallel to the pad (an object pressed into the gel
also touches it with its side walls) and use their force component along the pad normal. The normal
force measure seen by the pad is then nu = sum_f F_f delta_{x_f}, with x_f in the pad plane, and
its true frictional capacity is the support function of Prop 3.2 evaluated on these points.

Conventions of the grasp scenes (grip.py, bar_breakaway.py): the grip axis is world y, the grasp
plane is the world (x, z) plane, and its common coordinates are u = (z, x), so that the planar
cross product u1 x u2 is the torque about +y. Every pad has a GRID x GRID taxel footprint centred at
the reference point o = 0.
"""

from typing import NamedTuple

import numpy as np

GRIP_AXIS = np.array([0.0, 1.0, 0.0])  # world axis normal to the grasp plane, in the grasp scenes
GRID = 0.032  # side of the square taxel footprint of each pad in the grip scene (m), centred at o = 0


def to_common(p_W):
    """World points (n, 3) -> grasp-plane coordinates u = (z, x) (common frame, torques about +y).

    The order (z, x) makes (e_z, e_x, e_y) right-handed, so the 2D cross product of pfb (J of eq. (17))
    gives the torque about +y without a sign flip.
    """
    return np.asarray(p_W)[:, [2, 0]]


class PadContact(NamedTuple):
    """Faces of one pad contact parallel to the grasp plane: points u (n, 2), normal forces (n,),
    side-wall share of the whole contact, face areas (n,) projected on the grasp plane, and, only when
    requested (for drawing), the face triangles (n, 3, 2) in the same coordinates."""
    points: np.ndarray
    forces: np.ndarray
    walls: float
    areas: np.ndarray
    triangles: np.ndarray | None = None


def contact_state(plant, ctx, cos_min=0.9, triangles=False):
    """One PadContact per hydroelastic contact, sorted by the y of its faces (pad at -y first).

    Drake does not order the contacts by geometry, so they are sorted by position along the grip axis:
    index 0 is always pad 1 (at -y), which the one-pad policy treats as the sensed pad.
    cos_min = 0.9 keeps faces within 26 deg of the grasp plane. The kept forces and areas are
    projected on the plane (times cos), so the pressure F/A is unchanged and the forces are the
    components along the grip axis, which is what a pad normal to the axis measures.
    triangles: also return the face triangles (slower; used by the viewer only).
    """
    res = plant.get_contact_results_output_port().Eval(ctx)
    out = []
    for i in range(res.num_hydroelastic_contacts()):
        s = res.hydroelastic_contact_info(i).contact_surface()
        x_W, fn, normals, A = contact_faces(s)
        cos = np.abs(normals @ GRIP_AXIS)
        keep = cos >= cos_min
        tri = face_triangles(s)[keep][..., [2, 0]] if triangles else None  # common frame, as to_common
        pad = PadContact(to_common(x_W[keep]), fn[keep] * cos[keep], side_wall_share(s, GRIP_AXIS, cos_min),
                         A[keep] * cos[keep], tri)
        out.append((x_W[keep, 1].mean(), pad))
    return [pad for _, pad in sorted(out, key=lambda o: o[0])]


def contact_faces(surface):
    """Face centroids (n, 3), normal forces A_f p(x_f) (n,), unit normals (n, 3) and areas A_f (n,), world frame.

    p(x_f) is the piecewise-linear contact pressure at the centroid, which integrates it exactly over a
    triangle. Requires the triangle representation of the contact surface.
    """
    if not surface.is_triangle():
        raise ValueError("use contact_surface_representation='triangle'")
    field = surface.tri_e_MN()
    n = surface.num_faces()
    x = np.array([surface.centroid(f) for f in range(n)])
    p = np.array([field.EvaluateCartesian(f, x[f]) for f in range(n)])
    A = np.array([surface.area(f) for f in range(n)])
    normals = np.array([surface.face_normal(f) for f in range(n)])
    return x, A * p, normals, A


def face_triangles(surface):
    """Vertices (n, 3, 3) of the triangles of the contact surface, world frame, in face order."""
    mesh = surface.tri_mesh_W()
    V = np.array(mesh.vertices())
    return V[np.array([[t.vertex(j) for j in range(3)] for t in mesh.triangles()])]


def pad_measure(surface, X_WP, cos_min=0.9, triangles=False):
    """PadContact of the faces parallel to a pad with its own frame P, in pad coordinates (x_P, y_P).

    X_WP: pose of the pad frame P (a RigidTransform), whose z axis is the pad normal. Faces whose
    normal makes an angle larger than arccos(cos_min) with it (side walls) are dropped, and forces
    and areas are projected on the pad plane as in contact_state. Used for the smoke test; the grasp
    scenes use contact_state and the common frame instead.
    """
    x_W, fn, normals, A = contact_faces(surface)
    z_W = X_WP.rotation().matrix()[:, 2]
    cos = np.abs(normals @ z_W)
    keep = cos >= cos_min
    X_PW = X_WP.inverse()
    x_P = np.array([X_PW.multiply(xi) for xi in x_W[keep]])
    tri = None
    if triangles:
        tri = np.array([[X_PW.multiply(v)[:2] for v in t] for t in face_triangles(surface)[keep]])
    off = fn[~keep].sum() / fn.sum()
    return PadContact(x_P[:, :2], fn[keep] * cos[keep], off, A[keep] * cos[keep], tri)


def side_wall_share(surface, normal_W, cos_min=0.9):
    """Share of the total face force carried by faces not parallel to the pad (should be ~0).

    The planar model ignores these faces, so a scene is valid only if the share is small (the study
    rejects episodes above 0.5%).
    """
    _, fn, normals, _ = contact_faces(surface)
    off = np.abs(normals @ normal_W) < cos_min
    return fn[off].sum() / fn.sum()


def taxel_forces(points, forces, taxels, offset):
    """Taxel forces F_i: each face force assigned to the taxel containing its centroid.

    points are in the common frame; the taxel grid lives on [0, Lx] x [0, Ly], placed at `offset`
    (x_common = x_taxel + offset). Centroids outside the grid are clipped to the nearest taxel, so the
    total force is conserved (a real pad would lose it; the study rejects episodes where more than 0.5%
    of the force falls outside, see grip.outside_share). Assigning a whole face to one taxel is the
    sensing model; it is consistent with the certificate because the true measure (the face point
    loads) then has exactly these taxel forces.
    """
    size = (taxels.hi - taxels.lo)[0]
    ij = np.floor((points - offset) / size).astype(int)
    ij = np.clip(ij, 0, [taxels.nx - 1, taxels.ny - 1])
    return np.bincount(ij[:, 1] * taxels.nx + ij[:, 0], weights=forces, minlength=taxels.r)


def support(points, forces, q, mu):
    """h_W(nu)(q) = mu sum_f F_f ||a + bJ(x_f - c)|| for the discrete measure (Prop 3.2), q of shape (3,) or (m, 3).

    The torque is taken about the CoP c of the measure, as in Prop 3.2 (use pfb.certify for a fixed o).
    """
    q = np.atleast_2d(np.asarray(q, float))
    c = forces @ points / forces.sum()
    d = points - c
    v = q[:, None, :2] + q[:, None, 2:3] * np.stack([-d[:, 1], d[:, 0]], axis=1)[None]  # a + bJ(x - c)
    h = mu * np.linalg.norm(v, axis=2) @ forces
    return h[0] if len(h) == 1 else h
