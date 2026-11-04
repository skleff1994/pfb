"""S3 scene and episode: closed-loop grip control of a bar held between two compliant pads.

A rigid bar (gravity off) is squeezed between pad 1 (at -y) and pad 2 (at +y), each on a prismatic
joint along y and pushed with the same squeeze force, which follows the commanded value through a
first-order filter (time constant TAU_GRIP, the gripper bandwidth). A known external wrench, in the
grasp plane, is applied to the bar; the grasp must supply the demanded wrench D(t) w(t).

The bar moves on a planar joint (translations x, z and rotation about the grip axis y). This enforces
the assumption of the planar model that out-of-plane moments are carried without using friction. With a
free bar, pads whose normal-force centres do not line up create a couple about an in-plane axis that
must be balanced by opposite tangential forces at the two faces, which the planar certificate ignores
(observed in the pilot with tilted pads: the oracle slipped at a 5% margin).

Controller (same for every policy): every DT_CTRL the policy estimates the load factor s_hat of the
current contact along w (pfb.certify conventions: w = (f_z, f_x, tau_y) in the common frame u = (z, x),
o = 0), and commands
    f_cmd = clip(f_meas D (1 + m) / s_hat, F_MIN, f_max),
with decreases limited to RELAX per cycle. D and w are taken LEAD ahead (feedforward of the known load
profile, covering the control period and the gripper lag); when w changes along the profile, the
larger of the commands for the current and the upcoming load is used. The update assumes the capacity scales linearly with the
squeeze (a declared heuristic, outside the theory); a certificate is sound for the measured state only.
f_max is the force budget (the pad's working range in the study). When the requested squeeze exceeds it,
the policy cannot carry the load at the budget by its own estimate: the episode is flagged infeasible
(t_flag) and the squeeze stays at f_max. A slip without an earlier flag is a silent slip.

Pad families (compliant hydroelastic, the contact surface lies on the flat rigid bar face):
    dome     ellipsoid, Hertz-like field whose area grows with the squeeze
    tilted   box tilted about x or z: ramp field / partial edge contact
    flat     box, near-uniform field (control case)
Run a single episode (text log): pixi run -e sim python sim/grip.py --seed 0 --policy oracle
Watch one (Meshcat, dashboard, video): pixi run -e sim sim-show --seed 0 --policy MT16
"""

import argparse
import time
from dataclasses import dataclass, field

import numpy as np
from pydrake.all import (AddCompliantHydroelasticProperties, AddContactMaterial, AddMultibodyPlant,
                         AddRigidHydroelasticProperties, Box, CoulombFriction, DiagramBuilder, Ellipsoid,
                         ExternallyAppliedSpatialForce, FixedOffsetFrame, MultibodyPlantConfig, PlanarJoint,
                         PrismaticJoint, ProximityProperties, RigidTransform, RollPitchYaw, Simulator, SpatialForce,
                         SpatialInertia, UnitInertia)

from policies import POLICIES, estimate, oracle
from sensing import GRID, contact_state  # noqa: F401  (GRID re-exported: grip.GRID is the taxel footprint)

MU = 0.8
BAR = np.array([0.150, 0.020, 0.060])   # bar size along x, y (grip axis), z (m)
MODULUS = 5e5                           # hydroelastic modulus of the pads (Pa), a soft elastomer
CONTACT0 = 1e-4                         # initial overlap of each pad with the bar (m)
F_MIN, F_MAX = 1.0, 100.0               # squeeze limits (N); F_MAX is the default budget (no limit in practice)
F_TARGET = 10.0                         # D_max: demand the oracle would carry at F_TARGET, by linear scaling
DT, DT_CTRL, DT_SUB = 1e-3, 0.05, 5e-3  # physics step, control period (20 Hz), load/filter update period (s)
TAU_GRIP = 0.05                         # gripper bandwidth (first-order filter time constant, s)
LEAD = DT_CTRL + 2 * TAU_GRIP           # feedforward horizon: one control period + the filter's ~86% rise time
RELAX = 0.02                            # max relative squeeze decrease per control cycle (increases are immediate)
DEPTH_MAX = 0.5                         # valid scene: pad penetration below this fraction of its semi-thickness
SLIP_FAIL, SLIP_STOP = 2e-4, 2e-3       # slip at the pad faces that counts as a failure / ends the episode (m)
SLIP_ONSET = SLIP_FAIL / 4              # slip at which the onset of a failure is diagnosed (m)
CAUSE_TERMS = ("estimate", "load", "state", "actuator")  # terms of the shortfall at slip onset (diagnose)
RIM = GRID / 2                          # lever arm converting a rotation about y into a slip at the pad rim (m)
FAMILIES = ("dome", "tilted", "flat")
LOAD_ARROW_Y = -0.045                   # the load arrow is drawn in front of pad 1 (Meshcat), not inside the bar


@dataclass
class PadSpec:
    """Geometry of one pad: family, size (ellipsoid semi-axes or box sizes along x, y, z, m), tilt, offset."""
    family: str
    size: tuple          # ellipsoid semi-axes or box sizes along (x, y, z) (m)
    tilt: tuple = (0.0, 0.0)   # rotation about world x and z (rad)
    offset: tuple = (0.0, 0.0)  # in-plane offset of the pad centre along (x, z) (m)

    def semi_thickness(self):
        """Half of the pad's extent along the grip axis before tilting (m): the depth scale of DEPTH_MAX."""
        return self.size[1] if self.family == "dome" else self.size[1] / 2

    def extent(self):
        """Half-extent of the rotated geometry along y (towards the bar), used to place the pad in contact."""
        R = RollPitchYaw(self.tilt[0], 0.0, self.tilt[1]).ToRotationMatrix().matrix()
        half = np.asarray(self.size) / (1 if self.family == "dome" else 2)
        if self.family == "dome":  # support function of an ellipsoid along y
            return np.linalg.norm(half * R[1])
        return np.abs(R[1]) @ half


@dataclass
class EpisodeSpec:
    """One seed: the two pads, the in-plane force direction theta and the lever arm of the demanded load.

    lever is the ratio torque / force in multiples of the RMS contact radius rho measured at F_MIN
    (inf: pure torque), so that the load is torsion-dominated on every pad, whatever its size.
    """
    seed: int
    pads: tuple
    theta: float         # in-plane direction of the demanded force
    lever: float         # demanded torque / force, in multiples of the RMS contact radius at F_MIN (inf: pure torque)
    family: str = field(init=False)

    def __post_init__(self):
        self.family = self.pads[0].family


def sample_pad(rng, family):
    """Random pad of a family, with an in-plane offset of up to 2 mm (the CoPs of the two pads differ).

    dome: semi-axes 8-16 mm in the plane, 4-8 mm along y. tilted: 20 x 6 x 20 mm box tilted by
    1-6 deg about x or z, so the contact is a ramp or a partial edge contact. flat: same box, no tilt.
    """
    off = tuple(rng.uniform(-0.002, 0.002, 2))
    if family == "dome":
        a, c = rng.uniform(0.008, 0.016, 2)
        return PadSpec("dome", (a, rng.uniform(0.004, 0.008), c), offset=off)
    tilt = (0.0, 0.0)
    if family == "tilted":
        ang = np.deg2rad(rng.uniform(1.0, 6.0)) * rng.choice([-1, 1])
        tilt = (ang, 0.0) if rng.random() < 0.5 else (0.0, ang)
    return PadSpec(family, (0.020, 0.006, 0.020), tilt=tilt, offset=off)


def sample_episode(seed):
    """Seeds cycle through the pad families; both pads are drawn independently within the family.

    The load direction is uniform; 20% of the seeds demand a pure torque, the others a lever arm of
    2-6 RMS contact radii of either sign, so that torsion binds (peak squeeze >= 3x the shear-only value).
    """
    rng = np.random.default_rng(seed)
    family = FAMILIES[seed % len(FAMILIES)]
    pads = (sample_pad(rng, family), sample_pad(rng, family))
    theta = rng.uniform(0, 2 * np.pi)
    lever = np.inf if rng.random() < 0.2 else rng.uniform(2.0, 6.0) * rng.choice([-1, 1])
    return EpisodeSpec(seed, pads, theta, lever)


class Scene:
    """Drake diagram of the grip: two pads on prismatic joints and the bar on a planar joint.

    approx, stiction: discrete contact solver and its stiction tolerance (m/s). The default (lagged,
    1e-6) is the setting whose break-away matched the ideal Coulomb capacity within 0.992-1.001 in the
    S2 sweep (sweep_breakaway.py); Drake's default 1e-4 creeps and breaks away at 0.89 of it.
    hint: resolution hint of the pad meshes (m); the bar uses hint / 2.
    meshcat: a pydrake Meshcat to draw the scene and the contact forces (viz.add_visualizers); then
    self.visualizer controls the recording. None (default) builds the bare scene of the study.
    """

    def __init__(self, pads, approx="lagged", stiction=1e-6, hint=3e-3, meshcat=None):
        builder = DiagramBuilder()
        config = MultibodyPlantConfig(time_step=DT, discrete_contact_approximation=approx, stiction_tolerance=stiction,
                                      contact_model="hydroelastic_with_fallback",
                                      contact_surface_representation="triangle")
        plant, scene_graph = AddMultibodyPlant(config, builder)
        plant.mutable_gravity_field().set_gravity_vector([0, 0, 0])  # the external load is the only load
        self.slides, self.pads = [], pads
        for side, spec in zip((-1, 1), pads):
            body = plant.AddRigidBody(f"pad{side}", SpatialInertia(0.05, np.zeros(3), UnitInertia.SolidBox(0.02, 0.01, 0.02)))
            props = ProximityProperties()
            AddCompliantHydroelasticProperties(resolution_hint=hint, hydroelastic_modulus=MODULUS, properties=props)
            AddContactMaterial(dissipation=5.0, friction=CoulombFriction(MU, MU), properties=props)
            shape = Ellipsoid(*spec.size) if spec.family == "dome" else Box(*spec.size)
            y = side * (BAR[1] / 2 + spec.extent() - CONTACT0)  # pad surface CONTACT0 into the bar face
            X = RigidTransform(RollPitchYaw(spec.tilt[0], 0.0, spec.tilt[1]), [spec.offset[0], y, spec.offset[1]])
            plant.RegisterCollisionGeometry(body, X, shape, f"pad{side}", props)
            # joint damping (N s/m) damps the pad's own oscillation on the squeeze actuator
            joint = plant.AddJoint(PrismaticJoint(f"slide{side}", plant.world_frame(), body.body_frame(), [0, -side, 0],
                                                  damping=20.0))
            plant.AddJointActuator(f"squeeze{side}", joint)  # positive force pushes the pad towards the bar
            self.slides.append(joint)
        self.bar = plant.AddRigidBody("bar", SpatialInertia(0.2, np.zeros(3), UnitInertia.SolidBox(*BAR)))
        # planar joint in the grasp plane: its frames have z along the grip axis y (roll by -pi/2)
        X_plane = RigidTransform(RollPitchYaw(-np.pi / 2, 0.0, 0.0), np.zeros(3))
        F = plant.AddFrame(FixedOffsetFrame("grasp_plane", plant.world_frame(), X_plane))
        M = plant.AddFrame(FixedOffsetFrame("bar_plane", self.bar.body_frame(), X_plane))
        plant.AddJoint(PlanarJoint("planar", F, M, damping=np.zeros(3)))
        props = ProximityProperties()
        AddRigidHydroelasticProperties(resolution_hint=hint / 2, properties=props)
        AddContactMaterial(friction=CoulombFriction(MU, MU), properties=props)
        plant.RegisterCollisionGeometry(self.bar, RigidTransform(), Box(*BAR), "bar", props)
        plant.Finalize()
        self.plant, self.meshcat, self.visualizer = plant, meshcat, None
        if meshcat is not None:
            import viz  # noqa: PLC0415  (pulls in matplotlib; not needed by the study)
            self.visualizer = viz.add_visualizers(builder, plant, scene_graph, meshcat)
            viz.init_load_arrow(meshcat)
        self.diagram = builder.Build()
        self.sim = Simulator(self.diagram)
        self.ctx = plant.GetMyMutableContextFromRoot(self.sim.get_mutable_context())
        self.f_app = 0.0
        self.set_squeeze(0.0)
        self.set_load(np.zeros(3))

    def set_squeeze(self, f):
        """Apply the squeeze force f (N) on both prismatic joints (same force on each pad)."""
        self.f_app = f
        self.plant.get_actuation_input_port().FixValue(self.ctx, [f, f])

    def set_load(self, w_ext):
        """External wrench (f_z, f_x, tau_y) on the bar, at its origin (mid-plane)."""
        F = ExternallyAppliedSpatialForce()
        F.body_index = self.bar.index()
        F.p_BoBq_B = np.zeros(3)
        F.F_Bq_W = SpatialForce([0, w_ext[2], 0], [w_ext[1], 0, w_ext[0]])
        self.plant.get_applied_spatial_force_input_port().FixValue(self.ctx, [F])
        if self.meshcat is not None:
            import viz  # noqa: PLC0415
            p = self.pose().translation()
            viz.set_load_arrow(self.meshcat, [p[0], LOAD_ARROW_Y, p[2]], [w_ext[1], 0, w_ext[0]], self.t)

    @property
    def t(self):
        """Simulation time (s)."""
        return self.sim.get_context().get_time()

    def advance(self, dt, f_cmd):
        """Advance by dt with the squeeze filtered towards f_cmd (exact first-order update per DT_SUB).

        The filter is the gripper: within each DT_SUB the applied force is held, then moved to the exact
        solution of tau f' = f_cmd - f, so the response does not depend on the physics step.
        """
        for _ in range(int(round(dt / DT_SUB))):
            self.set_squeeze(f_cmd + (self.f_app - f_cmd) * np.exp(-DT_SUB / TAU_GRIP))
            self.sim.AdvanceTo(self.t + DT_SUB)

    def depth(self):
        """Max over the pads of penetration / semi-thickness (the bar cannot move along y)."""
        return max((j.get_translation(self.ctx) + CONTACT0) / spec.semi_thickness()
                   for j, spec in zip(self.slides, self.pads))

    def pose(self):
        """World pose of the bar (RigidTransform)."""
        return self.plant.EvalBodyPoseInWorld(self.ctx, self.bar)

    def slip(self, X0):
        """In-plane slip of the bar relative to the pads since pose X0: face-centre displacement + rim rotation.

        The pads do not move in the plane, so the bar's displacement is the relative slip. A rotation
        by phi moves the edge midpoints of the GRID footprint by RIM * phi, so the sum is the slip at
        the footprint edge in the worst direction (one number for translation and twist alike).
        """
        X = self.pose()
        R = X0.rotation().inverse().multiply(X.rotation()).matrix()
        rot = abs(np.arctan2(R[0, 2], R[2, 2]))
        d = max(np.linalg.norm((X.multiply(p) - X0.multiply(p))[[0, 2]]) for p in ([0, -BAR[1] / 2, 0], [0, BAR[1] / 2, 0]))
        return d + RIM * rot


def demand(spec, scenario, rho, d_max, t, rate=1.0):
    """Demanded grasp wrench D(t) w(t); w = (f_z, f_x, tau_y), unit force part (or w = (0, 0, rho)).

    Returns (T, D, w) with T the duration of the scenario. A: D ramps to d_max over 3 s along a fixed
    w, then holds for 1 s (the D1 hardware task). B: D oscillates between 0.3 and 1 x d_max while the
    lever arm swings between 0 and its maximum (7 s), so the binding direction changes during the hold.
    rate (S4) speeds the load up: the A ramp lasts 3 / rate s (same 1 s hold); B runs at rate x time.
    """
    if scenario == "A":  # D1-like: monotone ramp along a fixed direction, then hold
        T, D = 3.0 / rate + 1.0, d_max * min(rate * t / 3.0, 1.0)
        L = spec.lever
    else:  # B: direction oscillates between shear and torque, magnitude rises and falls
        T, t = 7.0 / rate, rate * t
        D = d_max * min(t, 1.0) * (0.65 + 0.35 * np.sin(2 * np.pi * t / 2.5))
        L = (spec.lever if np.isfinite(spec.lever) else 6.0) * (1 - np.cos(2 * np.pi * t / 3.5)) / 2
    w = np.array([0.0, 0.0, rho]) if np.isinf(L) else np.array([np.cos(spec.theta), np.sin(spec.theta), L * rho])
    return T, D, w


def direction_at_max(spec, scenario, rho, rate=1.0):
    """Load direction used to set D_max (the most torsional one of the profile)."""
    t = 0.0 if scenario == "A" else 1.75 / rate
    return demand(spec, scenario, rho, 1.0, t, rate)[2]


def episode(seed, policy="oracle", margin=0.0, scenario="A", d_max=None, record=False, f_max=F_MAX, delay=0.0,
            preview=True, rate=1.0, **scene_kw):
    """Run one closed-loop episode and return its summary (and, with record=True, the per-step log).

    seed, scenario: the pads and load profile (sample_episode, demand). policy, margin: the capacity
    estimator of policies.py and the multiplicative margin m of the command. d_max: peak demand; by
    default the load the oracle carries at F_TARGET, by linear scaling from F_MIN, so every seed
    needs a squeeze of the same order. f_max: the force budget (N). scene_kw: passed to Scene (solver
    settings, meshcat).
    Temporal settings (S4; the defaults are the S3 loop):
        delay    sensing delay (s, a multiple of DT_SUB): the policy and f_meas use the contact state of
                 t - delay (tactile image and gripper force sensor alike)
        preview  True: the load profile is known ahead and the command certifies D(t + LEAD) (S3).
                 False: the load is only measured, with the same delay, and the command certifies
                 D(t - delay), the D1 situation (a load that grows in an unknown way)
        rate     load-rate factor of `demand`
    Summary keys: fail (slip > SLIP_FAIL), slip (max, m), f_mean / f_peak (true current squeeze, N), solve
    (median decision time, s), walls / outside (max side-wall and outside-taxel force shares, validity
    filters), saturated, completed, t_flag (first request above f_max) and t_slip (first slip above
    SLIP_FAIL), NaN when the event does not occur; delay, preview, rate; and the slip-onset diagnosis
    t_onset (first slip above SLIP_ONSET), onset_ratio, cause and the shortfall terms dlog_<term> (see
    `diagnose`; "" and NaN without onset).
    Log (record=True), one entry per control step: t, D (demand used by the command), s_hat, f_meas,
    f_req (requested squeeze before clipping), f_cmd, slip, solve, walls, outside, f_true (current
    squeeze; f_meas is the sensed one, older by the delay); and, for the viewer,
    D_now (demand applied at t), s_true (oracle of the current contact along the applied w), w (load
    direction of the binding estimate) and state (the current PadContact list with face triangles).
    With a Meshcat in scene_kw, the run is recorded from the start (see viz.save_html).
    """
    n_sub = int(round(DT_CTRL / DT_SUB))
    n_delay = int(round(delay / DT_SUB))
    if abs(n_delay * DT_SUB - delay) > 1e-9:
        raise ValueError(f"delay must be a multiple of DT_SUB = {DT_SUB} s")
    spec = sample_episode(seed)
    scene = Scene(spec.pads, **scene_kw)
    if scene.visualizer is not None:
        scene.visualizer.StartRecording()
    scene.set_squeeze(F_MIN)
    scene.advance(0.5, F_MIN)  # settle
    st = contact_state(scene.plant, scene.ctx)
    pts = np.vstack([pad.points for pad in st]); frc = np.concatenate([pad.forces for pad in st])
    c = frc @ pts / frc.sum()
    rho = np.sqrt(frc @ np.sum((pts - c) ** 2, axis=1) / frc.sum())  # RMS contact radius at F_MIN
    if d_max is None:  # oracle capacity at F_MIN, scaled linearly to F_TARGET
        d_max = oracle(st, direction_at_max(spec, scenario, rho, rate), MU) * F_TARGET / F_MIN
    load = lambda t: demand(spec, scenario, rho, d_max, max(t, 0.0), rate)  # noqa: E731  (no load before t0)
    X0, T = scene.pose(), load(0.0)[0]
    t0, f_cmd = scene.t, F_MIN
    # delayed sensing: states sampled at sub-step index (decision index) - n_delay; before t0, the settled state
    settled = contact_state(scene.plant, scene.ctx, triangles=record) if n_delay else None
    samples, i_sub = {}, 0
    log = dict(t=[], D=[], s_hat=[], f_meas=[], f_req=[], f_cmd=[], slip=[], solve=[], walls=[], outside=[], f_true=[])
    extra = dict(D_now=[], s_true=[], w=[], state=[])  # record=True only
    onset, prev = None, None
    while scene.t - t0 < T - 1e-9:
        t = scene.t - t0
        st = contact_state(scene.plant, scene.ctx, triangles=record)
        sensed = st if n_delay == 0 else samples.pop(i_sub - n_delay, settled)
        f_meas = np.mean([pad.forces.sum() for pad in sensed])  # the two pads see the same squeeze up to dynamics
        tic = time.perf_counter()
        if preview:
            loads = [load(t + LEAD)[1:]]
            if scenario != "A":  # the direction changes: also certify the current load
                loads.append(load(t)[1:])
        else:
            loads = [load(t - delay)[1:]]  # the measured load, as old as the tactile state
        target, s_hat, w_bind, D_bind = 0.0, np.inf, None, np.nan
        for D, w in loads:
            s = estimate(policy, sensed, w, MU)
            # linear scaling: the squeeze that would make s_hat = D (1 + m); inf if nothing is certified
            target = max(target, np.inf if not s > 0 else f_meas * D * (1 + margin) / s)
            if s < s_hat:
                s_hat, w_bind, D_bind = s, w, D
        solve = (time.perf_counter() - tic) / len(loads)
        D = loads[-1][0]
        f_cmd_prev, f_cmd = f_cmd, float(np.clip(max(target, (1 - RELAX) * f_cmd), F_MIN, f_max))
        slip = scene.slip(X0)
        f_true = np.mean([pad.forces.sum() for pad in st])  # current squeeze (= f_meas without delay)
        for key, v in zip(log, (t, D, s_hat, f_meas, target, f_cmd, slip, solve, max(pad.walls for pad in st),
                                outside_share(st), f_true)):
            log[key].append(v)
        if record:
            _, D_now, w_now = load(t)
            extra["D_now"].append(D_now)
            extra["s_true"].append(oracle(st, w_now, MU))
            extra["w"].append(w_bind)
            extra["state"].append(st)
        decision = (sensed, f_meas, w_bind, D_bind, s_hat)
        if onset is None and slip > SLIP_ONSET:  # diagnose the decision that was active while the slip began
            onset = (t, *diagnose(st, load(t)[1:], prev if prev is not None else decision, f_cmd_prev, margin))
        prev = decision
        if slip > SLIP_STOP:
            break
        # load: piecewise constant over DT_SUB, applied as the external wrench -D w
        for k in range(n_sub):
            _, D, w = load(t + k * DT_SUB)
            scene.set_load(-D * w)
            scene.advance(DT_SUB, f_cmd)
            i_sub += 1
            if n_delay and (i_sub + n_delay) % n_sub == 0:  # this state is read n_delay sub-steps later
                samples[i_sub] = contact_state(scene.plant, scene.ctx, triangles=record)
    log = {k: np.asarray(v) for k, v in log.items()}
    first = lambda m: float(log["t"][np.argmax(m)]) if m.any() else np.nan  # noqa: E731
    out = dict(seed=seed, family=spec.family, policy=policy, margin=margin, scenario=scenario, d_max=d_max, rho=rho,
               fail=bool(log["slip"].max() > SLIP_FAIL), slip=log["slip"].max(), f_mean=log["f_true"].mean(),
               f_peak=log["f_true"].max(), solve=np.median(log["solve"]), walls=log["walls"].max(),
               outside=log["outside"].max(), saturated=bool(np.any(log["f_cmd"] >= f_max)), completed=bool(log["slip"].max() <= SLIP_STOP),
               f_max=f_max, t_flag=first(log["f_req"] > f_max), t_slip=first(log["slip"] > SLIP_FAIL),
               delay=delay, preview=preview, rate=rate, t_onset=onset[0] if onset else np.nan,
               onset_ratio=onset[1] if onset else np.nan, cause=onset[2] if onset else "",
               **{f"dlog_{k}": (onset[3][k] if onset else np.nan) for k in CAUSE_TERMS + ("slack",)})
    if record:
        out["log"] = {**log, "D_now": np.asarray(extra["D_now"]), "s_true": np.asarray(extra["s_true"]),
                      "w": extra["w"], "state": extra["state"]}
    if scene.visualizer is not None:
        out["scene"] = scene  # keeps the visualizer (recording) and the Meshcat alive for the caller
    return out


def diagnose(state, applied, decision, f_cmd, margin):
    """Why a slip began: (onset_ratio, cause, terms), from the current contact and the decision in force.

    state: current contact; applied: (D_now, w_now), the load applied now; decision: (sensed state,
    f_sens, w, D, s_hat) of the last control step before the onset, which commanded f_cmd.
    onset_ratio = s_true(now) / D_now, the ideal Coulomb capacity of the current contact over the
    applied load. With kappa = s_true / f the true capacity per newton along w, and the command
    target f_sens D (1 + m) / s_hat of the decision, the shortfall splits exactly into log terms:
        ln(D_now / s_true_now) = estimate + load + state + actuator + slack - ln(1 + m)
        estimate  ln(s_hat / s_true(sensed))     the estimator's optimism on its own information
        load      ln(D_now / D)                  the load grew since the one that was certified
        state     ln(kappa_sensed / kappa_now)   the contact (capacity per newton) changed since sensing
        actuator  ln(f_cmd / f_now)              the gripper had not reached the command
        slack     ln(target / f_cmd) <= 0        the floor F_MIN or the RELAX limiter kept the command
                                                 above the target (extra capacity, not a cause)
    cause: "model" if onset_ratio >= 1 (Coulomb says the contact holds: the slip comes from the
    simulated friction dynamics, as for the S3 zero-margin oracle), else the largest of the four
    cause terms; "stale" when the decision had no load to certify (with a measured load, none was
    seen yet) so that the terms are undefined. The load
    and state terms compare the same direction w, which holds in scenario A (fixed direction).
    """
    D_now, w_now = applied
    s_now = oracle(state, w_now, MU)
    f_now = np.mean([pad.forces.sum() for pad in state])
    ratio = s_now / D_now if D_now > 0 else np.inf
    sensed, f_sens, w, D, s_hat = decision
    nan = dict.fromkeys(CAUSE_TERMS + ("slack",), np.nan)
    if w is None or not np.isfinite(s_hat) or not D > 0:
        return ratio, "model" if ratio >= 1 else "stale", nan
    s_sens = oracle(sensed, w, MU)
    terms = dict(estimate=np.log(s_hat / s_sens), load=np.log(D_now / D),
                 state=np.log((s_sens / f_sens) / (s_now / f_now)), actuator=np.log(f_cmd / f_now),
                 slack=np.log(f_sens * D * (1 + margin) / (s_hat * f_cmd)))
    cause = "model" if ratio >= 1 else max(CAUSE_TERMS, key=terms.get)
    return ratio, cause, terms


def squeeze_limit(seed, rate=50.0):
    """Squeeze at which a pad of this seed reaches DEPTH_MAX of its semi-thickness (static ramp, no load).

    Beyond it the hydroelastic force-depth law of a thin pad flattens and then drops (the rigid face
    passes the pad's medial axis: the pad "bottoms out"), which is outside the scope of the model.
    The squeeze is ramped at `rate` (N/s), i.e. 2.5 N per gripper time constant TAU_GRIP.
    """
    scene = Scene(sample_episode(seed).pads)
    f = F_MIN
    while f < F_MAX:
        scene.advance(0.02, f)
        if scene.depth() > DEPTH_MAX:
            return scene.f_app
        f += rate * 0.02
    return np.inf


def outside_share(state):
    """Share of the normal force on faces outside the taxel footprints (they would be clipped)."""
    pts = np.vstack([pad.points for pad in state]); frc = np.concatenate([pad.forces for pad in state])
    return frc[np.any(np.abs(pts) > GRID / 2, axis=1)].sum() / frc.sum()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default="oracle", choices=list(POLICIES))
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--scenario", default="A", choices=["A", "B"])
    a = ap.parse_args()
    tic = time.time()
    r = episode(a.seed, a.policy, a.margin, a.scenario, record=True)
    print(sample_episode(a.seed))
    lg = r.pop("log")
    for i in range(0, len(lg["t"]), 10):
        print(f"  t={lg['t'][i]:.2f}  D={lg['D'][i]:.3f}  s_hat={lg['s_hat'][i]:.3f}  f_meas={lg['f_meas'][i]:.2f}"
              f"  f_cmd={lg['f_cmd'][i]:.2f}  slip={1e3 * lg['slip'][i]:.3f} mm  solve={1e3 * lg['solve'][i]:.0f} ms")
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, f"wall {time.time() - tic:.0f} s")
