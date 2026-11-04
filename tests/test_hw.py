"""Checks of the hardware-analysis pipeline (hw/) on synthetic data with known answers."""

import json

import numpy as np
import pytest

from hw import boss_solve, frames, latency, pressure_fit

RNG = np.random.default_rng(0)


# --- frames ---------------------------------------------------------------------------------
def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def test_point_force_has_no_moment_about_its_point():
    r = np.array([0.004, -0.001, -0.003])
    f = np.array([0.3, -2.0, 0.1])
    W0 = np.r_[f, np.cross(r, f)]               # wrench of the point force about the origin
    assert np.allclose(frames.shift_reference(W0, r)[3:], 0.0)


def test_transform_matches_direct_computation_and_inverts():
    R, t = rot_y(0.4) @ frames.quat_to_matrix([0.1, 0.2, 0.3, 0.9]), np.array([0.01, 0.02, -0.03])
    r_A, f_A = np.array([0.003, 0.001, -0.002]), np.array([0.5, -1.5, 0.2])
    W_A = np.r_[f_A, np.cross(r_A, f_A)]
    r_B, f_B = R @ r_A + t, R @ f_A
    assert np.allclose(frames.transform_wrench(W_A, R, t), np.r_[f_B, np.cross(r_B, f_B)])
    Ri, ti = frames.invert(R, t)
    assert np.allclose(frames.transform_wrench(frames.transform_wrench(W_A, R, t), Ri, ti), W_A)


# --- boss_solve ------------------------------------------------------------------------------
def surface_wrench(positions, F, ft, y_c):
    """Exact wrench about the surface origin of normal loads F_i and tangential loads ft_i = (fx, fz)
    applied at (x_i, y_c, z_i)."""
    W = np.zeros(6)
    for (x, z), Fi, (fx, fz) in zip(positions, F, ft):
        r, f = np.array([x, y_c, z]), np.array([fx, Fi, fz])
        W += np.r_[f, np.cross(r, f)]
    return W


TRI = np.array([[0.004, 0.0], [-0.002, 0.0035], [-0.002, -0.0035]])  # 3 bosses, ~4 mm from the centre


@pytest.mark.parametrize("n", [1, 2, 3])
def test_exact_recovery_with_tangential_loads_in_an_offset_plane(n):
    P = TRI[:n]
    F = -RNG.uniform(0.5, 3.0, n)                    # compression negative (legacy convention)
    ft = RNG.normal(0.0, 0.3, (n, 2))                # arbitrary distribution of tangential load
    y_c = -0.0008                                    # 0.8 mm indentation
    W = surface_wrench(P, F, ft, y_c)
    sol = boss_solve.solve(P, W, y_c=y_c)
    assert np.allclose(sol.F, F, atol=1e-9)
    assert sol.residual < 1e-9 and sol.rank == n


def test_torsion_is_not_used():
    F, ft = -np.array([1.0, 2.0, 1.5]), RNG.normal(0, 0.2, (3, 2))
    W = surface_wrench(TRI, F, ft, 0.0)
    W2 = W.copy()
    W2[4] += 0.05
    assert np.allclose(boss_solve.solve(TRI, W).F, boss_solve.solve(TRI, W2).F)


def test_ignored_plane_offset_biases_the_moments_by_y_c_times_tangential_total():
    """Neglecting y_c (true contact plane below the assumed surface) errs by about y_c F_t / d."""
    F = -np.array([1.0, 2.0, 1.5])
    ft = np.tile([0.2, 0.0], (3, 1))                 # 0.6 N total along x
    y_c = -0.001
    err = boss_solve.solve(TRI, surface_wrench(TRI, F, ft, y_c), y_c=0.0).F - F
    d = np.linalg.norm(TRI, axis=1).mean()
    assert 0.1 * abs(y_c) * 0.6 / d < np.abs(err).max() < 10 * abs(y_c) * 0.6 / d


def test_two_bosses_residual_flags_an_inconsistent_wrench():
    P, F = TRI[:2], -np.array([1.0, 2.0])
    W = surface_wrench(P, F, np.zeros((2, 2)), 0.0)
    assert boss_solve.solve(P, W).residual < 1e-9
    W[5] += 0.004                                     # a load outside the two bosses
    assert boss_solve.solve(P, W).residual > 0.1


def test_collinear_three_bosses_are_rank_deficient():
    P = np.array([[-0.004, 0.0], [0.0, 0.0], [0.004, 0.0]])
    sol = boss_solve.solve(P, surface_wrench(P, -np.ones(3), np.zeros((3, 2)), 0.0))
    assert sol.rank == 2 and not np.isfinite(sol.cond)


def test_covariance_matches_monte_carlo():
    F = -np.array([1.0, 2.0, 1.5])
    W = surface_wrench(TRI, F, np.zeros((3, 2)), 0.0)
    sf, st = 0.02, 5e-4                               # [N], [N m]
    cov = boss_solve.solve(TRI, W, sigma_force=sf, sigma_torque=st).cov
    samples = []
    for _ in range(4000):
        Wn = W + np.r_[0, RNG.normal(0, sf), 0, RNG.normal(0, st), 0, RNG.normal(0, st)]
        samples.append(boss_solve.solve(TRI, Wn).F)
    assert np.allclose(np.cov(np.array(samples).T), cov, rtol=0.15, atol=1e-5)


def test_cop_and_registration():
    x, z = 0.003, -0.002
    W = surface_wrench([[x, z]], [-2.0], [[0.1, -0.2]], -0.0005)
    assert np.allclose(boss_solve.cop(W, y_c=-0.0005), [x, z])
    M_true = np.array([[6.34e-5, 1e-6, -0.0101], [-2e-6, -6.34e-5, 0.0076]])
    px = RNG.uniform(0, 320, (12, 2))
    pts = boss_solve.apply_registration(M_true, px) + RNG.normal(0, 2e-5, (12, 2))
    M, rms = boss_solve.fit_registration(px, pts)
    assert np.allclose(M[:, :2], M_true[:, :2], atol=1e-6)   # scale/rotation [m/px]
    assert np.allclose(M[:, 2], M_true[:, 2], atol=5e-5)     # offset [m], noise 2e-5 m per point
    assert rms < 4e-5


# --- pressure_fit ----------------------------------------------------------------------------
def synthetic_sample(centres_px, depths_mm, radius_px, law, mm_per_px=0.0634, shape=(240, 320)):
    """Spherical-cap indentations at the given pixel centres; boss loads from the given pressure law."""
    v, u = np.indices(shape)
    depth = np.zeros(shape)
    regions = []
    for (cu, cv), d0 in zip(centres_px, depths_mm):
        r2 = (u - cu) ** 2 + (v - cv) ** 2
        cap = np.clip(d0 * (1 - r2 / radius_px ** 2), 0.0, None)
        depth = np.maximum(depth, cap)
        regions.append(cap > 0)
    F = np.array([law(depth[r]).sum() * mm_per_px ** 2 for r in regions])
    return pressure_fit.Sample(depth, regions, F, mm_per_px)


CENTRES = [(110, 120), (200, 80), (200, 160)]


def samples_from(law):
    return [synthetic_sample(CENTRES, RNG.uniform(0.2, 1.0, 3), 25, law) for _ in range(6)]


def test_winkler_recovered():
    S = samples_from(lambda d: 0.8 * d)
    fit = pressure_fit.fit_winkler(S)
    assert fit.params["k"] == pytest.approx(0.8) and fit.rel_rms < 1e-9


def test_power_law_recovered():
    S = samples_from(lambda d: 0.5 * d ** 1.6)
    fit = pressure_fit.fit_power(S)
    assert fit.params["alpha"] == pytest.approx(1.6, abs=1e-3)
    assert fit.params["kappa"] == pytest.approx(0.5, rel=1e-3)


def test_kernel_reproduces_a_local_law():
    S = samples_from(lambda d: 0.8 * d)
    fit = pressure_fit.fit_kernel(S, radius_px=3, ridge=1e-8)
    assert fit.rel_rms < 0.02


def test_region_labelling_and_matching():
    s = synthetic_sample(CENTRES, [0.5, 0.6, 0.7], 20, lambda d: d)
    regs = pressure_fit.label_regions(np.any(s.regions, axis=0))
    assert len(regs) == 3
    order = pressure_fit.match_regions([c for _, c, _ in regs], np.array(CENTRES, float))
    for boss, k in enumerate(order):
        assert np.allclose(regs[k][1], CENTRES[boss], atol=0.5)


# --- latency ---------------------------------------------------------------------------------
def test_lag_and_onsets_recover_the_camera_delay():
    d_cam = 0.037
    t_f = np.arange(0.0, 30.0, 0.001)
    force = np.zeros_like(t_f)
    for t0 in np.arange(2.0, 28.0, 3.0):             # presses: 0.5 s ramp, 1 s hold, 0.5 s release
        force += np.interp(t_f, [t0, t0 + 0.5, t0 + 1.5, t0 + 2.0], [0, 3.0, 3.0, 0], left=0, right=0)
    force += RNG.normal(0, 0.005, t_f.size)
    t_img = np.arange(0.013, 30.0, 0.04)             # grab stamps; the frame shows the force d_cam earlier
    activity = 4.0 * np.sqrt(np.clip(np.interp(t_img - d_cam, t_f, force), 0, None)) + RNG.normal(0, 0.02, t_img.size)
    best, _, _ = latency.estimate_lag(t_img, activity, t_f, force)
    assert best == pytest.approx(d_cam, abs=0.003)

    zf = latency.baseline_normalize(t_f, force, (0.0, 1.5))
    za = latency.baseline_normalize(t_img, activity, (0.0, 1.5))
    on_f = latency.rising_crossings(t_f, zf, 10.0)
    on_a = latency.rising_crossings(t_img, za, 10.0)
    delays = latency.pair_onsets(on_f, on_a)
    assert len(delays) == len(on_f) == 9
    assert np.median(delays) == pytest.approx(d_cam, abs=0.02)  # frame quantization, threshold mismatch


# --- io (round trip through a synthetic ROS1 bag) --------------------------------------------
def test_session_reads_a_ros1_bag(tmp_path):
    rosbag1 = pytest.importorskip("rosbags.rosbag1")
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore
    from hw.io import Session

    ts = get_typestore(Stores.ROS1_NOETIC)
    # not in the built-in store; real bags carry their own definitions, which the reader uses
    ts.register(get_types_from_msg("geometry_msgs/TransformStamped[] transforms", "tf2_msgs/msg/TFMessage"))
    T = ts.types
    Time = T["builtin_interfaces/msg/Time"]
    Header = T["std_msgs/msg/Header"]
    V3, Q = T["geometry_msgs/msg/Vector3"], T["geometry_msgs/msg/Quaternion"]

    def hdr(t, frame="", seq=0):
        return Header(seq=seq, stamp=Time(sec=int(t), nanosec=int(round((t % 1) * 1e9))), frame_id=frame)

    def tfs(parent, child, xyz, q):
        return T["geometry_msgs/msg/TransformStamped"](
            header=hdr(0.0, parent), child_frame_id=child,
            transform=T["geometry_msgs/msg/Transform"](translation=V3(x=xyz[0], y=xyz[1], z=xyz[2]),
                                                       rotation=Q(x=q[0], y=q[1], z=q[2], w=q[3])))

    s2 = np.sqrt(0.5)
    path = tmp_path / "session.bag"
    with rosbag1.Writer(path) as w:
        cw = w.add_connection("/ati_ft_sensor/sensor_wrench_corrected", "geometry_msgs/msg/WrenchStamped", typestore=ts)
        ci = w.add_connection("/gelsight/visuotactile_image", "sensor_msgs/msg/Image", typestore=ts)
        ct = w.add_connection("/tf_static", "tf2_msgs/msg/TFMessage", typestore=ts)
        ce = w.add_connection("/experiment/event", "std_msgs/msg/String", typestore=ts)
        tf_msg = T["tf2_msgs/msg/TFMessage"](transforms=[
            tfs("ati", "gelsight", (0, 0, 0.01), (0, 0, 0, 1)),
            tfs("gelsight", "gelsight_contact_surface", (0, 0, 0.03), (s2, 0, 0, s2)),  # +90 deg about x
        ])
        w.write(ct, 1, ts.serialize_ros1(tf_msg, "tf2_msgs/msg/TFMessage"))
        for k in range(5):  # written out of order on purpose: the reader must sort by header stamp
            t = 10.0 + 0.001 * (4 - k)
            msg = T["geometry_msgs/msg/WrenchStamped"](
                header=hdr(t, "ati"),
                wrench=T["geometry_msgs/msg/Wrench"](force=V3(x=0.0, y=0.0, z=-float(4 - k)), torque=V3(x=0.0, y=0.0, z=0.0)))
            w.write(cw, int(t * 1e9) + 5_000_000, ts.serialize_ros1(msg, "geometry_msgs/msg/WrenchStamped"))
        img = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        im = T["sensor_msgs/msg/Image"](header=hdr(10.02, "gelsight_camera", seq=7), height=4, width=6,
                                        encoding="rgb8", is_bigendian=0, step=18, data=img.ravel())
        w.write(ci, int(10.03e9), ts.serialize_ros1(im, "sensor_msgs/msg/Image"))
        ev = T["std_msgs/msg/String"](data=json.dumps({"stamp": 10.01, "label": "dwell_end", "value": 2.0, "info": ""}))
        w.write(ce, int(10.01e9), ts.serialize_ros1(ev, "std_msgs/msg/String"))

    (tmp_path / "manifest.json").write_text(json.dumps({"timing": {"d_cam_s": 0.005}}))
    s = Session(tmp_path)
    t, W = s.wrench("/ati_ft_sensor/sensor_wrench_corrected")
    assert np.all(np.diff(t) > 0) and np.allclose(W[:, 2], [0, -1, -2, -3, -4])
    (ti, seq, im_rd), = list(s.images())
    assert ti == pytest.approx(10.015) and seq == 7 and np.array_equal(im_rd, img)
    assert s.events()[0]["label"] == "dwell_end"
    R, p = s.static_transform("ati", "gelsight_contact_surface")
    assert np.allclose(p, [0, 0, 0.04]) and np.allclose(R @ [0, 1, 0], [0, 0, 1])


def test_unsynchronized_sessions_are_refused(tmp_path):
    from hw.io import Session

    (tmp_path / "x.bag").write_bytes(b"")  # Session only needs a bag path to exist here
    (tmp_path / "manifest.json").write_text(json.dumps({"timing": {"synchronized": False, "sync_problem": "offset"}}))
    with pytest.raises(RuntimeError, match="not clock-synchronized"):
        Session(tmp_path).require_synchronized()
    Session(tmp_path).require_synchronized(allow_unsynced=True)  # warns only
    (tmp_path / "manifest.json").write_text(json.dumps({"timing": {"synchronized": True}}))
    Session(tmp_path).require_synchronized()


def test_lag_scan_needs_press_windows_when_the_record_is_sparse():
    """Sparse presses plus baseline drift: a whole-record rank correlation is swamped; windows fix it.

    Measured on a real H1 session (20261027_193914_h1_latency_hand, 19 paired presses): over the
    whole record rho_max = -0.048 at 0.144 s, i.e. the lag was the argmax of noise; the same data
    restricted to the press windows gave rho_max = +0.788 at 0.114 s, agreeing with the onset
    median (0.124 s) to 10 ms.
    """
    d_cam = 0.120
    t_f = np.arange(0.0, 200.0, 0.002)
    force = np.zeros_like(t_f)
    onsets = np.arange(5.0, 195.0, 10.0)   # 19 presses, >= 8 s unloaded between them: ~20% duty cycle
    for t0 in onsets:
        force += np.interp(t_f, [t0, t0 + 0.4, t0 + 1.4, t0 + 1.8], [0, 5.0, 5.0, 0], left=0, right=0)
    force += RNG.normal(0, 0.01, t_f.size)

    t_img = np.arange(0.011, 200.0, 0.04)  # grab stamps; the frame shows the force d_cam earlier
    act = 4.0 * np.sqrt(np.clip(np.interp(t_img - d_cam, t_f, force), 0, None))
    # Unloaded frames are not independent noise in practice: slow illumination and gel drift dominate,
    # and there are ~4x more of them than loaded frames.
    act = act + 0.5 * np.sin(2 * np.pi * t_img / 60.0) + RNG.normal(0, 0.05, t_img.size)

    # POST spans ramp + hold + release + d_cam; cutting the release biases the lag, because the
    # image trails the force and the truncation is then asymmetric between the two signals.
    windows = [(o - 0.3, o + 3.0) for o in onsets]
    _, _, rho_all = latency.estimate_lag(t_img, act, t_f, np.abs(force))
    lag_win, _, rho_win = latency.estimate_lag(t_img, act, t_f, np.abs(force), windows=windows)

    assert lag_win == pytest.approx(d_cam, abs=0.005)
    assert np.nanmax(rho_win) > 0.6
    assert np.nanmax(rho_win) > np.nanmax(rho_all)   # the windows are what make the scan informative


def test_estimate_lag_refuses_windows_that_select_too_little():
    t_img = np.arange(0.0, 1.0, 0.04)
    t_f = np.arange(0.0, 1.0, 0.002)
    with pytest.raises(ValueError, match="too few"):
        latency.estimate_lag(t_img, np.zeros_like(t_img), t_f, np.zeros_like(t_f),
                             windows=[(0.5, 0.52)])


# --- H3 friction coefficient ------------------------------------------------------------------
def synthetic_slide(mu_s=1.1, mu_k=0.9, fn=2.0, direction=90.0, t0=1.0, t_peak=2.0, t_slip=2.3, t_end=3.5):
    """Surface wrenches of one slide: elastic rise to the static peak, drop to the kinetic plateau."""
    t = np.arange(0.0, 4.0, 0.002)
    ft = np.interp(t, [t0, t_peak, t_peak + 0.1, t_end - 0.5, t_end], [0.0, mu_s * fn, mu_k * fn, mu_k * fn, 0.3])
    d = np.radians(direction)
    W = np.zeros((t.size, 6))
    W[:, 0], W[:, 2], W[:, 1] = -np.cos(d) * ft, -np.sin(d) * ft, fn   # friction opposes the slide
    return t, W + 0.005 * RNG.normal(size=W.shape)


def test_slide_mu_separates_static_and_kinetic():
    from hw import h3_mu

    t, W = synthetic_slide()
    ms, mk = h3_mu.slide_mu(t, W, 90.0, 1.0, 2.3, 3.5)
    assert ms == pytest.approx(1.1, abs=0.02) and mk == pytest.approx(0.9, abs=0.01)
    assert np.isnan(h3_mu.slide_mu(t, W, 90.0, 1.0, None, 3.5)[1])


def test_h3_session_from_events(tmp_path):
    rosbag1 = pytest.importorskip("rosbags.rosbag1")
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore
    from hw import h3_mu
    from hw.io import Session

    ts = get_typestore(Stores.ROS1_NOETIC)
    ts.register(get_types_from_msg("geometry_msgs/TransformStamped[] transforms", "tf2_msgs/msg/TFMessage"))
    T = ts.types
    V3 = T["geometry_msgs/msg/Vector3"]

    def hdr(t, frame=""):
        return T["std_msgs/msg/Header"](seq=0, stamp=T["builtin_interfaces/msg/Time"](
            sec=int(t), nanosec=int(round((t % 1) * 1e9))), frame_id=frame)

    with rosbag1.Writer(tmp_path / "s.bag") as w:
        cw = w.add_connection("/ati_ft_sensor/sensor_wrench_corrected", "geometry_msgs/msg/WrenchStamped", typestore=ts)
        ct = w.add_connection("/tf_static", "tf2_msgs/msg/TFMessage", typestore=ts)
        ce = w.add_connection("/experiment/event", "std_msgs/msg/String", typestore=ts)
        identity = T["geometry_msgs/msg/TransformStamped"](
            header=hdr(0.0, "ati"), child_frame_id="gelsight_contact_surface",
            transform=T["geometry_msgs/msg/Transform"](translation=V3(x=0.0, y=0.0, z=0.0),
                                                       rotation=T["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0)))
        w.write(ct, 1, ts.serialize_ros1(T["tf2_msgs/msg/TFMessage"](transforms=[identity]), "tf2_msgs/msg/TFMessage"))
        events = []
        for k, (direction, slipped) in enumerate([(0.0, True), (90.0, True), (180.0, False)]):
            base = 10.0 + 5.0 * k
            t, W = synthetic_slide(direction=direction)   # slide at 1.0-3.5 s, slip at 2.3 s, shifted by base
            for tk, wk in zip(t[::5] + base, W[::5]):
                msg = T["geometry_msgs/msg/WrenchStamped"](header=hdr(tk, "ati"), wrench=T["geometry_msgs/msg/Wrench"](
                    force=V3(x=wk[0], y=wk[1], z=wk[2]), torque=V3(x=wk[3], y=wk[4], z=wk[5])))
                w.write(cw, int(tk * 1e9), ts.serialize_ros1(msg, "geometry_msgs/msg/WrenchStamped"))
            params = {"test": "h3", "normal_force": 2.0, "direction": direction, "speed": 0.0005, "repeat": 0}
            events += [(base + 1.0, "trial_start", params)] + ([(base + 2.3, "slip", {})] if slipped else []) \
                + [(base + 3.5, "slide_end", {})]
        for stamp, label, info in events:
            ev = json.dumps({"stamp": stamp, "label": label, "value": None, "info": info, "trial_id": "h3_x_c"})
            w.write(ce, int(stamp * 1e9), ts.serialize_ros1(T["std_msgs/msg/String"](data=ev), "std_msgs/msg/String"))

    rows = h3_mu.analyse(Session(tmp_path))
    assert [r["slipped"] for r in rows] == [True, True, False]
    s = h3_mu.summarize(rows)
    assert s["mu_min"] == pytest.approx(0.9, abs=0.01) and s["excluded_no_slip"] == 1
    assert s["by_load"][0]["n"] == 2
