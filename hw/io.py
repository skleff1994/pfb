"""Read the bags of one session with `rosbags` (pure Python, no ROS install; reads ROS1 and ROS2 bags).

A session directory (written by gelsight_ros/record_session.py) holds manifest.json, the GelSight-side
bag and, copied next to it, the controller-side bag(s) of the same session (e.g. from run_trial.sh).
All bags are read together; samples are timed by their header stamps (chrony-synchronized across
hosts), or by the bag receive time for unstamped topics (flagged in `Series.stamped`).

    s = Session("~/friction_sessions/20261103_101500_h2_pilot")
    t, W = s.wrench("/ati_ft_sensor/sensor_wrench_corrected")
    for t, seq, rgb in s.images(): ...
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from .frames import quat_to_matrix


@dataclass
class Series:
    t: np.ndarray       # (N,) [s]
    x: np.ndarray       # (N, ...) samples
    stamped: bool       # True: header stamps; False: bag receive time

    def __iter__(self):  # allows  t, x = series
        return iter((self.t, self.x))


def _stamp(msg):
    return msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec


class Session:
    def __init__(self, session_dir, bags=None):
        self.dir = Path(session_dir).expanduser()
        self.bags = [Path(b).expanduser() for b in bags] if bags else sorted(self.dir.glob("*.bag"))
        if not self.bags:
            raise FileNotFoundError(f"no bag in {self.dir}")
        manifest = self.dir / "manifest.json"
        self.manifest = json.loads(manifest.read_text()) if manifest.exists() else {}
        self._typestore = get_typestore(Stores.ROS1_NOETIC)

    def require_synchronized(self, allow_unsynced=False):
        """Refuse cross-host timing on a session not recorded under chrony synchronization.

        Image stamps (GPU host) and ATI/controller stamps (RT host) are only comparable if
        record_session.py found the clocks synchronized at start and end (manifest timing.synchronized).
        """
        timing = self.manifest.get("timing", {})
        if timing.get("synchronized") is True:
            return
        reason = timing.get("sync_problem") or ("no manifest" if not self.manifest else "no synchronization record")
        if allow_unsynced:
            print(f"WARNING: {self.dir.name} is not clock-synchronized ({reason}); cross-host timing is unreliable")
            return
        raise RuntimeError(f"{self.dir.name}: not clock-synchronized ({reason}); refusing cross-host timing")

    # -- generic access ----------------------------------------------------------------------
    def _messages(self, topic):
        with AnyReader(self.bags, default_typestore=self._typestore) as reader:
            conns = [c for c in reader.connections if c.topic == topic]
            for conn, t_ns, raw in reader.messages(connections=conns):
                yield t_ns * 1e-9, reader.deserialize(raw, conn.msgtype)

    def topics(self):
        """{topic: (msgtype, message count)} over all bags."""
        with AnyReader(self.bags, default_typestore=self._typestore) as reader:
            out = {}
            for c in reader.connections:
                n = out.get(c.topic, (c.msgtype, 0))[1] + c.msgcount
                out[c.topic] = (c.msgtype, n)
            return out

    def _series(self, topic, extract):
        t, x, stamped = [], [], True
        for t_bag, msg in self._messages(topic):
            if hasattr(msg, "header"):
                t.append(_stamp(msg))
            else:
                t.append(t_bag)
                stamped = False
            x.append(extract(msg))
        if not t:
            raise KeyError(f"topic {topic} not found or empty in {[b.name for b in self.bags]}")
        t = np.asarray(t)
        order = np.argsort(t, kind="stable")  # bags are merged; keep time order
        return Series(t[order], np.asarray(x)[order], stamped)

    # -- typed access ------------------------------------------------------------------------
    def wrench(self, topic):
        """(N, 6) = (fx, fy, fz, tx, ty, tz) from geometry_msgs/WrenchStamped."""
        def ex(m):
            w = m.wrench
            return [w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z]
        return self._series(topic, ex)

    def pose(self, topic):
        """(N, 7) = (x, y, z, qx, qy, qz, qw) from geometry_msgs/PoseStamped."""
        def ex(m):
            p, q = m.pose.position, m.pose.orientation
            return [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        return self._series(topic, ex)

    def vector3(self, topic):
        """(N, 3) from geometry_msgs/Vector3Stamped or PointStamped."""
        def ex(m):
            v = m.vector if hasattr(m, "vector") else m.point
            return [v.x, v.y, v.z]
        return self._series(topic, ex)

    def scalar(self, topic):
        """(N,) from std_msgs Float32/Float64/Bool (unstamped: bag receive time)."""
        return self._series(topic, lambda m: float(m.data))

    def events(self):
        """Operator/procedure events from /experiment/event (JSON strings), in time order."""
        try:
            return sorted((json.loads(m.data) for _, m in self._messages("/experiment/event")),
                          key=lambda e: e["stamp"])
        except KeyError:
            return []

    def latched_json(self, topic):
        for _, m in self._messages(topic):
            return json.loads(m.data)
        return None

    # -- images ------------------------------------------------------------------------------
    def d_cam(self, override=None):
        """Camera delay [s] (exposure -> grab) from the manifest (H1), or the override, or 0."""
        if override is not None:
            return float(override)
        return float(self.manifest.get("timing", {}).get("d_cam_s") or 0.0)

    def images(self, topic="/gelsight/visuotactile_image", t0=None, t1=None, d_cam=None):
        """Yield (acquisition time, seq, image (H, W, 3) uint8 in the message's encoding order).

        Acquisition time = header stamp (grab time) - d_cam.
        """
        dc = self.d_cam(d_cam)
        for _, m in self._messages(topic):
            t = _stamp(m) - dc
            if (t0 is not None and t < t0) or (t1 is not None and t > t1):
                continue
            yield t, m.header.seq, _image_array(m)

    def image_times(self, topic="/gelsight/visuotactile_image", d_cam=None):
        dc = self.d_cam(d_cam)
        ts = [(_stamp(m) - dc, m.header.seq) for _, m in self._messages(topic)]
        t, seq = (np.asarray(v) for v in zip(*ts)) if ts else (np.empty(0), np.empty(0))
        return t, seq

    def background(self):
        """Reconstruction background as BGR uint8 (from the bag; else background.png)."""
        for _, m in self._messages("/gelsight/background"):
            img = _image_array(m)
            return img[..., ::-1].copy() if m.encoding == "rgb8" else img
        png = self.dir / "background.png"
        if png.exists():
            import matplotlib.image as mpimg
            rgb = mpimg.imread(png)
            rgb = (rgb[..., :3] * 255).round().astype(np.uint8) if rgb.dtype != np.uint8 else rgb[..., :3]
            return rgb[..., ::-1].copy()
        raise FileNotFoundError("no background in the bags or the session directory")

    # -- static geometry ---------------------------------------------------------------------
    def static_transform(self, target, source):
        """Pose (R, t) of frame `source` in frame `target` (x_target = R x_source + t), from /tf_static."""
        edges = {}
        for _, m in self._messages("/tf_static"):
            for tf in m.transforms:
                q = tf.transform.rotation
                p = tf.transform.translation
                R = quat_to_matrix([q.x, q.y, q.z, q.w])
                t = np.array([p.x, p.y, p.z])
                parent, child = tf.header.frame_id.lstrip("/"), tf.child_frame_id.lstrip("/")
                edges.setdefault(parent, {})[child] = (R, t)              # child in parent
                edges.setdefault(child, {})[parent] = (R.T, -R.T @ t)     # parent in child
        # breadth-first search from target, composing poses of each reached frame in target
        poses = {target: (np.eye(3), np.zeros(3))}
        frontier = [target]
        while frontier and source not in poses:
            nxt = []
            for f in frontier:
                Rf, tf_ = poses[f]
                for g, (R, t) in edges.get(f, {}).items():
                    if g not in poses:
                        poses[g] = (Rf @ R, Rf @ t + tf_)
                        nxt.append(g)
            frontier = nxt
        if source not in poses:
            raise KeyError(f"no static transform chain between {target} and {source}")
        return poses[source]


def _image_array(m):
    channels = {"rgb8": 3, "bgr8": 3, "mono8": 1}[m.encoding]
    a = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.step)[:, : m.width * channels]
    return a.reshape(m.height, m.width, channels) if channels > 1 else a.reshape(m.height, m.width)


def nearest(t_query, t, x):
    """Samples of (t, x) nearest in time to each query time, and the time offsets."""
    t_query = np.atleast_1d(t_query)
    i = np.clip(np.searchsorted(t, t_query), 1, len(t) - 1)
    i -= (t_query - t[i - 1]) < (t[i] - t_query)
    return x[i], t[i] - t_query


def window_mean(t, x, t0, t1):
    """Mean of x over [t0, t1] and the number of samples."""
    m = (t >= t0) & (t <= t1)
    return x[m].mean(axis=0), int(m.sum())
