"""
utils.py

A collection of utility functions and constants for handling multi-sensor ROS data
used throughout the nuScenes-lite dataset generation and loading pipeline.

Core responsibilities:
- Timestamp and topic extraction from MCAP-wrapped ROS messages
- Parsing and transforming image, LiDAR, camera, and transform data
- Frame synchronization (nearest timestamps, segmenting)
- Pose and transformation matrix operations (including quaternion handling)
- Loading and validating files (images, point clouds, calibration, poses)

Constants:
- CAM_TOPIC: camera topic (compressed image)
- LIDAR_TOPIC: lidar topic (PointCloud2)
- TF_TOPIC: dynamic transforms
- CAMINFO_TOPIC: camera calibration
- TF_STATIC_TOPIC: static transforms

Key functions:

Timestamps and message access:
- `get_topic(wrap)`: Return topic string from wrapped message.
- `get_log_time_ns(wrap)`: Extract publish/log time in nanoseconds.
- `get_ros_message(wrap)`: Extract the ROS message from wrapper.
- `header_stamp_ns(msg)`: Extract timestamp from message header.

Sensor data loaders:
- `read_pointcloud4d(msg)`: Convert PointCloud2 message to Nx4 numpy array (x,y,z,intensity).
- `load_image(path)`: Load image from disk using OpenCV.
- `load_pointcloud(path)`: Load point cloud binary from disk, validate shape and values.
- `load_pose(path)`: Load 3x3 rotation and 3x1 translation from JSON with `R` and `t`.
- `load_pose_T_json(path)`: Load full 4x4 pose matrix `T` from JSON.

Transform utilities:
- `quat_xyzw_to_mat44(...)`: Convert quaternion + translation to 4x4 matrix.
- `tf_msg_to_list(...)`: Extract parent/child frame and transform matrix from TF message.
- `invert(T)`: Invert 4x4 transformation matrix.
- `compose(A, B)`: Compose two 4x4 transforms.
- `solve_chain(graph, src, dst)`: Find transform from src to dst by walking the static TF graph.

Data segmentation and pairing:
- `build_segments(ts, max_gap_ns)`: Segment sorted timestamps by temporal gaps.
- `intersect(a, b)`: Compute intersection of time segments.
- `lens(segs)`: Return duration (in seconds) of each segment.
- `nearest(ts_sorted, t, tol_ns)`: Find index of nearest timestamp within tolerance.

Camera calibration:
- `camera_info_to_json(msg)`: Convert ROS CameraInfo message to dict with intrinsics.

"""


import datetime as _dt
import numpy as np
import json
import cv2
from pointcloud2 import read_points
from bisect import bisect_left


CAM_TOPIC = "/sensor/camera/left/image_raw/compressed"
LIDAR_TOPIC = "/sensor/lidar/front/points"
TF_TOPIC = "/tf"
CAMINFO_TOPIC = "/sensor/camera/left/camera_info"
TF_STATIC_TOPIC = "/tf_static"


def _to_ns(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return (
            int(val * 1e9)
            if isinstance(val, float)
            else int(val if val > 1e12 else val * 1e9)
        )
    if isinstance(val, _dt.datetime):
        if val.tzinfo is None:
            val = val.replace(tzinfo=_dt.timezone.utc)
        return int(val.timestamp() * 1e9)
    return None


def get_topic(wrap):
    if hasattr(wrap, "topic"):
        return wrap.topic
    ch = getattr(wrap, "channel", None)
    if ch is not None:
        if hasattr(ch, "topic"):
            return ch.topic
        if isinstance(ch, dict) and "topic" in ch:
            return ch["topic"]
    return "<unknown>"


def get_log_time_ns(wrap):
    for attr in ("log_time", "publish_time", "logTime", "publishTime", "timestamp"):
        if hasattr(wrap, attr):
            ns = _to_ns(getattr(wrap, attr))
            if ns is not None:
                return ns
    return None


def get_ros_message(wrap):
    for attr in ("ros_msg", "message", "msg"):
        if hasattr(wrap, attr):
            return getattr(wrap, attr)
    return None


def header_stamp_ns(msg):
    if msg is None:
        return None
    hdr = getattr(msg, "header", None)
    if hdr is None:
        return None
    st = getattr(hdr, "stamp", None)
    if st is None:
        return None
    sec = getattr(st, "sec", None)
    nsec = getattr(st, "nanosec", None)
    if sec is None or nsec is None:
        return None
    return int(sec) * 1_000_000_000 + int(nsec)


def read_pointcloud4d(msg):
    """Return Nx4 array (x,y,z,intensity) from PointCloud2 ROS msg."""
    arr = read_points(msg)
    x = arr["x"].astype(np.float32)
    y = arr["y"].astype(np.float32)
    z = arr["z"].astype(np.float32)
    i = (
        arr["intensity"].astype(np.float32)
        if "intensity" in arr.dtype.names
        else np.zeros_like(x, dtype=np.float32)
    )
    return np.stack([x, y, z, i], axis=1)


def build_segments(ts, max_gap_ns):
    if not ts:
        return []
    segs = []
    s = ts[0]
    last = ts[0]
    for t in ts[1:]:
        if t - last > max_gap_ns:
            segs.append((s, last))
            s = t
        last = t
    segs.append((s, last))
    return segs


def intersect(a, b):
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if s < e:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def lens(segs):
    return [round((e - s) / 1e9, 3) for (s, e) in segs]


def nearest(ts_sorted, t, tol_ns):
    if len(ts_sorted) == 0:
        return -1
    i = bisect_left(ts_sorted, t)
    best_idx, best_diff = None, None
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(ts_sorted):
            diff = abs(ts_sorted[j] - t)
            if best_diff is None or diff < best_diff:
                best_diff, best_idx = diff, j
    return best_idx if best_diff is not None and best_diff <= tol_ns else -1


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def load_pointcloud(path):
    pc = np.fromfile(path, dtype=np.float32)
    if pc.size % 4 != 0:
        raise ValueError(f"Pointcloud not multiple of 4 floats: {path}")
    pc = pc.reshape(-1, 4)
    if not np.isfinite(pc).all():
        raise ValueError(f"Pointcloud has non-finite values: {path}")
    return pc


def load_pose(path):
    d = json.load(open(path, "r"))
    R = np.array(d["R"], dtype=np.float32)
    t = np.array(d["t"], dtype=np.float32)
    if R.shape != (3, 3) or t.shape != (3,):
        raise ValueError(f"Pose shape invalid: {path}")
    return R, t


def quat_xyzw_to_mat44(x, y, z, w, tx, ty, tz):
    n = x * x + y * y + z * z + w * w
    if n == 0.0:
        R = np.eye(3, dtype=np.float32)
    else:
        s = 2.0 / n
        xx, yy, zz = x * x * s, y * y * s, z * z * s
        xy, xz, yz = x * y * s, x * z * s, y * z * s
        wx, wy, wz = w * x * s, w * y * s, w * z * s
        R = np.array(
            [
                [1 - (yy + zz), xy - wz, xz + wy],
                [xy + wz, 1 - (xx + zz), yz - wx],
                [xz - wy, yz + wx, 1 - (xx + yy)],
            ],
            dtype=np.float32,
        )
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return T


def tf_msg_to_list(transform_stamped):
    hdr = getattr(transform_stamped, "header", None)
    parent_frame_id = getattr(hdr, "frame_id", "") if hdr else ""
    child_frame_id = getattr(transform_stamped, "child_frame_id", "")
    tr = getattr(transform_stamped, "transform", None)
    if tr is None:
        return None
    t = getattr(tr, "translation", None)
    q = getattr(tr, "rotation", None)
    if t is None or q is None:
        return None
    T = quat_xyzw_to_mat44(
        float(getattr(q, "x", 0.0)),
        float(getattr(q, "y", 0.0)),
        float(getattr(q, "z", 0.0)),
        float(getattr(q, "w", 1.0)),
        float(getattr(t, "x", 0.0)),
        float(getattr(t, "y", 0.0)),
        float(getattr(t, "z", 0.0)),
    )
    return (parent_frame_id.strip(), child_frame_id.strip(), T)


def invert(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float32)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def compose(A, B):
    return (A @ B).astype(np.float32)


def solve_chain(static_graph, src, dst):
    """Find T_src_dst by BFS over /tf_static graph (bidirectional)."""
    from collections import deque

    q = deque([(src, np.eye(4, dtype=np.float32))])
    seen = {src}
    while q:
        cur, T_acc = q.popleft()
        if cur == dst:
            return T_acc
        for (nbr, T_cur_nbr) in static_graph.get(cur, []):
            if nbr in seen:
                continue
            seen.add(nbr)
            q.append((nbr, compose(T_acc, T_cur_nbr)))
    return None


def camera_info_to_json(msg):
    d = {
        "width": int(getattr(msg, "width", 0) or 0),
        "height": int(getattr(msg, "height", 0) or 0),
        "K": list(map(float, getattr(msg, "k", getattr(msg, "K", [0] * 9)))),
        "D": list(map(float, getattr(msg, "d", getattr(msg, "D", [])))),
        "distortion_model": str(getattr(msg, "distortion_model", "")),
    }
    if len(d["K"]) != 9:  # normalize to 9
        arr = getattr(msg, "k", None)
        if arr is not None and hasattr(arr, "__len__"):
            d["K"] = [float(x) for x in arr][:9]
    return d


def load_pose_T_json(path):
    """Load a 4x4 pose matrix stored under key 'T' in a JSON file."""
    d = json.load(open(path, "r"))
    T = np.array(d["T"], dtype=np.float32)
    if T.shape != (4, 4):
        raise ValueError(f"Pose 'T' must be 4x4: {path}")
    return T
