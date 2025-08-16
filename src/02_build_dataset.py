#!/usr/bin/env python
"""
02_build_dataset.py — Build a nuScenes-lite-style dataset from an MCAP recording and chunk manifest.

This script reads a ROS 2-style `.mcap` file and a chunk manifest (generated via 01_split_chunks.py),
and outputs a lightweight dataset with camera images, LiDAR point clouds, poses, and calibration data.

Expected input topics:
- /sensor/camera/left/image_raw/compressed   → JPEG-compressed camera frames
- /sensor/lidar/front/points                 → LiDAR point clouds (PointCloud2)
- /sensor/camera/left/camera_info            → Camera intrinsics (sensor_msgs/CameraInfo)
- /tf_static                                 → Static extrinsics between frames
- /tf                                        → Dynamic transforms (optional)

Output directory structure (per scene):
  scene_<id>/
    images/left/<timestamp>.jpg             # JPEG image
    lidar/<timestamp>.bin                   # LiDAR point cloud in Nx4 float32: x,y,z,intensity
    poses/<timestamp>.json                  # (optional) dynamic TF pose if available
    calib/
      camera_left.json                      # intrinsics: K, D, width, height
      T_<cam>__<lidar>.json                 # static extrinsics (4x4 transform matrix)
    index/samples.json                      # metadata for all paired samples

Optional:
- If `--write-parquet` is enabled (and pyarrow is installed), a global `manifest.parquet` file
  is created with flat scene/sample metadata, suitable for fast indexing.

Usage:
  python src/02_build_dataset.py

Note:
- All configuration (paths, tolerances, frames) must be defined in `configs/build_dataset.yaml`
"""


import argparse, os, json
from collections import defaultdict
import numpy as np
import cv2

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    _HAS_PA = True
except ImportError:
    _HAS_PA = False

from mcap_ros2.reader import read_ros2_messages

from utils import (
    get_topic,
    get_log_time_ns,
    get_ros_message,
    header_stamp_ns,
    read_pointcloud4d,
    nearest,
    CAM_TOPIC,
    LIDAR_TOPIC,
    TF_TOPIC,
    CAMINFO_TOPIC,
    TF_STATIC_TOPIC,
    tf_msg_to_list,
    invert,
    solve_chain,
    camera_info_to_json,
)

import yaml

with open("configs/build_dataset.yaml") as f:
    cfg = yaml.safe_load(f)

mcap_path = cfg["mcap_path"]
manifest_path = cfg["manifest"]
out_dir = cfg["out_dir"]
pair_tolerance_ms = cfg.get("pair_tolerance_ms", 50)
pair_offset_ms = cfg.get("pair_offset_ms", 0)
cam_frame = cfg.get("cam_frame", "camera_frame")
lidar_frame = cfg.get("lidar_frame", "lidar_frame")
write_parquet = cfg.get("write_parquet", True)
debug = cfg.get("debug", False)
jpeg_quality = cfg.get("jpeg_quality", 95)

pair_tol_ns = int(pair_tolerance_ms * 1e6)
pair_off_ns = int(pair_offset_ms * 1e6)


def _sanitize_fname(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in s)


def main():
    os.makedirs(out_dir, exist_ok=True)
    manifest = json.load(open(manifest_path, "r"))
    chunks = manifest.get("chunks", [])
    if not chunks:
        print("Manifest has no chunks. Nothing to build.")
        return

    cam_topic = manifest.get("camera_topic", CAM_TOPIC)
    lidar_topic = manifest.get("lidar_topic", LIDAR_TOPIC)

    # Scan tf_static and camera_info
    static_graph = defaultdict(list)
    caminfo_msg = None
    for wrap in read_ros2_messages(mcap_path, topics=[TF_STATIC_TOPIC, CAMINFO_TOPIC]):
        topic = get_topic(wrap)
        msg = get_ros_message(wrap)
        if not msg:
            continue
        if topic == TF_STATIC_TOPIC:
            for tfs in getattr(msg, "transforms", []) or []:
                rec = tf_msg_to_list(tfs)
                if rec:
                    parent, child, T = rec
                    static_graph[parent].append((child, T))
                    static_graph[child].append((parent, invert(T)))
        elif topic == CAMINFO_TOPIC and caminfo_msg is None:
            caminfo_msg = msg

    if debug:
        print("Static graph:")
        for k, v in static_graph.items():
            for child, _ in v:
                print(f"  {k} -> {child}")

    T_cam_lidar = solve_chain(static_graph, cam_frame, lidar_frame)
    if T_cam_lidar is None:
        T_lidar_cam = solve_chain(static_graph, lidar_frame, cam_frame)
        if T_lidar_cam is not None:
            T_cam_lidar = invert(T_lidar_cam)

    parquet_rows = []

    # Iterate over scenes
    for c in chunks:
        cid = int(c["id"])
        start_ns = int(c["start_ns"])
        end_ns = int(c["end_ns"])
        scene_dir = os.path.join(out_dir, f"scene_{cid:03d}")
        paths = {
            "images": os.path.join(scene_dir, "images", "left"),
            "lidar": os.path.join(scene_dir, "lidar"),
            "poses": os.path.join(scene_dir, "poses"),
            "calib": os.path.join(scene_dir, "calib"),
            "index": os.path.join(scene_dir, "index"),
        }
        for p in paths.values():
            os.makedirs(p, exist_ok=True)

        calib_files = {}

        if caminfo_msg:
            cam_json = camera_info_to_json(caminfo_msg)
            with open(os.path.join(paths["calib"], "camera_left.json"), "w") as f:
                json.dump(cam_json, f, indent=2)
            calib_files["cam_intrinsics"] = "calib/camera_left.json"

        if T_cam_lidar is not None:
            fname = (
                f"T_{_sanitize_fname(cam_frame)}__{_sanitize_fname(lidar_frame)}.json"
            )
            with open(os.path.join(paths["calib"], fname), "w") as f:
                json.dump(
                    {
                        "frame_from": cam_frame,
                        "frame_to": lidar_frame,
                        "T": T_cam_lidar.tolist(),
                    },
                    f,
                    indent=2,
                )
            calib_files["static_extrinsics"] = os.path.join("calib", fname)
        elif debug:
            print("[warn] No static extrinsics between camera and lidar")

        cam_ts, cam_paths = [], []
        lidar_ts, lidar_paths = [], []
        tf_buffer = []

        for wrap in read_ros2_messages(
            mcap_path, topics=[cam_topic, lidar_topic, TF_TOPIC]
        ):
            topic = get_topic(wrap)
            msg = get_ros_message(wrap)
            if msg is None:
                continue
            tns = header_stamp_ns(msg) or get_log_time_ns(wrap)
            if not (start_ns <= tns <= end_ns):
                continue

            if topic == cam_topic:
                img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    fname = f"{tns}.jpg"
                    cv2.imwrite(
                        os.path.join(paths["images"], fname),
                        img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
                    )
                    cam_ts.append(tns)
                    cam_paths.append(os.path.join("images", "left", fname))

            elif topic == lidar_topic:
                pc = read_pointcloud4d(msg)
                fname = f"{tns}.bin"
                pc.astype(np.float32).tofile(os.path.join(paths["lidar"], fname))
                lidar_ts.append(tns)
                lidar_paths.append(os.path.join("lidar", fname))

            elif topic == TF_TOPIC:
                for tfs in getattr(msg, "transforms", []):
                    rec = tf_msg_to_list(tfs)
                    if rec:
                        parent, child, T = rec
                        ts_tf = header_stamp_ns(tfs) or tns
                        tf_buffer.append(
                            (int(ts_tf), parent, child, T.astype(np.float32))
                        )

        cam_ts_sorted = np.array(sorted(cam_ts), dtype=np.int64)
        lidar_ts_sorted = np.array(sorted(lidar_ts), dtype=np.int64)
        lidar_ts_adj = lidar_ts_sorted - pair_off_ns
        cam_map = dict(zip(cam_ts, cam_paths))
        lidar_map = dict(zip(lidar_ts, lidar_paths))
        tf_buffer.sort()
        tf_ts = np.array([t[0] for t in tf_buffer], dtype=np.int64)

        frames = []
        for t in cam_ts_sorted:
            j = nearest(lidar_ts_adj, t, pair_tol_ns)
            if j == -1:
                continue
            t_lidar = int(lidar_ts_sorted[j])

            pose_path = ""
            k = nearest(tf_ts, t, pair_tol_ns)
            if k != -1:
                ts_tf, parent, child, T = tf_buffer[k]
                pose_fname = f"{t}.json"
                pose_path = os.path.join("poses", pose_fname)
                with open(os.path.join(paths["poses"], pose_fname), "w") as f:
                    json.dump(
                        {
                            "timestamp_ns": int(t),
                            "frame_from": parent,
                            "frame_to": child,
                            "T": T.tolist(),
                        },
                        f,
                        indent=2,
                    )

            frame = {
                "sample_id": str(t),
                "timestamp_ns": int(t),
                "image_left": cam_map[t],
                "lidar_front": lidar_map[t_lidar],
                "pose": pose_path,
                "calib": {
                    "cam_intrinsics": calib_files.get("cam_intrinsics", ""),
                    "static_extrinsics": calib_files.get("static_extrinsics", ""),
                },
            }
            frames.append(frame)

            if write_parquet:
                parquet_rows.append(
                    {"scene_id": f"scene_{cid:03d}", **frame, **frame["calib"]}
                )

        samples = {
            "scene_id": f"scene_{cid:03d}",
            "start_ns": start_ns,
            "end_ns": end_ns,
            "num_frames": len(frames),
            "frames": frames,
        }
        with open(os.path.join(paths["index"], "samples.json"), "w") as f:
            json.dump(samples, f, indent=2)

        print(f"scene_{cid:03d}: {len(frames)} paired frames.")

    if write_parquet and _HAS_PA:
        table = pa.Table.from_pylist(parquet_rows)
        pq.write_table(table, os.path.join(out_dir, "manifest.parquet"))
        print(
            f"manifest.parquet written to {os.path.join(out_dir, 'manifest.parquet')}"
        )
    elif write_parquet and not _HAS_PA:
        print("WARNING: pyarrow not available for writing manifest.parquet")


if __name__ == "__main__":
    main()
