#!/usr/bin/env python
"""
01_split_chunks.py — Segment an MCAP recording into training-ready chunks.

This script processes a ROS 2-style `.mcap` file and identifies "usable" data segments
based on timing constraints. It outputs a JSON manifest of valid chunks.

A usable chunk must satisfy:
- Consecutive LiDAR message gaps ≤ `max_gap_ms` (default: 150 ms)
- Consecutive camera message gaps ≤ `max_gap_ms`
- Chunk duration ≥ `min_length_s` (default: 40 seconds)

Time source:
- By default, timestamps are taken from the message header (`header.stamp`).
- Alternatively, you can set `time_source: log` to use MCAP log/publish times.

Configuration is loaded from: configs/split_chunks.yaml

Example config:

  mcap_path: "kitti.mcap"
  out: "out/chunks_demo.json"
  lidar_topic: "/sensor/lidar/front/points"
  cam_topic: "/sensor/camera/left/image_raw/compressed"
  max_gap_ms: 150
  min_length_s: 40
  time_source: "sensor"
  debug: true

Usage:
  python src/01_split_chunks.py

Output:
- A JSON manifest listing valid chunks, saved to the `out` path.
"""

import argparse, json, os, datetime as _dt
from bisect import bisect_left, bisect_right
from typing import List, Tuple, Dict
import numpy as np
from mcap_ros2.reader import read_ros2_messages
from utils import (
    _to_ns,
    get_topic,
    build_segments,
    intersect,
    lens,
    CAM_TOPIC,
    LIDAR_TOPIC,
)


import yaml

with open("configs/split_chunks.yaml") as f:
    cfg = yaml.safe_load(f)
mcap_path = cfg["mcap_path"]
out_path = cfg["out"]
lidar_topic = cfg.get("lidar_topic", LIDAR_TOPIC)
cam_topic = cfg.get("cam_topic", CAM_TOPIC)
max_gap_ms = cfg.get("max_gap_ms", 150)
min_length_s = cfg.get("min_length_s", 40)
time_source = cfg.get("time_source", "sensor")
debug = cfg.get("debug", False)


os.makedirs(os.path.dirname(out_path), exist_ok=True)
max_gap_ns = int(max_gap_ms * 1e6)
min_len_ns = int(min_length_s * 1e9)


def get_time_ns(wrap, mode: str):
    """mode in {'sensor','log'}"""
    if mode == "sensor":
        msg = getattr(wrap, "message", None)
        if msg is not None:
            hdr = getattr(msg, "header", None)
            if hdr is not None and getattr(hdr, "stamp", None) is not None:
                stamp = hdr.stamp
                sec = getattr(stamp, "sec", None)
                nsec = getattr(stamp, "nanosec", None)
                if sec is not None and nsec is not None:
                    return int(sec) * 1_000_000_000 + int(nsec)
    # fallback to log/publish time
    for attr in ("log_time", "publish_time", "logTime", "publishTime", "timestamp"):
        if hasattr(wrap, attr):
            ns = _to_ns(getattr(wrap, attr))
            if ns is not None:
                return ns
    return None


def gap_stats(ts):
    if len(ts) < 2:
        return {"count": len(ts), "median_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    d = np.diff(np.array(ts, dtype=np.int64)) / 1e6
    return {
        "count": len(ts),
        "median_ms": float(np.median(d)),
        "p95_ms": float(np.percentile(d, 95)),
        "max_ms": float(np.max(d)),
    }


def count_in_range(sorted_ts, start_ns, end_ns):
    lo = bisect_left(sorted_ts, start_ns)
    hi = bisect_right(sorted_ts, end_ns)
    return max(0, hi - lo)


def main():

    lidar_ts = []
    cam_ts = []
    for wrap in read_ros2_messages(mcap_path, topics=[lidar_topic, cam_topic]):

        topic = get_topic(wrap)
        tns = get_time_ns(wrap, time_source)
        if tns is None:
            continue
        if topic == lidar_topic:
            lidar_ts.append(tns)
        elif topic == cam_topic:
            cam_ts.append(tns)

    lidar_ts.sort()
    cam_ts.sort()

    if debug:
        ls = gap_stats(lidar_ts)
        cs = gap_stats(cam_ts)
        print(
            f"[{time_source}] Collected: lidar={ls['count']} ts, cam={cs['count']} ts"
        )
        print(
            f"[{time_source}] LiDAR gaps ms: median={ls['median_ms']:.1f}, p95={ls['p95_ms']:.1f}, max={ls['max_ms']:.1f}"
        )
        print(
            f"[{time_source}] Camera gaps ms: median={cs['median_ms']:.1f}, p95={cs['p95_ms']:.1f}, max={cs['max_ms']:.1f}"
        )

    seg_lidar = build_segments(lidar_ts, max_gap_ns)
    seg_cam = build_segments(cam_ts, max_gap_ns)
    common = intersect(seg_lidar, seg_cam)
    chunks = [(s, e) for (s, e) in common if (e - s) >= min_len_ns]

    if debug:
        print(f"[{time_source}] LiDAR segments (s): {lens(seg_lidar)}")
        print(f"[{time_source}] Camera segments (s): {lens(seg_cam)}")
        print(f"[{time_source}] Common segments (s): {lens(common)}")

    manifest = {
        "mcap": os.path.abspath(mcap_path),
        "lidar_topic": lidar_topic,
        "camera_topic": cam_topic,
        "max_gap_ms": max_gap_ms,
        "min_length_s": min_length_s,
        "time_source": time_source,
        "chunks": [],
    }

    for idx, (s, e) in enumerate(chunks):
        manifest["chunks"].append(
            {
                "id": idx,
                "start_ns": int(s),
                "end_ns": int(e),
                "duration_s": round((e - s) / 1e9, 3),
                "lidar_msgs": count_in_range(lidar_ts, s, e),
                "camera_msgs": count_in_range(cam_ts, s, e),
            }
        )

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if manifest["chunks"]:
        print(f"[{time_source}] Found {len(manifest['chunks'])} usable chunk(s):")
        for c in manifest["chunks"]:
            print(
                f"  - chunk_{c['id']:03d}: {c['duration_s']}s (lidar={c['lidar_msgs']}, cam={c['camera_msgs']})"
            )
        print(f"Manifest written to: {out_path}")
    else:
        print(f"[{time_source}] No usable chunks found with current thresholds.")
        print(f"Manifest written to: {out_path}")


if __name__ == "__main__":
    main()
