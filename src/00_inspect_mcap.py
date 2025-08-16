#!/usr/bin/env python
"""
00_inspect_mcap.py — MCAP inspection utility for ROS 2-style datasets.

This script provides a quick overview of a `.mcap` recording, including:

- Listing all available topics with message counts.
- Displaying the first and last timestamps per topic, with total duration in seconds.
- Optionally saving a single sample:
  - One camera image as JPEG.
  - One LiDAR point cloud as a binary `.bin` file (x, y, z, intensity).

Requires:
- `pointcloud2` Python package for decoding ROS 2 point cloud messages.
  See: https://github.com/mrkbac/pointcloud2

Typical usage:

    python src/00_inspect_mcap.py --mcap /workspace/kitti.mcap
    python src/00_inspect_mcap.py --mcap /workspace/kitti.mcap --save-sample

Arguments:
    --mcap           Path to the input `.mcap` file.
    --save-sample    If set, saves a sample image and point cloud to `out_smoke/`.
"""

import argparse
import os
from collections import defaultdict
import numpy as np
import cv2
from mcap_ros2.reader import read_ros2_messages

from utils import (
    get_topic,
    get_log_time_ns,
    get_ros_message,
    read_pointcloud4d,
    CAM_TOPIC,
    LIDAR_TOPIC
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcap", required=True, help="Path to .mcap file")
    ap.add_argument("--save-sample", action="store_true", help="Save one image and one LiDAR sample to out_smoke/")
    args = ap.parse_args()

    if not os.path.exists(args.mcap):
        raise FileNotFoundError(f"MCAP not found: {args.mcap}")

    counts = defaultdict(int)
    first_ns = {}
    last_ns = {}

    # First pass: collect topic stats
    for wrap in read_ros2_messages(args.mcap):
        topic = get_topic(wrap)
        tns = get_log_time_ns(wrap)
        counts[topic] += 1
        if tns is not None:
            if topic not in first_ns or tns < first_ns[topic]:
                first_ns[topic] = tns
            if topic not in last_ns or tns > last_ns[topic]:
                last_ns[topic] = tns

    if not counts:
        print("No messages found. Check file format/path.")
        return

    print("== Topics ==")
    for k in sorted(counts.keys()):
        print(f"- {k}")

    print("\n== Message counts ==")
    for k in sorted(counts.keys()):
        print(f"{k}: {counts[k]}")

    print("\n== Time ranges (ns) ==")
    for k in sorted(first_ns.keys()):
        dur = (last_ns[k] - first_ns[k]) / 1e9
        print(f"{k}: {first_ns[k]} .. {last_ns[k]}  (~{dur:.2f}s)")

    if args.save_sample:
        os.makedirs("out_smoke", exist_ok=True)
        saved_img = False
        saved_pcl = False

        for wrap in read_ros2_messages(args.mcap, topics=[CAM_TOPIC, LIDAR_TOPIC]):
            topic = get_topic(wrap)
            msg = get_ros_message(wrap)


            if topic == CAM_TOPIC and not saved_img:
                buf = np.frombuffer(msg.data, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is not None:
                    cv2.imwrite("out_smoke/frame0.jpg", img)
                    print("Saved image: out_smoke/frame0.jpg")
                    saved_img = True

            if topic == LIDAR_TOPIC and not saved_pcl:
                xyz_i = read_pointcloud4d(msg)
                xyz_i.astype(np.float32).tofile("out_smoke/cloud0.bin")
                print(f"Saved point cloud: out_smoke/cloud0.bin  shape={xyz_i.shape}")
                saved_pcl = True

            if saved_img and saved_pcl:
                break

        if not saved_img:
            print("Warning: no camera sample saved.")
        if not saved_pcl:
            print("Warning: no LiDAR sample saved.")

if __name__ == "__main__":
    main()
