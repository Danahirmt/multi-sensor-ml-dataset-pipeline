#!/usr/bin/env python
"""
01a_dump_gaps.py — Analyze LiDAR and camera gaps in an MCAP file.

This script inspects a ROS 2-style MCAP file and reports:
- Timestamp gaps between LiDAR and camera messages
- Summary statistics (median, p95, max gaps)
- Top-10 largest gaps per modality
- Segmented intervals where both sensors have continuous data
- Whether any common segment satisfies both max_gap and min_length criteria

The report is saved as a plain text file.

Time source:
- By default, uses `header.stamp` from sensor messages.
- Can be overridden with `--time-source log` to use MCAP log/publish time.

Example usage:

  python src/01a_dump_gaps.py --mcap /workspace/kitti.mcap
  python src/01a_dump_gaps.py --mcap /workspace/kitti.mcap --time-source log
  python src/01a_dump_gaps.py --mcap /workspace/kitti.mcap --max-gap-ms 150.5 --out-report out/gaps_150_5ms.txt

Outputs:
- Summary and segment analysis written to `--out-report` (default: out/gaps_report.txt)
"""

import argparse, datetime as _dt, os, json, numpy as np
from mcap_ros2.reader import read_ros2_messages
from utils import (
    get_topic,
    get_log_time_ns,
    get_ros_message,
    header_stamp_ns,
    build_segments,
    intersect,
    lens,
    CAM_TOPIC,
    LIDAR_TOPIC
)





def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcap", required=True)
    ap.add_argument("--max-gap-ms", type=float, default=150.0)
    ap.add_argument("--min-length-s", type=float, default=40.0)
    ap.add_argument("--time-source", choices=["sensor","log"], default="sensor",
                    help="Prefer sensor header.stamp or MCAP log/publish time")
    ap.add_argument("--out-report", default="out/gaps_report.txt")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_report), exist_ok=True)
    max_gap_ns = int(args.max_gap_ms*1e6)
    min_len_ns = int(args.min_length_s*1e9)

    # 1) collect timestamps
    lidar, cam = [], []
    for w in read_ros2_messages(args.mcap, topics=[LIDAR_TOPIC, CAM_TOPIC]):
        t = None
        if args.time_source == "sensor":
            t = header_stamp_ns(get_ros_message(w))
        if t is None:  # fallback or mode=log
            t = get_log_time_ns(w)
        if t is None: 
            continue
        topic = get_topic(w)
        if topic == LIDAR_TOPIC: lidar.append(t)
        elif topic == CAM_TOPIC: cam.append(t)

    lidar.sort(); cam.sort()
    if not lidar or not cam:
        print("Missing lidar/camera timestamps"); 
        return

    def gaps(ts): 
        if len(ts) < 2: return np.array([], dtype=np.int64)
        return np.diff(np.array(ts, dtype=np.int64))

    gL = gaps(lidar); gC = gaps(cam)
    segL = build_segments(lidar, max_gap_ns)
    segC = build_segments(cam,   max_gap_ns)
    common = intersect(segL, segC)
    common_len = sorted([(e-s,(s,e)) for (s,e) in common], reverse=True)

    with open(args.out_report, "w") as f:
        def wline(x=""): f.write(str(x)+"\n")
        wline(f"File: {args.mcap}")
        wline(f"Time source: {args.time_source}")
        wline(f"Max gap: {args.max_gap_ms} ms | Min length: {args.min_length_s} s")
        wline()
        for name, g in [("LiDAR", gL), ("Camera", gC)]:
            if g.size:
                ms = g/1e6
                wline(f"[{name}] gaps ms: count={ms.size}, median={np.median(ms):.2f}, p95={np.percentile(ms,95):.2f}, max={ms.max():.2f}")
                top_idx = np.argsort(ms)[-10:][::-1]
                wline(f"  Top 10 gaps (ms): {ms[top_idx].round(2).tolist()}")
            else:
                wline(f"[{name}] gaps ms: count=0")
        wline()
        wline(f"LiDAR segments (s): {lens(segL)}")
        wline(f"Camera segments (s): {lens(segC)}")
        wline(f"Common segments (s): {lens(common)}")
        wline()
        wline("Top-5 longest COMMON segments (start_ns, end_ns, length_s):")
        for k,(L,(s,e)) in enumerate(common_len[:5]):
            wline(f"  {k+1}. {s} .. {e}  ({L/1e9:.3f}s)")
        ok = any((e-s) >= min_len_ns for (s,e) in common)
        wline()
        wline(f"Exists >= {args.min_length_s}s common chunk?  {'YES' if ok else 'NO'}")

    print(f"Report written: {args.out_report}")
    print(open(args.out_report).read())

if __name__ == "__main__":
    main()
