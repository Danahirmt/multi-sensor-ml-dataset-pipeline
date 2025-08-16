#!/usr/bin/env python
"""
01b_plot_gaps.py — Visualize sensor message gaps and timeline segments.

This script creates diagnostic plots to analyze the temporal structure of an MCAP recording:
- Inter-message gap plots for LiDAR and camera streams
- A timeline plot showing valid segments per sensor and their overlap

It supports both sensor timestamps (`header.stamp`) and log time.

Generated outputs (saved to --out-dir):
  - lidar_gaps.png:     Gap plot for LiDAR messages
  - camera_gaps.png:    Gap plot for camera messages
  - segments_timeline.png: Timeline of valid segments (LiDAR / Camera / Common)

Usage examples:

  python src/01b_plot_gaps.py --mcap /workspace/kitti.mcap
  python src/01b_plot_gaps.py --mcap /workspace/kitti.mcap --time-source log --out-dir out/figs
  python src/01b_plot_gaps.py --mcap /workspace/kitti.mcap --max-gap-ms 100 --out-dir out/figs

Notes:
- Set `--time-source` to `log` to use MCAP log/publish time instead of sensor timestamps.
- Figures are saved in PNG format in the specified output directory.
"""

import argparse, os, numpy as np, datetime as _dt
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
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




def plot_gaps(gaps_ms, thr_ms, title, outpath):
    if gaps_ms.size == 0:
        plt.figure(figsize=(10,3)); plt.title(title); plt.text(0.5,0.5,"No gaps (0/1 sample)", ha="center"); plt.axis("off")
        plt.savefig(outpath, bbox_inches="tight", dpi=150); plt.close(); return
    x = np.arange(1, gaps_ms.size+1)
    plt.figure(figsize=(12,4))
    plt.plot(x, gaps_ms, linewidth=0.8)
    plt.axhline(thr_ms, linestyle="--")
    plt.xlabel("Frame index")
    plt.ylabel("Gap [ms]")
    plt.title(title + f" | median={np.median(gaps_ms):.2f} ms, p95={np.percentile(gaps_ms,95):.2f} ms, max={gaps_ms.max():.2f} ms")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_timeline(lidar_segs, cam_segs, common_segs, outpath):
    # convert to seconds relative to global min start
    all_segs = lidar_segs + cam_segs + common_segs if (lidar_segs or cam_segs or common_segs) else []
    if not all_segs:
        plt.figure(figsize=(12,2)); plt.title("No segments"); plt.axis("off"); plt.savefig(outpath, dpi=150); plt.close(); return
    t0 = min(s for s,_ in all_segs)
    rows = [
        ("LiDAR", 30, lidar_segs),
        ("Camera", 20, cam_segs),
        ("Common", 10, common_segs),
    ]
    plt.figure(figsize=(12,3))
    ax = plt.gca()
    for label, y, segs in rows:
        bars = [((s - t0)/1e9, (e - s)/1e9) for (s,e) in segs]
        if bars:
            ax.broken_barh(bars, (y, 8))
        ax.text(-0.5, y+4, label, va="center")
    ax.set_xlabel("Time [s] (relative)")
    ax.set_ylim(0, 45)
    ax.set_yticks([])
    ax.set_title("Segments timeline (LiDAR / Camera / Common)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcap", required=True)
    ap.add_argument("--max-gap-ms", type=float, default=150.0)
    ap.add_argument("--min-length-s", type=float, default=40.0)  # not used in plotting, but useful
    ap.add_argument("--time-source", choices=["sensor","log"], default="sensor")
    ap.add_argument("--out-dir", default="out/figs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    max_gap_ns = int(args.max_gap_ms*1e6)

    # Collect timestamps
    lidar, cam = [], []
    for w in read_ros2_messages(args.mcap, topics=[LIDAR_TOPIC, CAM_TOPIC]):
        t = ( header_stamp_ns(get_ros_message(w))  if args.time_source == "sensor" else get_log_time_ns(w))
        if t is None:
            continue
        topic = get_topic(w)
        if topic == LIDAR_TOPIC: lidar.append(t)
        elif topic == CAM_TOPIC: cam.append(t)

    lidar.sort(); cam.sort()
    if not lidar or not cam:
        raise RuntimeError("Missing lidar/camera timestamps")

    # Gaps (ms)
    gL = np.diff(np.array(lidar, dtype=np.int64))/1e6 if len(lidar)>1 else np.array([])
    gC = np.diff(np.array(cam, dtype=np.int64))/1e6 if len(cam)>1 else np.array([])

    # Segments & common
    segL = build_segments(lidar, max_gap_ns)
    segC = build_segments(cam,   max_gap_ns)
    common = intersect(segL, segC)

    # Plots
    plot_gaps(gL, args.max_gap_ms, f"LiDAR gaps ({args.time_source} time)", os.path.join(args.out_dir, "lidar_gaps.png"))
    plot_gaps(gC, args.max_gap_ms, f"Camera gaps ({args.time_source} time)", os.path.join(args.out_dir, "camera_gaps.png"))
    plot_timeline(segL, segC, common, os.path.join(args.out_dir, "segments_timeline.png"))

    # Console summary
    print(f"[{args.time_source}] LiDAR segments (s): {lens(segL)}")
    print(f"[{args.time_source}] Camera segments (s): {lens(segC)}")
    print(f"[{args.time_source}] Common segments (s): {lens(common)}")
    print(f"Saved figures to: {args.out_dir}")

if __name__ == "__main__":
    main()
