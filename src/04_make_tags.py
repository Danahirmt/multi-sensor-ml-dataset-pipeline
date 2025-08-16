#!/usr/bin/env python
"""
04_make_tags.py — Metadata and tagging tool for nuScenes-lite dataset scenes.

Generates structured metadata and simple tag summaries for each scene built by `02_build_dataset.py`.

Per-scene outputs:
- `tags.json`: metadata including timing, resolution, pose/extrinsics coverage, LiDAR density, and derived tags.

Optional global outputs:
- `tags.parquet` (if `--write-parquet` and `pyarrow` installed)
- `tags.csv` (if `--write-csv`)

What it extracts:
- General:
    - Scene ID, number of frames, start/end timestamps, duration (s)
    - Frame rate estimate (`paired_fps`), median and p95 inter-frame interval (ms)
- Camera:
    - Resolution, distortion model, intrinsics file path
- Extrinsics:
    - Presence of static transform file `T_*.json` and its relative path
- Pose:
    - Ratio of frames with pose information
    - Sampled (frame_from, frame_to) pairs from pose files (up to --max-poses-to-sample)
- LiDAR:
    - Approximate point count statistics (avg/min/max), inferred from `.bin` sizes
- Tags:
    - Simple categorical tags for filtering or analysis:
      e.g., "pose", "no-extrinsics", "dense-lidar", "res=1241x376", "cam_model=plumb_bob"

Usage:
  python src/04_make_tags.py --root out/dataset --write-parquet --write-csv --max-poses-to-sample 100
"""


import os
import json
import argparse
import statistics
from typing import Dict, List, Tuple

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_PA = True
except ImportError:
    _HAS_PA = False

def _exists(p: str) -> bool:
    return os.path.exists(p)

def _read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def _find_any_extrinsics_file(calib_dir: str) -> str:
    if not _exists(calib_dir):
        return ""
    for fn in os.listdir(calib_dir):
        if fn.startswith("T_") and fn.endswith(".json"):
            return os.path.join(calib_dir, fn)
    return ""

def _count_points_fast(bin_path: str) -> int:
    """Each point is 4 float32 = 16 bytes. Count = filesize // 16."""
    try:
        sz = os.path.getsize(bin_path)
        return sz // 16
    except OSError:
        return 0

def _compute_dt_stats_ns(timestamps_ns: List[int]) -> Tuple[float, float]:
    """Return (median_dt_ms, p95_dt_ms) from consecutive timestamp diffs."""
    if len(timestamps_ns) < 2:
        return 0.0, 0.0
    diffs_ms = []
    for i in range(1, len(timestamps_ns)):
        diffs_ms.append((timestamps_ns[i] - timestamps_ns[i-1]) / 1e6)
    diffs_ms.sort()
    med = statistics.median(diffs_ms)
    p95 = diffs_ms[int(0.95 * (len(diffs_ms)-1))] if diffs_ms else 0.0
    return float(med), float(p95)

def _collect_pose_pairs(scene_dir: str, frames: List[dict], max_poses_to_sample: int = 100) -> List[Tuple[str, str]]:
    pairs = []
    n = 0
    for fr in frames:
        pp = fr.get("pose", "")
        if not pp:
            continue
        path = os.path.join(scene_dir, pp)
        if not _exists(path):
            continue
        try:
            d = _read_json(path)
            pairs.append((d.get("frame_from",""), d.get("frame_to","")))
            n += 1
            if n >= max_poses_to_sample:
                break
        except Exception:
            continue
    # unique, stable order
    seen = set()
    uniq = []
    for p in pairs:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq

def _tags_from_stats(has_pose: bool, has_extr: bool, w: int, h: int, avg_pts: float, fps: float) -> List[str]:
    tags = []
    tags.append("pose" if has_pose else "no-pose")
    tags.append("extrinsics" if has_extr else "no-extrinsics")
    if w > 0 and h > 0:
        tags.append(f"res={w}x{h}")
    if fps > 0:
        tags.append(f"fps~{round(fps,1)}")

    # simple lidar density bucket
    if avg_pts >= 120_000:
        tags.append("dense-lidar")
    elif avg_pts >= 60_000:
        tags.append("mid-lidar")
    else:
        tags.append("sparse-lidar")
    return tags

def process_scene(scene_dir: str, max_poses_to_sample: int = 100) -> Tuple[dict, dict]:
    """Return (tags_json, flat_row_for_table)."""
    scene_id = os.path.basename(scene_dir)
    idx_path = os.path.join(scene_dir, "index", "samples.json")
    calib_dir = os.path.join(scene_dir, "calib")
    if not _exists(idx_path):
        raise FileNotFoundError(f"Missing {idx_path}")

    data = _read_json(idx_path)
    frames = data.get("frames", [])
    start_ns = int(data.get("start_ns", 0))
    end_ns   = int(data.get("end_ns", 0))
    duration_s = (end_ns - start_ns) / 1e9 if end_ns > start_ns else 0.0
    num_frames = len(frames)

    # camera intrinsics
    cam_json_path = os.path.join(calib_dir, "camera_left.json")
    width = height = 0
    cam_model = ""
    if _exists(cam_json_path):
        camj = _read_json(cam_json_path)
        width = int(camj.get("width", 0))
        height = int(camj.get("height", 0))
        cam_model = camj.get("distortion_model", "")

    # extrinsics static
    extr_path_abs = _find_any_extrinsics_file(calib_dir)
    extr_file_rel = os.path.relpath(extr_path_abs, scene_dir) if extr_path_abs else ""
    has_extr = bool(extr_file_rel)

    # timestamps (paired frames)
    ts = [int(fr.get("timestamp_ns", -1)) for fr in frames if "timestamp_ns" in fr]
    ts = sorted([t for t in ts if t >= 0])
    median_dt_ms, p95_dt_ms = _compute_dt_stats_ns(ts)
    fps_pair = (num_frames / duration_s) if duration_s > 0 and num_frames > 0 else 0.0

    # lidar points stats (fast count)
    pts_counts = []
    for fr in frames:
        rel = fr.get("lidar_front", "")
        if not rel:
            continue
        lp = os.path.join(scene_dir, rel)
        if _exists(lp):
            pts_counts.append(_count_points_fast(lp))
    avg_pts = float(sum(pts_counts) / len(pts_counts)) if pts_counts else 0.0
    min_pts = int(min(pts_counts)) if pts_counts else 0
    max_pts = int(max(pts_counts)) if pts_counts else 0

    # poses coverage
    frames_with_pose = sum(1 for fr in frames if fr.get("pose"))
    has_pose = frames_with_pose > 0
    pose_coverage = frames_with_pose / num_frames if num_frames > 0 else 0.0
    pose_pairs = _collect_pose_pairs(scene_dir, frames, max_poses_to_sample=max_poses_to_sample)

    # tag strings
    tag_list = _tags_from_stats(has_pose, has_extr, width, height, avg_pts, fps_pair)
    if cam_model:
        tag_list.append(f"cam_model={cam_model}")

    tags_json = {
        "scene_id": scene_id,
        "num_frames": num_frames,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_s": round(duration_s, 3),
        "paired_fps": round(fps_pair, 3),
        "median_dt_ms": round(median_dt_ms, 3),
        "p95_dt_ms": round(p95_dt_ms, 3),
        "camera": {
            "width": width,
            "height": height,
            "distortion_model": cam_model,
            "intrinsics_file": os.path.relpath(cam_json_path, scene_dir) if _exists(cam_json_path) else "",
        },
        "extrinsics": {
            "present": has_extr,
            "file": extr_file_rel,
        },
        "lidar_points": {
            "avg": int(avg_pts),
            "min": min_pts,
            "max": max_pts,
        },
        "pose": {
            "present": has_pose,
            "coverage": round(pose_coverage, 3),
            "pairs_sampled": [{"from": a, "to": b} for (a, b) in pose_pairs],
        },
        "tags": tag_list,
    }

    # flat row for global table
    row = {
        "scene_id": scene_id,
        "num_frames": num_frames,
        "duration_s": round(duration_s, 3),
        "paired_fps": round(fps_pair, 3),
        "median_dt_ms": round(median_dt_ms, 3),
        "p95_dt_ms": round(p95_dt_ms, 3),
        "cam_width": width,
        "cam_height": height,
        "cam_model": cam_model,
        "extrinsics_present": has_extr,
        "extrinsics_file": extr_file_rel,
        "lidar_avg_pts": int(avg_pts),
        "lidar_min_pts": min_pts,
        "lidar_max_pts": max_pts,
        "pose_present": has_pose,
        "pose_coverage": round(pose_coverage, 3),
        "tags": ",".join(tag_list),
    }

    return tags_json, row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root (e.g., out/dataset)")
    ap.add_argument("--write-parquet", action="store_true", help="Write <root>/tags.parquet")
    ap.add_argument("--write-csv", action="store_true", help="Write <root>/tags.csv")
    ap.add_argument("--max-poses-to-sample", type=int, default=100, help="Limit pose files to inspect for pair IDs")
    args = ap.parse_args()

    scenes = sorted(d for d in os.listdir(args.root) if d.startswith("scene_"))
    if not scenes:
        print("No scenes found."); return

    rows = []
    for sc in scenes:
        scene_dir = os.path.join(args.root, sc)
        try:
            tags_json, row = process_scene(scene_dir, max_poses_to_sample=args.max_poses_to_sample)
            # write per-scene tags.json
            out_path = os.path.join(scene_dir, "tags.json")
            with open(out_path, "w") as f:
                json.dump(tags_json, f, indent=2)
            print(f"[{sc}] tags.json written. tags={tags_json['tags']}")
            rows.append(row)
        except Exception as e:
            print(f"[{sc}] tagging failed: {e}")

    # global outputs
    if rows and args.write_parquet:
        if not _HAS_PA:
            print("WARNING: --write-parquet was requested but pyarrow is not installed.")
        else:
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, os.path.join(args.root, "tags.parquet"))
            print(f"Wrote {os.path.join(args.root, 'tags.parquet')} with {len(rows)} rows")

    if rows and args.write_csv:
        import csv
        out_csv = os.path.join(args.root, "tags.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote {out_csv} with {len(rows)} rows")

if __name__ == "__main__":
    main()
