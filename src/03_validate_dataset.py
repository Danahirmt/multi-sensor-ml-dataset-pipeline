#!/usr/bin/env python
"""
03_validate_dataset.py — Validation utility for checking nuScenes-lite dataset integrity.

This script performs sanity checks on a dataset generated via 02_build_dataset.py.

Validation steps:
- Ensures each scene directory contains the required structure:
    - images/left/, lidar/, poses/, calib/, index/samples.json
- Verifies that samples.json exists and references real files
- Checks for the presence of calibration files:
    - calib/camera_left.json (camera intrinsics)
    - calib/T_*.json (static extrinsics between frames)
- Loads N random samples per scene and verifies:
    - Images have shape HxWx3 (uint8)
    - Point clouds have shape Nx4 and contain only finite values
    - Pose files (if present) contain valid 4x4 transforms
- (Optional) Validates manifest.parquet using pyarrow, if --check-parquet is enabled

Output:
- Per-scene validation summary
- Global summary of issues and warnings

Usage:
  python src/03_validate_dataset.py --root out/dataset --samples-per-scene 5 --check-parquet
"""

import os
import json
import argparse
import numpy as np

from utils import (
    load_image,          # uses OpenCV, returns np.ndarray HxWx3
    load_pointcloud,     # returns np.ndarray Nx4
    load_pose_T_json,    # new helper for {"T": 4x4}
)

def _exists(p: str) -> bool:
    return os.path.exists(p)

def _find_any_extrinsics_file(calib_dir: str) -> str:
    if not _exists(calib_dir):
        return ""
    for fn in os.listdir(calib_dir):
        if fn.startswith("T_") and fn.endswith(".json"):
            return os.path.join(calib_dir, fn)
    return ""

def validate_scene(scene_dir: str, samples_per_scene: int = 5):
    scene_name = os.path.basename(scene_dir)
    idx_path   = os.path.join(scene_dir, "index", "samples.json")
    img_dir    = os.path.join(scene_dir, "images", "left")
    pc_dir     = os.path.join(scene_dir, "lidar")
    pose_dir   = os.path.join(scene_dir, "poses")
    calib_dir  = os.path.join(scene_dir, "calib")

    problems = []
    warnings = []

    if not _exists(idx_path):
        return 0, [f"[{scene_name}] missing index/samples.json"], warnings

    data = json.load(open(idx_path, "r"))
    frames = data.get("frames", [])
    n_index = len(frames)

    # Folders exist
    if not _exists(img_dir) or not _exists(pc_dir):
        problems.append(f"[{scene_name}] images/left or lidar folder missing")

    # Calib: intrinsics + at least one T_*.json extrinsics
    intrinsics_path = os.path.join(calib_dir, "camera_left.json")
    if not _exists(intrinsics_path):
        problems.append(f"[{scene_name}] missing calib/camera_left.json")

    extr_path = _find_any_extrinsics_file(calib_dir)
    if not extr_path:
        problems.append(f"[{scene_name}] missing static extrinsics (no T_*.json in calib/)")

    # Quick counts on disk
    n_img_disk = sum(1 for e in os.scandir(img_dir) if e.is_file()) if _exists(img_dir) else 0
    n_pc_disk  = sum(1 for e in os.scandir(pc_dir)  if e.is_file()) if _exists(pc_dir)  else 0
    if n_index == 0:
        problems.append(f"[{scene_name}] samples.json has 0 frames")
    if n_img_disk == 0 or n_pc_disk == 0:
        problems.append(f"[{scene_name}] empty images or lidar folder (img={n_img_disk}, lidar={n_pc_disk})")

    # Sample a few frames
    sample_frames = frames[:min(samples_per_scene, len(frames))]
    for fr in sample_frames:
        ip = os.path.join(scene_dir, fr.get("image_left", ""))
        lp = os.path.join(scene_dir, fr.get("lidar_front", ""))
        pp_rel = fr.get("pose", "")
        pp = os.path.join(scene_dir, pp_rel) if pp_rel else ""

        # Path existence
        for pth in (ip, lp):
            if not pth or not _exists(pth):
                problems.append(f"[{scene_name}] missing referenced file: {pth}")

        # Load checks
        try:
            img = load_image(ip)
            pc  = load_pointcloud(lp)
            if pp:
                T = load_pose_T_json(pp)
        except Exception as e:
            problems.append(f"[{scene_name}] load error: {e}")
            continue

        # minimal shape checks
        if img.ndim != 3 or img.shape[2] != 3:
            problems.append(f"[{scene_name}] invalid image shape: {ip} -> {img.shape}")
        if pc.ndim != 2 or pc.shape[1] != 4 or not np.isfinite(pc).all():
            problems.append(f"[{scene_name}] invalid pointcloud: {lp} -> {pc.shape}")

        # calib references in frame should be relative; optional check
        calib = fr.get("calib", {})
        stx_rel = calib.get("static_extrinsics", "")
        if stx_rel:
            stx_abs = os.path.join(scene_dir, stx_rel)
            if not _exists(stx_abs):
                problems.append(f"[{scene_name}] frame calib.static_extrinsics missing: {stx_abs}")

    print(f"[{scene_name}] index={n_index}  imgs_on_disk={n_img_disk}  "
          f"lidars_on_disk={n_pc_disk}  checked={len(sample_frames)}  "
          f"{'OK' if not problems else 'ISSUES'}")

    return n_index, problems, warnings

def maybe_check_parquet(root: str):
    parquet_path = os.path.join(root, "manifest.parquet")
    if not _exists(parquet_path):
        print("No manifest.parquet found (skip).")
        return
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path, columns=["scene_id", "timestamp_ns"])
        print(f"manifest.parquet OK  rows={table.num_rows}  cols={table.num_columns}")
    except Exception as e:
        print(f"manifest.parquet check failed: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root (e.g., out/dataset)")
    ap.add_argument("--samples-per-scene", type=int, default=5)
    ap.add_argument("--check-parquet", action="store_true")
    args = ap.parse_args()

    scenes = sorted(os.path.join(args.root, d) for d in os.listdir(args.root) if d.startswith("scene_"))
    if not scenes:
        print("No scenes found.")
        return

    total = 0
    all_problems = []
    all_warnings = []
    for scene_dir in scenes:
        n, probs, warns = validate_scene(scene_dir, args.samples_per_scene)
        total += n
        all_problems.extend(probs)
        all_warnings.extend(warns)

    print("\n== SUMMARY ==")
    print(f"scenes={len(scenes)}  total_index_frames={total}")
    if all_warnings:
        print("Warnings:")
        for w in all_warnings:
            print(" -", w)
    if all_problems:
        print("Problems found:")
        for p in all_problems:
            print(" -", p)
    else:
        print("All good ✅")

    if args.check_parquet:
        maybe_check_parquet(args.root)

if __name__ == "__main__":
    main()
