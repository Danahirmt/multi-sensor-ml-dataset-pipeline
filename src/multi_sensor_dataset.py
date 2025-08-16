#!/usr/bin/env python
"""
multi_sensor_nuscenes_lite.py

PyTorch Dataset for loading nuScenes-lite formatted multi-sensor data.

This dataset class is designed to load the data exported by `02_build_dataset.py`,
which organizes LiDAR and camera samples in per-scene folders with synchronized timestamps,
poses, and calibration.

Features:
- Supports loading image (as torch.Tensor CxHxW), point cloud (Nx4 float32), optional pose (4x4)
- Parses intrinsic calibration (K, D, distortion model) and static extrinsics (T)
- Auto-discovers scenes or accepts a subset
- Optional image and point cloud transforms
- Default collate_fn included: stacks fixed-size data, keeps variable-length lists

Each returned sample dictionary contains:
- `image`: torch.Tensor [C,H,W], RGB image (float32 or uint8)
- `pointcloud`: torch.Tensor [N,4], point cloud (x,y,z,intensity)
- `pose_T`: torch.Tensor [4,4] or None
- `intrinsics`: dict with K, D, distortion_model, width, height
- `static_extrinsics`: dict with frame_from, frame_to, T [4,4]
- `timestamp_ns`: int64
- `scene`: scene ID string
- (Optional) file paths if `return_paths=True`

Usage:

    from multi_sensor_nuscenes_lite import MultiSensorNuScenesLite, collate_default
    from torch.utils.data import DataLoader

    dataset = MultiSensorNuScenesLite(
        root="out/dataset",
        image_to_float=True,
        image_div255=True,
        return_paths=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_default
    )

    batch = next(iter(dataloader))
    print(batch["image"].shape)         # [B, C, H, W]
    print(len(batch["pointcloud"]))     # B elements, each with shape [Ni, 4]
    print(batch["pose_T"][0].shape)     # [4,4] or None

"""


import os
import json
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import (
    load_image,  # np.ndarray HxWx3 (BGR)
    load_pointcloud,  # np.ndarray Nx4 (x,y,z,i) float32
    load_pose_T_json,  # np.ndarray 4x4 float32
)


def _to_rgb_torch(
    img_bgr: np.ndarray, to_float: bool = True, div255: bool = True
) -> torch.Tensor:
    # BGR -> RGB, HWC -> CHW
    img_rgb = img_bgr[:, :, ::-1].copy()
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous()
    if to_float:
        t = t.float()
        if div255:
            t = t / 255.0
    return t  # [C,H,W]


def _read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _maybe(path: str) -> Optional[str]:
    return path if (path and os.path.isfile(path)) else None


class MultiSensorNuScenesLite(Dataset):
    def __init__(
        self,
        root: str,
        scenes: Optional[Sequence[str]] = None,
        transform_img: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        transform_pc: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        image_to_float: bool = True,
        image_div255: bool = True,
        return_paths: bool = False,
    ):
        """
        Args:
            root: dataset root (e.g., "out/dataset")
            scenes: list of scene folder names (e.g., ["scene_000", ...]). If None, auto-discovers all.
            transform_img: optional callable(tensor[C,H,W]) -> tensor[C,H,W]
            transform_pc: optional callable(tensor[N,4]) -> tensor[N,4]
            image_to_float: convert image to float32
            image_div255: if float image, divide by 255.0
            return_paths: include absolute file paths in returned sample
        """
        self.root = root
        self.transform_img = transform_img
        self.transform_pc = transform_pc
        self.image_to_float = image_to_float
        self.image_div255 = image_div255
        self.return_paths = return_paths

        # Discover scenes
        if scenes is None:
            scenes = [d for d in os.listdir(root) if d.startswith("scene_")]
        self.scenes = sorted(scenes)

        # Build sample index across scenes (reading samples.json)
        self.samples: List[Dict] = []
        self._scene_calib_cache: Dict[
            str, Dict[str, str]
        ] = {}  # scene -> {"cam_intrinsics": ..., "static_extrinsics": ...}

        for scene in self.scenes:
            scene_dir = os.path.join(root, scene)
            idx_path = os.path.join(scene_dir, "index", "samples.json")
            if not os.path.isfile(idx_path):
                continue
            data = _read_json(idx_path)
            frames = data.get("frames", [])
            # Cache scene-level calib references if present
            # (We still trust per-frame calib entries, but many will be identical)
            if frames:
                first_calib = frames[0].get("calib", {})
                self._scene_calib_cache[scene] = {
                    "cam_intrinsics": first_calib.get("cam_intrinsics", ""),
                    "static_extrinsics": first_calib.get("static_extrinsics", ""),
                }
            for fr in frames:
                rec = {
                    "scene": scene,
                    "scene_dir": scene_dir,
                    "timestamp_ns": int(fr.get("timestamp_ns", -1)),
                    "image_rel": fr.get("image_left", ""),
                    "lidar_rel": fr.get("lidar_front", ""),
                    "pose_rel": fr.get("pose", ""),
                    "calib": {
                        "cam_intrinsics": fr.get("calib", {}).get("cam_intrinsics", ""),
                        "static_extrinsics": fr.get("calib", {}).get(
                            "static_extrinsics", ""
                        ),
                    },
                }
                self.samples.append(rec)

        # Sort by time for nicer iteration (optional)
        self.samples.sort(key=lambda r: (r["scene"], r["timestamp_ns"]))

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_calib_paths(self, rec: Dict) -> Tuple[Optional[str], Optional[str]]:
        """Return absolute paths for camera_left.json and T_*.json if referenced, else try scene-level cache."""
        scene = rec["scene"]
        scene_dir = rec["scene_dir"]
        cam_intr_rel = rec["calib"].get(
            "cam_intrinsics"
        ) or self._scene_calib_cache.get(scene, {}).get("cam_intrinsics", "")
        stx_rel = rec["calib"].get("static_extrinsics") or self._scene_calib_cache.get(
            scene, {}
        ).get("static_extrinsics", "")
        cam_intr_abs = (
            _maybe(os.path.join(scene_dir, cam_intr_rel)) if cam_intr_rel else None
        )
        stx_abs = _maybe(os.path.join(scene_dir, stx_rel)) if stx_rel else None
        return cam_intr_abs, stx_abs

    @staticmethod
    def _parse_intrinsics(json_path: str) -> Dict:
        """Return dict with width, height, K (3x3), D (list), model."""
        d = _read_json(json_path)
        # Normalize to array 3x3
        K_list = d.get("K", [0.0] * 9)
        K = np.array(K_list, dtype=np.float32).reshape(3, 3)
        out = {
            "width": int(d.get("width", 0)),
            "height": int(d.get("height", 0)),
            "K": torch.from_numpy(K),  # [3,3] float32
            "D": torch.tensor(d.get("D", []), dtype=torch.float32),  # [Nd]
            "distortion_model": d.get("distortion_model", ""),
        }
        return out

    @staticmethod
    def _parse_static_extrinsics(json_path: str) -> Dict:
        """Return dict with frame_from, frame_to, T [4x4]."""
        if json_path is None:
            return {}
        d = _read_json(json_path)
        T = torch.from_numpy(np.array(d["T"], dtype=np.float32))  # [4,4]
        return {
            "frame_from": d.get("frame_from", ""),
            "frame_to": d.get("frame_to", ""),
            "T": T,
        }

    def __getitem__(self, i: int) -> Dict[str, Union[torch.Tensor, int, dict, str]]:
        rec = self.samples[i]
        scene_dir = rec["scene_dir"]

        # Paths
        img_abs = os.path.join(scene_dir, rec["image_rel"])
        lidar_abs = os.path.join(scene_dir, rec["lidar_rel"])
        pose_abs = os.path.join(scene_dir, rec["pose_rel"]) if rec["pose_rel"] else None

        # Load image
        img_bgr = load_image(img_abs)  # HxWx3 (BGR)
        img_t = _to_rgb_torch(
            img_bgr, self.image_to_float, self.image_div255
        )  # [C,H,W]
        if self.transform_img is not None:
            img_t = self.transform_img(img_t)

        # Load pointcloud
        pc_np = load_pointcloud(lidar_abs)  # Nx4 float32
        pc_t = torch.from_numpy(pc_np.copy())  # [N,4]
        if self.transform_pc is not None:
            pc_t = self.transform_pc(pc_t)

        # Load optional pose (4x4)
        pose_T = None
        if pose_abs and os.path.isfile(pose_abs):
            T = load_pose_T_json(pose_abs)  # 4x4 float32 np
            pose_T = torch.from_numpy(T)  # [4,4]

        # Calib (intrinsics + static extrinsics)
        intr_abs, stx_abs = self._resolve_calib_paths(rec)
        intr_dict = self._parse_intrinsics(intr_abs) if intr_abs else {}
        stx_dict = self._parse_static_extrinsics(stx_abs) if stx_abs else {}

        sample: Dict[str, Union[torch.Tensor, int, dict, str, None]] = {
            "image": img_t,  # [C,H,W] float or uint8
            "pointcloud": pc_t,  # [N,4] float32
            "pose_T": pose_T,  # [4,4] float32 or None
            "timestamp_ns": rec["timestamp_ns"],  # int
            "intrinsics": intr_dict,  # {"K":[3,3], "D":[Nd], ...} or {}
            "static_extrinsics": stx_dict,  # {"frame_from","frame_to","T":[4,4]} or {}
            "scene": rec["scene"],
        }
        if self.return_paths:
            sample.update(
                {
                    "image_path": img_abs,
                    "lidar_path": lidar_abs,
                    "pose_path": pose_abs or "",
                }
            )
        return sample


def collate_default(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader:
    - Stacks images [B,C,H,W]
    - Keeps variable-length pointclouds as a list of [Ni,4]
    - Stacks poses where present; else keeps None
    - Keeps intrinsics/static_extrinsics as list of dicts
    """
    imgs = torch.stack([b["image"] for b in batch], dim=0)  # [B,C,H,W]
    pcs = [b["pointcloud"] for b in batch]  # list of [Ni,4]
    ts = torch.tensor([b["timestamp_ns"] for b in batch], dtype=torch.int64)
    scenes = [b["scene"] for b in batch]

    poses_list = [b["pose_T"] for b in batch]
    if all(p is not None for p in poses_list):
        poses = torch.stack(poses_list, dim=0)  # [B,4,4]
    else:
        poses = poses_list  # mixed availability

    intrs = [b["intrinsics"] for b in batch]
    stxs = [b["static_extrinsics"] for b in batch]

    out = {
        "image": imgs,
        "pointcloud": pcs,
        "pose_T": poses,
        "timestamp_ns": ts,
        "intrinsics": intrs,
        "static_extrinsics": stxs,
        "scene": scenes,
    }
    if "image_path" in batch[0]:
        out["image_path"] = [b["image_path"] for b in batch]
        out["lidar_path"] = [b["lidar_path"] for b in batch]
        out["pose_path"] = [b.get("pose_path", "") for b in batch]
    return out
