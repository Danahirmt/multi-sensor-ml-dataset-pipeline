# Evidence from this run

This section contains the full sequence of commands and outputs used to generate, analyze, and validate the dataset from the original MCAP file. It demonstrates:
- End-to-end reproducibility using only the provided scripts and YAML configs.
- Sensor statistics and gap analysis motivating the chunking strategy.
- Successful creation of 3 scenes (~20.5s each) under strict timing constraints.
- Dataset integrity validation, auto-tagging, and PyTorch dataset loading.

All commands were run inside a Docker container using the current repository setup.

## 1. MCAP Inspection

```bash
python src/00_inspect_mcap.py --mcap kitti.mcap --save-sample
```

```text
== Topics ==
- /sensor/camera/left/camera_info
- /sensor/camera/left/image_raw/compressed
- /sensor/lidar/front/points
- /tf
- /tf_static

== Message counts ==
/sensor/camera/left/camera_info: 1087
/sensor/camera/left/image_raw/compressed: 1087
/sensor/lidar/front/points: 1081
/tf: 1101
/tf_static: 1

== Time ranges (ns) ==
/sensor/camera/left/camera_info: 1749641679709714944 .. 1749641793738514944  (~114.03s)
/sensor/camera/left/image_raw/compressed: 1749641679709714944 .. 1749641793738514944  (~114.03s)
/sensor/lidar/front/points: 1749641679709714944 .. 1749641793738514944  (~114.03s)
/tf: 1749641679709714944 .. 1749641793738514944  (~114.03s)
/tf_static: 1749641679709714944 .. 1749641679709714944  (~0.00s)

Saved image: out_smoke/frame0.jpg  
Saved point cloud: out_smoke/cloud0.bin  shape=(122829, 4)
```

---

## 2. Gap Analysis

```bash
python src/01a_dump_gaps.py --mcap kitti.mcap --time-source sensor
```

```text
[LiDAR] gaps ms: count=1080, median=103.63, p95=103.80, max=1140.60  
[Camera] gaps ms: count=1086, median=103.63, p95=103.81, max=1140.06  

LiDAR segments (s): [51.727, 50.689, 9.332]  
Camera segments (s): [20.632, 19.591, 20.523, 20.52, 20.53, 10.265]  
Common segments (s): [20.632, 19.591, 10.157, 9.225, 20.52, 20.53, 9.332]

Exists ≥ 40s common chunk?  NO
```

---

## 3. Segment Timeline Plot

```bash
python src/01b_plot_gaps.py --mcap kitti.mcap --max-gap-ms 150 --time-source sensor --out-dir out/figs
```

```text
[sensor] LiDAR segments (s): [51.727, 50.689, 9.332]
[sensor] Camera segments (s): [20.632, 19.591, 20.523, 20.52, 20.53, 10.265]
[sensor] Common segments (s): [20.632, 19.591, 10.157, 9.225, 20.52, 20.53, 9.332]
Saved figures to: out/figs
```

---

## 4. Chunk Splitting

```bash
cat configs/split_chunks.yaml
```

```yaml
mcap_path: "kitti.mcap"
out: "out/chunks_demo.json"
min_length_s: 20
max_gap_ms: 150
time_source: "sensor"
debug: true
```

```bash
python src/01_split_chunks.py
```

```text
[sensor] Collected: lidar=1081 ts, cam=1087 ts
[sensor] LiDAR gaps ms: median=103.6, p95=103.8, max=1140.6
[sensor] Camera gaps ms: median=103.6, p95=103.8, max=1140.1
[sensor] LiDAR segments (s): [51.727, 50.689, 9.331]
[sensor] Camera segments (s): [20.632, 19.591, 20.523, 20.52, 20.53, 10.265]
[sensor] Common segments (s): [20.632, 19.591, 10.157, 9.225, 20.52, 20.53, 9.331]
[sensor] Found 3 usable chunk(s):
  - chunk_000: 20.632s (lidar=200, cam=200)
  - chunk_001: 20.52s (lidar=199, cam=199)
  - chunk_002: 20.53s (lidar=199, cam=199)
Manifest written to: out/chunks_demo.json
```

---

## 5. Dataset Build

```bash
cat configs/build_dataset.yaml
```

```yaml
mcap_path: "kitti.mcap"
manifest: "out/chunks_demo.json"
out_dir: "out/dataset"
pair_tolerance_ms: 50
cam_frame: "camera_frame"
lidar_frame: "lidar_frame"
write_parquet: true
debug: true
```

```bash
python src/02_build_dataset.py
```

```text
Static graph:
  lidar_frame -> camera_frame
  camera_frame -> lidar_frame
scene_000: 199 paired frames.
scene_001: 198 paired frames.
scene_002: 199 paired frames.
manifest.parquet written to out/dataset/manifest.parquet
```

---

## 6. Dataset Validation

```bash
python src/03_validate_dataset.py --root out/dataset --samples-per-scene 5 --check-parquet
```

```text
[scene_000] index=199  imgs_on_disk=199  lidars_on_disk=199  checked=5  OK  
[scene_001] index=198  imgs_on_disk=198  lidars_on_disk=198  checked=5  OK  
[scene_002] index=199  imgs_on_disk=199  lidars_on_disk=199  checked=5  OK

== SUMMARY ==
scenes=3  total_index_frames=596  
All good ✅  
manifest.parquet OK  rows=596  cols=2
```

---

## 7. Auto-tagging

```bash
python src/04_make_tags.py --root out/dataset --write-parquet --write-csv
```

```text
[scene_000] tags.json written. tags=['pose', 'extrinsics', 'res=1241x376', 'fps~9.6', 'dense-lidar', 'cam_model=plumb_bob']
[scene_001] tags.json written. tags=['pose', 'extrinsics', 'res=1241x376', 'fps~9.6', 'mid-lidar', 'cam_model=plumb_bob']
[scene_002] tags.json written. tags=['pose', 'extrinsics', 'res=1241x376', 'fps~9.7', 'mid-lidar', 'cam_model=plumb_bob']
Wrote out/dataset/tags.parquet with 3 rows  
Wrote out/dataset/tags.csv with 3 rows
```

---

## 8. Dataset Access Test

```bash
python examples/load_dataset.py
```

```text
Loaded 596 samples  
Image shape: torch.Size([4, 3, 376, 1241])  
Pointcloud 0: torch.Size([102433, 4])  
Pose_T 0: torch.Size([4, 4])
```

---

## 9. Visualize Sample + Analyze Tags

```bash
python examples/visualize_sample.py
```

```text
Pointcloud shape: torch.Size([122829, 4])
Pose matrix:
 tensor([[ 1.0000e+00,  1.3187e-12, -2.3243e-10,  4.2608e-12],
         [-1.3187e-12,  1.0000e+00, -9.3145e-10, -1.1069e-10],
         [ 2.3243e-10,  9.3145e-10,  1.0000e+00,  7.5973e-09],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
```

```bash
python examples/analyze_tags.py
```

```text
    scene_id                                               tags
0  scene_000  pose,extrinsics,res=1241x376,fps~9.6,dense-lidar,cam_model=plumb_bob
1  scene_001  pose,extrinsics,res=1241x376,fps~9.6,mid-lidar,cam_model=plumb_bob
2  scene_002  pose,extrinsics,res=1241x376,fps~9.7,mid-lidar,cam_model=plumb_bob

Tag frequencies:
pose                   3  
extrinsics             3  
res=1241x376           3  
cam_model=plumb_bob    3  
fps~9.6                2  
mid-lidar              2  
dense-lidar            1  
fps~9.7                1
```
