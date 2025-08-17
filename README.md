# 🧩 Multi-Sensor ML Dataset Pipeline (nuScenes-lite format)

This repository converts `.mcap` logs (camera + LiDAR) into a lightweight **nuScenes-lite**-style dataset format. Ideal for fast experimentation with PyTorch.


## Features

- Input: `.mcap` file with camera, LiDAR, camera_info, `/tf`, `/tf_static`
- Output: scene folders with:
  - `.jpg` images
  - `.bin` point clouds (Nx4: x, y, z, intensity)
  - Calibration files
  - Optional pose transforms
- JSON and Parquet sample indexes
- Scene segmentation + dataset building
- Diagnostics and visualization
- PyTorch Dataset class (`multi_sensor_dataset.py`)
- Docker support
- Example scripts in `examples/`

---

## 🚀 Quickstart (Docker)

```bash
# Build Docker image
docker build -t multi_sensor-pipeline .

# Run container with current repo mounted
docker run -it --rm -v $(pwd):/workspace -w /workspace multi_sensor-pipeline
```

---

## 00_inspect_mcap.py

List topics, message counts, and optionally extract one image and LiDAR sample.

```bash
python src/00_inspect_mcap.py --mcap kitti.mcap --save-sample
```

Outputs:
- `out_smoke/frame0.jpg`
- `out_smoke/cloud0.bin`

---

## 01a_dump_gaps.py (optional)

Analyze message gaps and segment continuity.

```bash
python src/01a_dump_gaps.py --mcap kitti.mcap --time-source sensor
```

Outputs:
- Text report: `out/gaps_report.txt`

---

## 01b_plot_gaps.py (optional)

Generate diagnostic plots for LiDAR/camera gaps and segment timelines.

```bash
python src/01b_plot_gaps.py \
  --mcap kitti.mcap \
  --max-gap-ms 150 \
  --time-source sensor \
  --out-dir out/figs
```

Outputs:
- `lidar_gaps.png`
- `camera_gaps.png`
- `segments_timeline.png`

---

## 01_split_chunks.py

Splits `.mcap` into usable segments based on gap/duration constraints.

```bash
python src/01_split_chunks.py
```

Config: `configs/split_chunks.yaml`

---

##  02_build_dataset.py

Builds the final dataset from `.mcap` + chunk manifest.

```bash
python src/02_build_dataset.py
```

Config: `configs/build_dataset.yaml`

Outputs:
- Per-scene folders (`scene_XXX/`) containing:
  - `images/left/*.jpg`
  - `lidar/*.bin` (float32 Nx4: x, y, z, intensity)
  - `poses/*.json` (optional, dynamic TF pose)
  - `calib/camera_left.json` and `calib/T_*.json`
  - `index/samples.json` (scene-level sample index)
- Global:
  - `manifest.parquet` (optional, if `write_parquet=True)



## 03_validate_dataset.py

Checks dataset structure and file consistency.

```bash
python src/03_validate_dataset.py \
  --root out/dataset \
  --samples-per-scene 5 \
  --check-parquet
```

---

## 04_make_tags.py

Generates scene metadata and global tag summaries.

```bash
python src/04_make_tags.py \
  --root out/dataset \
  --write-parquet \
  --write-csv
```

Outputs:
- Per-scene: `tags.json`
- Global: `tags.parquet`, `tags.csv`


## PyTorch Dataset

Defined in [`src/multi_sensor_dataset.py`](src/multi_sensor_dataset.py)

### Example

```python
from src.multi_sensor_dataset import MultiSensorNuScenesLite, collate_default
from torch.utils.data import DataLoader

ds = MultiSensorNuScenesLite(root="out/dataset")
dl = DataLoader(ds, batch_size=4, collate_fn=collate_default)

batch = next(iter(dl))
print(batch["image"].shape)         # [B, 3, H, W]
print(batch["pointcloud"][0].shape) # [N, 4]
```

---

## 🧪 Examples

```bash
python examples/load_dataset.py         # Load and iterate samples
python examples/visualize_sample.py     # Visualize one image
python examples/analyze_tags.py         # Print tag stats from tags.csv
```

---

## 📁 Directory Structure

```
multi-sensor-ml-dataset-pipeline/
├── configs/                  # YAML config files
├── examples/                 # Example scripts
├── out/                      # Output folder
│   ├── dataset/              # Final dataset
│   └── figs/                 # Plot outputs
├── src/                      # All main code
│   ├── 00_inspect_mcap.py
│   ├── 01a_dump_gaps.py
│   ├── 01b_plot_gaps.py
│   ├── 01_split_chunks.py
│   ├── 02_build_dataset.py
│   ├── 03_validate_dataset.py
│   ├── 04_make_tags.py
│   ├── multi_sensor_dataset.py
│   └── utils.py
├── kitti.mcap                # Example input (optional)
├── Dockerfile
├── requirements.txt
└── README.md
```


##  Dependencies

- Python 3.10+
- `numpy`, `opencv-python`
- `matplotlib` (for plotting)
- `pyarrow` (for Parquet support)
- `torch` (for loading in PyTorch)
- `mcap`, `mcap_ros2`
- `pointcloud2` (for decoding PointCloud2)

Install:

```bash
pip install -r requirements.txt
```


## 🔖 License

MIT