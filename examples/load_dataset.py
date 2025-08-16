# examples/load_dataset.py

from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from multi_sensor_dataset import MultiSensorNuScenesLite, collate_default

ds = MultiSensorNuScenesLite(root="out/dataset", return_paths=True)
print(f"Loaded {len(ds)} samples")

dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_default)

batch = next(iter(dl))
print("Image shape:", batch["image"].shape)
print("Pointcloud 0:", batch["pointcloud"][0].shape)
print("Pose_T 0:", batch["pose_T"][0].shape if batch["pose_T"][0] is not None else None)
