# examples/visualize_sample.py

import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from multi_sensor_dataset import MultiSensorNuScenesLite

ds = MultiSensorNuScenesLite(root="out/dataset", return_paths=True)
sample = ds[0]

img = sample["image"].permute(1, 2, 0).numpy()  # C,H,W → H,W,C
plt.imshow(img)
plt.title(f"Scene: {sample['scene']}")
plt.show()

print("Pointcloud shape:", sample["pointcloud"].shape)
print("Pose matrix:\n", sample["pose_T"])
