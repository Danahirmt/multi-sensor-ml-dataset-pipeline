# examples/analyze_tags.py

import pandas as pd

df = pd.read_csv("out/dataset/tags.csv")
print(df[["scene_id", "tags"]])
print("\nTag frequencies:")
tags = df["tags"].str.split(",").explode()
print(tags.value_counts())
