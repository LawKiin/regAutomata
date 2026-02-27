# prepare_datasets.py — run once to create all 3
from sklearn.datasets import load_iris, load_wine, load_diabetes
import pandas as pd, os

for name, loader in [("iris", load_iris), ("wine", load_wine), ("diabetes", load_diabetes)]:
    os.makedirs(f"data/{name}", exist_ok=True)
    df = loader(as_frame=True).data
    df.to_csv(f"data/{name}/{name}_dataset.csv", index=False)
    print(f"{name}: {df.shape}  cols: {df.columns.tolist()}")
