from __future__ import annotations
import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def _load_xy(csv_path: str):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y

def train_and_save(cfg: dict) -> None:
    train_csv = os.path.join(cfg["paths"]["features_dir"], "train.csv")
    X, y = _load_xy(train_csv)

    model_cfg = cfg.get("model", {})
    params = model_cfg.get("params", {})
    clf = DecisionTreeClassifier(**params)
    clf.fit(X, y)

    out_path = cfg["paths"]["model_path"]
    joblib.dump(clf, out_path)
    print(f"Saved model to {out_path}")

def load_model(path: str):
    return joblib.load(path)
