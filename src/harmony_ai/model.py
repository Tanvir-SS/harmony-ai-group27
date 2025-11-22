from __future__ import annotations
import os, json
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
    params = dict(model_cfg.get("params", {}))
    # Ensure determinism
    if "random_state" not in params:
        params["random_state"] = 42

    clf = DecisionTreeClassifier(**params)
    clf.fit(X, y)

    out_path = cfg["paths"]["model_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(clf, out_path)
    print(f"Saved model to {out_path}")

    # Save params alongside model
    params_path = os.path.join(cfg["paths"]["reports_dir"], "params.json")
    os.makedirs(cfg["paths"]["reports_dir"], exist_ok=True)
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"Saved params to {params_path}")

def load_model(path: str):
    return joblib.load(path)
