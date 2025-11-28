from __future__ import annotations
from copy import deepcopy
import os, json
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
def _load_xy(csv_path: str):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y

def train_and_save(cfg: dict) -> None:
    # Load training features
    train_csv = os.path.join(cfg["paths"]["features_dir"], "train.csv")
    X, y = _load_xy(train_csv)

    # Read model config
    model_cfg = cfg.get("model", {})
    model_type = model_cfg.get("type", "DecisionTreeClassifier")
    params = deepcopy(model_cfg.get("params", {}))

    # Make runs reproducible if possible
    seed = cfg.get("seed", 27)
    params.setdefault("random_state", seed)

    # Choose model class based on config
    if model_type == "DecisionTreeClassifier":
        clf = DecisionTreeClassifier(**params)
    elif model_type == "RandomForestClassifier":
        clf = RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Train
    clf.fit(X, y)

    # Save model
    out_path = cfg["paths"]["model_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(clf, out_path)
    print(f"Saved model to {out_path}")

    # Save params alongside model
    params_path = os.path.join(cfg["paths"]["reports_dir"], "params.json")
    os.makedirs(cfg["paths"]["reports_dir"], exist_ok=True)
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(
            {"model_type": model_type, "params": params},
            f,
            indent=2,
        )
    print(f"Saved params to {params_path}")

def load_model(path: str):
    return joblib.load(path)
