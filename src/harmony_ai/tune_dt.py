from __future__ import annotations
import os, json, argparse
from pathlib import Path
from copy import deepcopy

import yaml
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, f1_score


def _load_xy(csv_path: str):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y

def resolve_paths(cfg: dict) -> dict:
    """
    Format {run_id} placeholders and make relative paths absolute,
    relative to the project root (folder containing 'configs/').
    """
    out = deepcopy(cfg)
    run_id = out.get("run_id", "")
   
    project_root = Path(__file__).resolve().parents[2]

    for k in ("gtzan_root", "splits_dir", "features_dir", "model_path", "reports_dir"):
        if k in out.get("paths", {}):
            v = out["paths"][k]
            if isinstance(v, str):
                v = v.format(run_id=run_id)
                p = Path(v)
                if not p.is_absolute():
                    p = project_root.joinpath(p)
                out["paths"][k] = str(p)
    return out

def tune_decision_tree(cfg: dict) -> dict:
    """
    Run GridSearchCV on DecisionTree using macro-F1.
    Returns: dict with best_params and cv_results.
    """
    features_dir = cfg["paths"]["features_dir"]
    train_csv = os.path.join(features_dir, "train.csv")
    X, y = _load_xy(train_csv)

    # Base classifier with deterministic seed
    base_params = dict(cfg.get("model", {}).get("params", {}))
    if "random_state" not in base_params:
        base_params["random_state"] = 42

    clf = DecisionTreeClassifier(**base_params)

    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
    }

    scorer = make_scorer(f1_score, average="macro")
    grid = GridSearchCV(
        clf,
        param_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    grid.fit(X, y)

    best = {
        "best_score_macro_f1": float(grid.best_score_),
        "best_params": grid.best_params_,
    }
   
    results = {
        k: grid.cv_results_[k]
        for k in (
            "params",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
        )
    }
    return {"best": best, "results": results, "best_estimator": grid.best_estimator_}

def save_tuning_outputs(cfg: dict, tuning: dict) -> None:
    reports_dir = cfg["paths"]["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, "tuning_dt.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tuning["best"], f, indent=2)
    print(f"Saved tuning summary to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = resolve_paths(cfg)

    model_type = cfg.get("model", {}).get("type", "DecisionTreeClassifier")
    if model_type != "DecisionTreeClassifier":
        raise ValueError(f"Tuning script expects DecisionTreeClassifier, got: {model_type}")

    tuning = tune_decision_tree(cfg)
    save_tuning_outputs(cfg, tuning)

    # Print quick summary (so you see it in the terminal)
    best = tuning["best"]
    print(f"Best macro-F1: {best['best_score_macro_f1']:.4f}")
    print("Best params:", best["best_params"])

if __name__ == "__main__":
    main()
