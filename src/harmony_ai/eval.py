from __future__ import annotations
import os, json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def _load_xy(csv_path: str):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y

def evaluate_and_report(cfg: dict, model) -> None:
    split = cfg.get("eval", {}).get("split", "test")
    csv_path = os.path.join(cfg["paths"]["features_dir"], f"{split}.csv")
    X, y_true = _load_xy(csv_path)
    y_pred = model.predict(X)
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=sorted(y_true.unique()))

    reports_dir = cfg["paths"]["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)

    # Save metrics JSON
    metrics_path = os.path.join(reports_dir, "baseline_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "n_samples": int(len(y_true))}, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Save confusion matrix plot
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y_true.unique()))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    fig.tight_layout()
    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {cm_path}")

def plot_confusion_matrix_cli(cfg: dict) -> None:
    from harmony_ai.model import load_model
    model = load_model(cfg["paths"]["model_path"])
    evaluate_and_report(cfg, model)
