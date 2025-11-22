from __future__ import annotations
import os, json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
)

def _load_xy(csv_path: str):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y

def evaluate_and_report(cfg: dict, model) -> None:
    # Which split to evaluate (default "test")
    split = cfg.get("eval", {}).get("split", "test")
    csv_path = os.path.join(cfg["paths"]["features_dir"], f"{split}.csv")
    X, y_true = _load_xy(csv_path)
    y_pred = model.predict(X)

    # Metrics
    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted"))
    labels_sorted = sorted(y_true.unique().tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)

    # Outputs
    reports_dir = cfg["paths"]["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)

    # Save metrics JSON
    metrics_path = os.path.join(reports_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": split,
                "accuracy": acc,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "n_samples": int(len(y_true)),
            },
            f,
            indent=2,
        )
    print(f"Saved metrics to {metrics_path}")

    # Save classification report
    clsrep_path = os.path.join(reports_dir, "classification_report.txt")
    with open(clsrep_path, "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, labels=labels_sorted))
    print(f"Saved classification report to {clsrep_path}")

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"Confusion Matrix ({split})")
    fig.tight_layout()
    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {cm_path}")

def plot_confusion_matrix_cli(cfg: dict) -> None:
    # Keep your import path (adjust if your package/module path differs)
    from harmony_ai.model import load_model
    model = load_model(cfg["paths"]["model_path"])
    evaluate_and_report(cfg, model)