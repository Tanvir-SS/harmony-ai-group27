import argparse
from harmony_ai.config import load_config
from harmony_ai.eval import plot_confusion_matrix_cli

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    plot_confusion_matrix_cli(cfg)
