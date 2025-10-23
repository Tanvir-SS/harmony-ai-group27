import argparse
from harmony_ai.config import load_config
from harmony_ai.model import load_model
from harmony_ai.eval import evaluate_and_report

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    model = load_model(cfg["paths"]["model_path"])
    evaluate_and_report(cfg, model)
