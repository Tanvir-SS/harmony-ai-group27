import argparse
from harmony_ai.config import load_config
from harmony_ai.features import extract_features_for_splits

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    extract_features_for_splits(cfg)
