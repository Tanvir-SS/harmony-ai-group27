import argparse
from harmony_ai.config import load_config
from harmony_ai.model import train_and_save

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    train_and_save(cfg)
