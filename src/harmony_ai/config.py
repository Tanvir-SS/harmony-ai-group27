import os
import yaml

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Normalize and ensure dirs
    _ensure_dirs(cfg)
    return cfg

def _ensure_dirs(cfg: dict) -> None:
    p = cfg.get("paths", {})
    for key in ["splits_dir", "features_dir", "reports_dir"]:
        d = p.get(key)
        if d:
            os.makedirs(d, exist_ok=True)
    # parent for model_path
    model_path = p.get("model_path")
    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
