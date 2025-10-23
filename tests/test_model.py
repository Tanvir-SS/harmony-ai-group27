import os, pytest
from harmony_ai.config import load_config

def test_config_has_paths():
    cfg = load_config("configs/baseline.yaml")
    for key in ["gtzan_root", "splits_dir", "features_dir", "model_path", "reports_dir"]:
        assert key in cfg["paths"]
