import os
from harmony_ai.config import load_config
from harmony_ai.features import feature_names

def test_feature_names_len():
    cfg = load_config("configs/baseline.yaml")
    names = feature_names(cfg["features"])
    # tempo + zcr + centroid + 13 mfcc + 12 chroma = 29
    assert len(names) == 29
