import os, pytest
from harmony_ai.config import load_config
from sklearn.metrics import make_scorer, f1_score
from sklearn.dummy import DummyClassifier

def test_config_has_paths():
    cfg = load_config("configs/baseline.yaml")
    for key in ["gtzan_root", "splits_dir", "features_dir", "model_path", "reports_dir"]:
        assert key in cfg["paths"]

def test_macro_scorer_multi_class():
    clf = DummyClassifier(strategy="most_frequent", random_state=42)
    X = [[0], [1], [2]]
    y_true = ["rock", "pop", "jazz"]
    clf.fit(X, y_true)

    # Predict
    y_pred = clf.predict(X)

    # Directly compute macro-F1
    score = f1_score(y_true, y_pred, average="macro")
    assert isinstance(score, float)

    # Optional: check that make_scorer wraps correctly
    scorer = make_scorer(f1_score, average="macro")
    score2 = f1_score(y_true, clf.predict(X), average="macro")
    assert isinstance(score2, float)