from __future__ import annotations
import os
import csv
import pandas as pd
import numpy as np
import librosa

def feature_names(cfg_feat: dict) -> list[str]:
    names = []
    if cfg_feat.get("tempo", True):
        names.append("tempo")
    if cfg_feat.get("use_zcr", True):
        names.append("zcr_mean")
    if cfg_feat.get("use_spectral_centroid", True):
        names.append("spectral_centroid_mean")
    n_mfcc = int(cfg_feat.get("n_mfcc", 13))
    names += [f"mfcc_{i+1}_mean" for i in range(n_mfcc)]
    if cfg_feat.get("use_chroma", True):
        names += [f"chroma_{i+1}_mean" for i in range(12)]
    return names

def extract_fixed_feature_vector(audio_path: str, cfg_feat: dict) -> list[float]:
    sr = int(cfg_feat.get("sample_rate", 22050))
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    feats = []

    if cfg_feat.get("tempo", True):
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        feats.append(float(tempo))

    if cfg_feat.get("use_zcr", True):
        zcr = librosa.feature.zero_crossing_rate(y=y)
        feats.append(float(np.mean(zcr)))

    if cfg_feat.get("use_spectral_centroid", True):
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)
        feats.append(float(np.mean(sc)))

    n_mfcc = int(cfg_feat.get("n_mfcc", 13))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feats += [float(np.mean(mfcc[i])) for i in range(n_mfcc)]

    if cfg_feat.get("use_chroma", True):
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        feats += [float(np.mean(chroma[i])) for i in range(12)]

    return feats

def _read_split_csv(csv_path: str) -> list[tuple[str,str]]:
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((row["filepath"], row["label"]))
    return rows

def extract_features_for_splits(cfg: dict) -> None:
    splits_dir = cfg["paths"]["splits_dir"]
    features_dir = cfg["paths"]["features_dir"]
    os.makedirs(features_dir, exist_ok=True)
    names = feature_names(cfg["features"])

    for split in ["train", "val", "test", "smoke"]:
        split_csv = os.path.join(splits_dir, f"{split}.csv")
        rows = _read_split_csv(split_csv)
        if not rows:
            continue
        data = []
        labels = []
        for fp, label in rows:
            vec = extract_fixed_feature_vector(fp, cfg["features"])
            data.append(vec)
            labels.append(label)
        df = pd.DataFrame(data, columns=names)
        df["label"] = labels
        out_csv = os.path.join(features_dir, f"{split}.csv")
        df.to_csv(out_csv, index=False)
        print(f"Wrote features: {out_csv}")
