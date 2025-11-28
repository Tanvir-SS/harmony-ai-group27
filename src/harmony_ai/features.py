from __future__ import annotations

"""
Feature extraction utilities for HarmonyAI.

- Reads manifest CSVs (train/val/test/smoke) from cfg["paths"]["splits_dir"]
  that contain columns: filepath, and either label or genre.
- Extracts fixed-length audio features (tempo, ZCR mean, spectral centroid mean,
  13 MFCC means, 12 chroma means) using librosa.
- Writes features/{split}.csv with numeric columns + a 'label' column.

This module intentionally keeps a stable column order so downstream
training/evaluation remains deterministic.
"""

from typing import Any, Iterable, List, Tuple, TypeVar
import os
import pandas as pd
import numpy as np
import librosa



# ----------------------------
# Optional progress bar helper
# ----------------------------
T = TypeVar("T")


import tqdm as _tqdm  # module

def pbar(it: Iterable[T], **kwargs: Any) -> Iterable[T]:
        # tqdm.tqdm is a class callable like a function; this returns an iterator wrapper
        return _tqdm.tqdm(it, **kwargs)



# ----------------------------
# Public API
# ----------------------------


def feature_names(cfg_feat: dict) -> List[str]:
    """
    Construct the feature column names in the exact order they are computed.
    """
    names: List[str] = []

    # Tempo
    if cfg_feat.get("tempo", True):
        names.append("tempo")

    # ZCR
    if cfg_feat.get("use_zcr", True):
        names += ["zcr_mean", "zcr_std"]

    # Spectral centroid
    if cfg_feat.get("use_spectral_centroid", True):
        names += ["spectral_centroid_mean", "spectral_centroid_std"]

    # RMSE
    if cfg_feat.get("use_rmse", True):
        names += ["rmse_mean", "rmse_std"]

    # Spectral bandwidth
    if cfg_feat.get("use_spectral_bandwidth", True):
        names += ["spec_bandwidth_mean", "spec_bandwidth_std"]

    # Spectral rolloff
    if cfg_feat.get("use_spectral_rolloff", True):
        names += ["spec_rolloff_mean", "spec_rolloff_std"]

    # MFCCs
    n_mfcc = int(cfg_feat.get("n_mfcc", 13))
    names += [f"mfcc_{i+1}_mean" for i in range(n_mfcc)]
    names += [f"mfcc_{i+1}_std"  for i in range(n_mfcc)]

    # Delta MFCCs
    if cfg_feat.get("use_mfcc_delta", True):
        names += [f"mfcc_delta_{i+1}_mean" for i in range(n_mfcc)]
        names += [f"mfcc_delta_{i+1}_std"  for i in range(n_mfcc)]

    # Delta-delta MFCCs
    if cfg_feat.get("use_mfcc_delta2", True):
        names += [f"mfcc_delta2_{i+1}_mean" for i in range(n_mfcc)]
        names += [f"mfcc_delta2_{i+1}_std"  for i in range(n_mfcc)]

    # Chroma
    if cfg_feat.get("use_chroma", True):
        names += [f"chroma_{i+1}_mean" for i in range(12)]
        names += [f"chroma_{i+1}_std"  for i in range(12)]

    # Spectral contrast (7 bands by default)
    if cfg_feat.get("use_spectral_contrast", True):
        for i in range(7):
            names.append(f"spec_contrast_{i+1}_mean")
        for i in range(7):
            names.append(f"spec_contrast_{i+1}_std")

    # Tonnetz (6 dims)
    if cfg_feat.get("use_tonnetz", True):
        for i in range(6):
            names.append(f"tonnetz_{i+1}_mean")
        for i in range(6):
            names.append(f"tonnetz_{i+1}_std")

    return names



def extract_fixed_feature_vector(audio_path: str, cfg_feat: dict) -> List[float]:
    """
    Extract a fixed-length vector from a single audio file based on cfg_feat.
    Includes tempo, ZCR, spectral features, MFCCs + deltas, chroma, etc.
    """
    sr = int(cfg_feat.get("sample_rate", 22050))
    duration = float(cfg_feat.get("duration", 30.0))   
    offset = float(cfg_feat.get("offset", 0.0))
    res_type = cfg_feat.get("res_type", "kaiser_fast")

    y, sr = librosa.load(
        audio_path,
        sr=sr,
        mono=True,
        duration=duration,
        offset=offset,
        res_type=res_type,
    )

    feats: List[float] = []

    def _mean_std(x: np.ndarray) -> tuple[float, float]:
        if x.size == 0:
            return 0.0, 0.0
        x = np.asarray(x, dtype=float)
        return float(np.nanmean(x)), float(np.nanstd(x))

    # Tempo
    if cfg_feat.get("tempo", True):
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        feats.append(float(tempo))

    # ZCR
    if cfg_feat.get("use_zcr", True):
        zcr = librosa.feature.zero_crossing_rate(y=y)
        m, s = _mean_std(zcr)
        feats.extend([m, s])

    # Spectral centroid
    if cfg_feat.get("use_spectral_centroid", True):
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)
        m, s = _mean_std(sc)
        feats.extend([m, s])

    # RMSE
    if cfg_feat.get("use_rmse", True):
        rmse = librosa.feature.rms(y=y)
        m, s = _mean_std(rmse)
        feats.extend([m, s])

    # Spectral bandwidth
    if cfg_feat.get("use_spectral_bandwidth", True):
        sbw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        m, s = _mean_std(sbw)
        feats.extend([m, s])

    # Spectral rolloff
    if cfg_feat.get("use_spectral_rolloff", True):
        roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
        m, s = _mean_std(roll)
        feats.extend([m, s])

    # MFCCs
    n_mfcc = int(cfg_feat.get("n_mfcc", 13))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    for i in range(n_mfcc):
        m, s = _mean_std(mfcc[i])
        feats.extend([m, s])

    # Delta MFCCs
    if cfg_feat.get("use_mfcc_delta", True):
        delta_mfcc = librosa.feature.delta(mfcc, order=1)
        for i in range(n_mfcc):
            m, s = _mean_std(delta_mfcc[i])
            feats.extend([m, s])

    # Delta-delta MFCCs
    if cfg_feat.get("use_mfcc_delta2", True):
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        for i in range(n_mfcc):
            m, s = _mean_std(delta2_mfcc[i])
            feats.extend([m, s])

    # Chroma
    if cfg_feat.get("use_chroma", True):
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for i in range(chroma.shape[0]):
            m, s = _mean_std(chroma[i])
            feats.extend([m, s])

    # Spectral contrast
    if cfg_feat.get("use_spectral_contrast", True):
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for i in range(contrast.shape[0]):
            m, s = _mean_std(contrast[i])
            feats.extend([m, s])

    # Tonnetz
    if cfg_feat.get("use_tonnetz", True):
        y_harm = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
        for i in range(tonnetz.shape[0]):
            m, s = _mean_std(tonnetz[i])
            feats.extend([m, s])

    # Clean NaNs/Infs
    feats = list(
        np.nan_to_num(
            np.asarray(feats, dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    )
    return feats



def extract_features_for_splits(cfg: dict) -> None:
    """
    For each split CSV (train/val/test/smoke), extract features and write features/{split}.csv
    with numeric columns + a 'label' column.
    """
    splits_dir = cfg["paths"]["splits_dir"]
    features_dir = cfg["paths"]["features_dir"]
    os.makedirs(features_dir, exist_ok=True)

    names = feature_names(cfg["features"])

    for split in ["train", "val", "test", "smoke"]:
        split_csv = os.path.join(splits_dir, f"{split}.csv")
        rows = _read_split_csv(split_csv)
        if not rows:
            # Split missing or empty; skip quietly
            continue

        data: List[List[float]] = []
        labels: List[str] = []

        for fp, label in pbar(rows, desc=f"Extracting {split}", unit="file"):
            try:
                # Normalize path separators so Windows backslashes don't break anything
                fp_norm = os.path.normpath(fp)
                vec = extract_fixed_feature_vector(fp_norm, cfg["features"])
                if len(vec) != len(names):
                    raise RuntimeError(
                        f"Feature length mismatch ({len(vec)} vs {len(names)}) for {fp_norm}"
                    )
                data.append(vec)
                labels.append(str(label))
            except Exception as e:
                # Continue on file-level errors to keep the batch moving
                print(f"[WARN] Skipping {fp}: {e}")

        if not data:
            print(f"[WARN] No features written for {split} (no rows or all failed).")
            continue

        df = pd.DataFrame(data, columns=names)
        df["label"] = labels
        out_csv = os.path.join(features_dir, f"{split}.csv")
        df.to_csv(out_csv, index=False)
        print(f"Wrote features: {out_csv} (rows={len(df)}, cols={len(df.columns)})")


# ----------------------------
# Internal helpers
# ----------------------------
def _read_split_csv(csv_path: str) -> List[Tuple[str, str]]:
    """
    Read a manifest CSV and return a list of (filepath, label) pairs.
    Accepts either a 'label' or 'genre' column for the target.
    """
    if not os.path.exists(csv_path):
        return []

    df = pd.read_csv(csv_path)
    if "filepath" not in df.columns:
        raise ValueError(f"{csv_path} must have a 'filepath' column")

    # Accept either 'label' (legacy) or 'genre' (your manifests)
    label_col: str | None
    if "label" in df.columns:
        label_col = "label"
    elif "genre" in df.columns:
        label_col = "genre"
    else:
        raise ValueError(f"{csv_path} must have a 'label' or 'genre' column")

    # Return as list of tuples
    filepaths = df["filepath"].tolist()
    labels = df[label_col].tolist()
    return list(zip(filepaths, labels))
