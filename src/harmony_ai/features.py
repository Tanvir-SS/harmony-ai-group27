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


def extract_fixed_feature_vector(audio_path: str, cfg_feat: dict) -> List[float]:
    """
    Extract a fixed-length vector from a single audio file based on cfg_feat.
    """
    sr = int(cfg_feat.get("sample_rate", 22050))
    # Load mono at target rate; audioread backend allows multiple formats if FFmpeg is present
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    feats: List[float] = []

    # Helpers
    def _mean(x: np.ndarray) -> float:
        return float(np.nanmean(x)) if x.size else 0.0

    # Tempo (BPM)
    if cfg_feat.get("tempo", True):
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        feats.append(float(tempo))

    # Zero-crossing rate (mean)
    if cfg_feat.get("use_zcr", True):
        zcr = librosa.feature.zero_crossing_rate(y=y)  # shape (1, frames)
        feats.append(_mean(zcr))

    # Spectral centroid (mean)
    if cfg_feat.get("use_spectral_centroid", True):
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)  # shape (1, frames)
        feats.append(_mean(sc))

    # MFCC means
    n_mfcc = int(cfg_feat.get("n_mfcc", 13))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # shape (n_mfcc, frames)
    feats += [ _mean(mfcc[i]) for i in range(n_mfcc) ]

    # 12-bin chroma means
    if cfg_feat.get("use_chroma", True):
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)  # shape (12, frames)
        feats += [ _mean(chroma[i]) for i in range(12) ]

    # Replace any lingering NaNs/Infs
    feats = list(np.nan_to_num(np.asarray(feats, dtype=float), nan=0.0, posinf=0.0, neginf=0.0))
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
