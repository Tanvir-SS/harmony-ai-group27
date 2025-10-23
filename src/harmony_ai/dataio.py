import os
import csv
import glob
import random
from typing import List, Tuple

def _find_audio(gtzan_root: str) -> List[Tuple[str, str]]:
    pairs = []
    # Expect structure: gtzan_root/<genre>/*.wav
    for genre_dir in sorted(glob.glob(os.path.join(gtzan_root, "*"))):
        if not os.path.isdir(genre_dir):
            continue
        genre = os.path.basename(genre_dir)
        for wav in sorted(glob.glob(os.path.join(genre_dir, "*.wav"))):
            pairs.append((wav, genre))
    return pairs

def make_splits_from_folder(cfg: dict) -> None:
    seed = cfg.get("seed", 27)
    gtzan_root = cfg["paths"]["gtzan_root"]
    splits_dir = cfg["paths"]["splits_dir"]

    pairs = _find_audio(gtzan_root)
    if not pairs:
        raise SystemExit(f"No audio found under {gtzan_root}. Expected folders like blues/*.wav")

    # Group by genre
    by_genre = {}
    for fp, g in pairs:
        by_genre.setdefault(g, []).append(fp)

    rng = random.Random(seed)
    rows_train, rows_val, rows_test = [], [], []

    for g, files in by_genre.items():
        files = sorted(files)
        rng.shuffle(files)
        n = len(files)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        train = files[:n_train]
        val = files[n_train:n_train+n_val]
        test = files[n_train+n_val:]
        rows_train += [(f, g) for f in train]
        rows_val += [(f, g) for f in val]
        rows_test += [(f, g) for f in test]

    os.makedirs(splits_dir, exist_ok=True)
    _write_csv(os.path.join(splits_dir, "train.csv"), rows_train)
    _write_csv(os.path.join(splits_dir, "val.csv"), rows_val)
    _write_csv(os.path.join(splits_dir, "test.csv"), rows_test)

    # Smoke: take up to 1 file per genre (or 10 total)
    smoke = []
    for g, files in by_genre.items():
        smoke.append((files[0], g))
    _write_csv(os.path.join(splits_dir, "smoke.csv"), smoke[:10])
    print(f"Wrote splits to {splits_dir}")

def _write_csv(path: str, rows: List[Tuple[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "label"])
        for fp, label in rows:
            w.writerow([fp, label])
