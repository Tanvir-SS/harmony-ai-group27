import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

GENRES_10 = [
    "blues","classical","country","disco","hiphop",
    "jazz","metal","pop","reggae","rock"
]
ALLOWED_EXTS = {".wav", ".au"}  # some mirrors use .au

def to_posix(path: Path) -> str:
    return path.as_posix()

def md5_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def collect_df(raw_root: Path) -> pd.DataFrame:
    rows = []
    for genre in sorted(GENRES_10):
        gdir = raw_root / genre
        if not gdir.is_dir():
            print(f"[WARN] Missing genre folder: {gdir}")
            continue
        for p in sorted(gdir.rglob("*")):
            if p.suffix.lower() in ALLOWED_EXTS and p.is_file():
                rel = p.relative_to(Path.cwd())
                song_id = f"{genre}_{p.stem}"
                rows.append({
                    "song_id": song_id,
                    "filepath": to_posix(rel),
                    "genre": genre,
                    "md5": md5_file(p)
                })
    if not rows:
        raise SystemExit(f"[ERROR] No audio files found under {raw_root}")
    df = pd.DataFrame(rows)
    # Basic sanity checks
    if df["song_id"].duplicated().any():
        dups = df[df["song_id"].duplicated()]["song_id"].tolist()
        raise SystemExit(f"[ERROR] Duplicate song_ids detected: {dups[:5]} ...")
    return df

def hash_split(ids: List[str], seed: int) -> str:
    # Stable hash of split membership for provenance
    payload = "\n".join(sorted(ids)) + f"|seed={seed}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def verify_no_overlap(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str):
    inter = set(a["song_id"]).intersection(set(b["song_id"]))
    if inter:
        raise SystemExit(f"[ERROR] Overlap between {name_a} and {name_b}: {list(inter)[:5]} ...")

def stratified_splits(df: pd.DataFrame, train: float, val: float, test: float, seed: int):
    assert abs((train + val + test) - 1.0) < 1e-6, "train+val+test must equal 1"
    # First: train vs temp (val+test)
    df_train, df_temp = train_test_split(
        df, test_size=(1.0 - train), stratify=df["genre"], random_state=seed
    )
    # Then: split temp into val and test with correct proportion
    val_ratio = val / (val + test)
    df_val, df_test = train_test_split(
        df_temp, test_size=(1.0 - val_ratio), stratify=df_temp["genre"], random_state=seed
    )
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

def make_smoke(df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    parts = []
    rng = np.random.RandomState(seed)
    for g, gdf in df.groupby("genre"):
        k = min(per_class, len(gdf))
        parts.append(gdf.sample(n=k, random_state=rng))
    smoke = pd.concat(parts, axis=0).sample(frac=1.0, random_state=rng).reset_index(drop=True)
    return smoke

def genre_counts(df: pd.DataFrame) -> Dict[str, int]:
    cc = df["genre"].value_counts().to_dict()
    # Include zeros for any missing genres (helps catch gaps)
    return {g: cc.get(g, 0) for g in GENRES_10}

def main():
    ap = argparse.ArgumentParser(description="Create stratified manifests (train/val/test + smoke) for GTZAN.")
    ap.add_argument("--raw-root", required=True, help="Path to data/raw/gtzan")
    ap.add_argument("--out-dir", required=True, help="Where to write manifest CSVs")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--smoke-per-class", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_root = Path(args.raw_root).resolve()
    out_dir = Path(args.out_dir)

    print(f"[INFO] Scanning audio under: {raw_root}")
    df = collect_df(raw_root)

    # Optional: warn if genre counts deviate from expected 100
    per_genre = df["genre"].value_counts().sort_index()
    print("[INFO] Genre counts (all):")
    print(per_genre.to_string())

    # Persist all.csv (handy for plotting class balance)
    write_csv(df[["song_id","filepath","genre"]], out_dir / "all.csv")

    # Create smoke split
    smoke = make_smoke(df, per_class=args.smoke_per_class, seed=args.seed)
    write_csv(smoke[["song_id","filepath","genre"]], out_dir / "smoke.csv")

    # Create stratified train/val/test
    df_train, df_val, df_test = stratified_splits(df, args.train, args.val, args.test, args.seed)

    # Verify no overlap and stable membership
    verify_no_overlap(df_train, df_val, "train","val")
    verify_no_overlap(df_train, df_test, "train","test")
    verify_no_overlap(df_val, df_test, "val","test")

    # Save manifests
    for name, part in [("train", df_train), ("val", df_val), ("test", df_test)]:
        write_csv(part[["song_id","filepath","genre"]], out_dir / f"{name}.csv")

    # Summaries & hashes (provenance)
    summary = {
        "seed": args.seed,
        "counts": {
            "all": genre_counts(df),
            "train": genre_counts(df_train),
            "val": genre_counts(df_val),
            "test": genre_counts(df_test),
            "smoke": genre_counts(smoke),
        },
        "split_hashes": {
            "train": hash_split(df_train["song_id"].tolist(), args.seed),
            "val": hash_split(df_val["song_id"].tolist(), args.seed),
            "test": hash_split(df_test["song_id"].tolist(), args.seed),
            "smoke": hash_split(smoke["song_id"].tolist(), args.seed),
        }
    }
    (out_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2))
    print("[INFO] Wrote:", out_dir / "train.csv")
    print("[INFO] Wrote:", out_dir / "val.csv")
    print("[INFO] Wrote:", out_dir / "test.csv")
    print("[INFO] Wrote:", out_dir / "smoke.csv")
    print("[INFO] Wrote:", out_dir / "manifest_summary.json")
    print("[OK] Done.")

if __name__ == "__main__":
    main()
