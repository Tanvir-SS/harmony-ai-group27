# HarmonyAI (Group 27)

Genre classification & playlist grouping using GTZAN + Decision Tree.

## Quickstart

```bash
# 1) Create & activate venv
python -m venv .venv
# Windows
. .venv/Scripts/Activate.ps1
# macOS/Linux
# source .venv/bin/activate

# 2) Install (dev + api extras optional)
pip install --upgrade pip
pip install -e ".[dev,api]"

# 3) Put GTZAN here (genre folders):
# data/raw/gtzan/blues/*.wav, classical/*.wav, ...

# 4) Run pipeline
harmonyai splits --config configs/baseline.yaml
harmonyai features --config configs/baseline.yaml
harmonyai train --config configs/baseline.yaml
harmonyai eval --config configs/baseline.yaml

# Optional: run API after training
uvicorn harmony_ai.api:app --reload --port 8000
```

## Layout

- `src/harmony_ai/` — reusable library code
- `scripts/` — thin CLI wrappers (optional)
- `data/` — raw dataset & split CSVs
- `features/` — feature CSVs
- `models/` — trained models
- `reports/` — metrics, plots
- `configs/` — YAML configs

## Notes
- Requires Python 3.10–3.12.
- Install ffmpeg if librosa has trouble reading MP3s.
