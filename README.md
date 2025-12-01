# HarmonyAI (Group 27)

Genre classification & playlist grouping using GTZAN + Random Forest.

## Quickstart

You will have to download GTZAN data set and place the genre folders that contain their .wav files into the data\raw\gtzan. After downloading the dataset. Please follow the instructions below to set up the virtual enviroment. You can run the pipeline or jump straight into the main model via running: python -m harmony_ai.ui_gradio It may take up to 500 seconds to classify and sort the first set of inputs. After that it takes about 20 seconds.

GTZAN data link: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
Put GTZAN here (genre folders):
data/raw/gtzan/blues/*.wav, classical/*.wav, ...

```bash
# 1) Create & activate venv
python -m venv .venv
# Windows
. .venv/Scripts/Activate.ps1
# macOS/Linux
source .venv/bin/activate

# 2) Install (dev + api extras optional)
pip install -e ".[dev,api]"

# 3) Run pipeline
harmonyai splits --config configs/rf.yaml
harmonyai features --config configs/rf.yaml
harmonyai train --config configs/rf.yaml
harmonyai eval --config configs/rf.yaml

# run API after training
python -m harmony_ai.ui_gradio
```

## Layout

- `src/harmony_ai/` — reusable library code
- `scripts/` — thin CLI wrappers 
- `data/` — raw dataset & split CSVs
- `features/` — feature CSVs
- `models/` — trained models
- `reports/` — metrics, plots
- `configs/` — YAML configs

## Notes
- Requires Python 3.10–3.12.
- Install ffmpeg if librosa has trouble reading MP3s.
