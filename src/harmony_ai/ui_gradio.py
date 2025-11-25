import os
from typing import List, Optional

import gradio as gr
import pandas as pd
import json
import time

from harmony_ai.features import feature_names, extract_fixed_feature_vector
from harmony_ai.model import load_model
from harmony_ai.config import load_config


CFG_PATH = "configs/rf_fast.yaml"
cfg = load_config(CFG_PATH)

feat_cfg = cfg["features"]
feat_cols = feature_names(feat_cfg)

_FEATURE_CACHE = {}
_CFG_SIG = json.dumps(feat_cfg, sort_keys=True)

model = load_model(cfg["paths"]["model_path"])


def classify_songs(files):
    if not files:
        return pd.DataFrame(columns=["filename", "predicted_genre"])

    files = files[:10]
    rows = []
    names = []

    total_feat = 0.0

    for path in files:
        names.append(os.path.basename(path))

        t0 = time.time()
        vec = extract_fixed_feature_vector(path, feat_cfg)
        dt = time.time() - t0
        total_feat += dt
        print(f"[TIMING] features for {os.path.basename(path)}: {dt:.2f}s")

        rows.append(vec)

    X = pd.DataFrame(rows, columns=feat_cols)

    # (keep your column alignment if you added it)
    expected = getattr(model, "feature_names_in_", None)
    if expected is not None:
        X = X.reindex(columns=list(expected), fill_value=0.0)

    t1 = time.time()
    preds = model.predict(X)
    dtp = time.time() - t1
    print(f"[TIMING] predict: {dtp:.2f}s")
    print(f"[TIMING] total feature time: {total_feat:.2f}s")

    out = pd.DataFrame({"filename": names, "predicted_genre": preds})
    return out.sort_values("predicted_genre", kind="stable").reset_index(drop=True)

def main():
    with gr.Blocks(title="HarmonyAI – Genre Sorter") as demo:
        gr.Markdown("# HarmonyAI – Upload up to 10 WAVs, get them sorted by genre")

        file_input = gr.File(
            label="Upload up to 10 .wav files",
            file_count="multiple",
            file_types=[".wav"],
            type="filepath",   # returns List[str]
        )

        btn = gr.Button("Classify & Sort")
        output = gr.Dataframe(headers=["filename", "predicted_genre"])

        btn.click(classify_songs, inputs=file_input, outputs=output)

    demo.launch()


if __name__ == "__main__":
    main()
