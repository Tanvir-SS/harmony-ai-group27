from fastapi import FastAPI, UploadFile
from harmony_ai.config import load_config
from harmony_ai.model import load_model
from harmony_ai.features import extract_fixed_feature_vector

import tempfile

app = FastAPI(title="HarmonyAI API")
CFG = load_config("configs/baseline.yaml")
MODEL = load_model(CFG["paths"]["model_path"])

@app.post("/predict")
async def predict(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=True, suffix=file.filename) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        feat = extract_fixed_feature_vector(tmp.name, CFG["features"])
    pred = MODEL.predict([feat])[0]
    return {"genre": str(pred)}
