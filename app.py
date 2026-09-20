import importlib
import pickle
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request

sys.modules.setdefault(
    "telco_churn_preprocessing",
    importlib.import_module("notebook.telco_churn_preprocessing"),
)

from notebook.telco_churn_preprocessing import TelcoChurnPreprocessor  # required to unpickle the pipeline

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "churn_model.pkl"

with MODEL_PATH.open("rb") as model_file:
    pipeline = pickle.load(model_file)

app = FastAPI()


@app.post("/predict")
async def predict(request: Request):
    customer = await request.json()
    frame = pd.DataFrame([customer])
    label = int(pipeline.predict(frame)[0])
    probability = float(pipeline.predict_proba(frame)[:, 1][0])
    return {
        "prediction": "Yes" if label == 1 else "No",
        "churn_probability": round(probability, 2),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8001)
