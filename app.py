import pickle

import pandas as pd
from fastapi import FastAPI, Request

from telco_churn_preprocessing import TelcoChurnPreprocessor  # required to unpickle the pipeline

with open("telco_churn_model.pkl", "rb") as model_file:
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
