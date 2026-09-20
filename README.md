# Telco Customer Churn Prediction API

This project builds and serves a machine learning model that predicts whether a Telco customer is likely to churn.

## Overview

- Predicts churn using a saved scikit-learn pipeline
- Exposes a REST API with FastAPI
- Uses a preprocessing pipeline consistent with the training notebook
- Includes sample input data for quick testing

## Project structure

- `app.py` — FastAPI application and prediction endpoint
- `requirements.txt` — Python dependencies
- `sample_request.json` — example payload for a churn prediction request
- `data/` — raw dataset and data dictionary
- `model/` — trained model artifacts
- `notebook/` — preprocessing logic and analysis notebook

## Setup

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run the API

Start the server:

```bash
python app.py
```

The app runs on:

```text
http://0.0.0.0:8001
```

## Prediction endpoint

### Request

```http
POST /predict
Content-Type: application/json
```

Example body:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}
```

### Response

```json
{
  "prediction": "Yes",
  "churn_probability": 0.76
}
```

## Test with curl

```bash
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

## Notes

- The preprocessing logic is defined in `notebook/telco_churn_preprocessing.py` and is used to align incoming records with the trained model schema.
- The model expects the same feature columns as the original Telco customer dataset.
