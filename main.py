print("🚀 THIS IS THE CORRECT main.py FILE")

import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# Load the trained model
model = joblib.load("best_rf_model.pkl")

app = FastAPI(title="Electricity Cost Predictor")
@app.get("/")
def home():
    return {"message": "THIS IS THE REAL APP"}
# Define input schema
class ElectricityInput(BaseModel):
    site_area: int
    structure_type: str  # One of: Commercial, Industrial, Mixed-use, Residential
    water_consumption: float
    recycling_rate: int
    utilisation_rate: int
    air_qality_index: int
    issue_reolution_time: int
    resident_count: int

# Encoding function
def encode_structure_type(structure_type: str):
    if structure_type == "Industrial":
        return [1, 0, 0]
    elif structure_type == "Mixed-use":
        return [0, 1, 0]
    elif structure_type == "Residential":
        return [0, 0, 1]
    else:  # Commercial (baseline)
        return [0, 0, 0]

# Prediction endpoint
@app.post("/predict")
def predict_cost(input_data: ElectricityInput):
    # Encode categorical feature
    structure_encoding = encode_structure_type(input_data.structure_type)

    # Form the feature vector (ordered to match training)
    features = [
        input_data.site_area,
        input_data.water_consumption,
        input_data.recycling_rate,
        input_data.utilisation_rate,
        input_data.air_qality_index,
        input_data.issue_reolution_time,
        input_data.resident_count,
    ] + structure_encoding  # 3 encoded features

    prediction = model.predict([features])
    return {"predicted_electricity_cost": round(prediction[0], 2)}
