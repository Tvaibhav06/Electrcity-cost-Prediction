print("🚀 THIS IS THE CORRECT main.py FILE!")

import joblib
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Load the trained model
model = joblib.load("best_rf_model.pkl")

app = FastAPI(title="Electricity Cost Predictor")

# Load templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("front.html", {"request": request})

# Define input schema
class ElectricityInput(BaseModel):
    site_area: int
    structure_type: str
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
    else:  # Commercial
        return [0, 0, 0]

# API POST endpoint
@app.post("/predict")
def predict_cost(input_data: ElectricityInput):
    structure_encoding = encode_structure_type(input_data.structure_type)
    features = [
        input_data.site_area,
        input_data.water_consumption,
        input_data.recycling_rate,
        input_data.utilisation_rate,
        input_data.air_qality_index,
        input_data.issue_reolution_time,
        input_data.resident_count,
    ] + structure_encoding
    prediction = model.predict([features])
    return {"predicted_electricity_cost": round(prediction[0], 2)}

# Web form submission
@app.post("/predict_form", response_class=HTMLResponse)
def predict_cost_form(
    request: Request,
    site_area: int = Form(...),
    structure_type: str = Form(...),
    water_consumption: float = Form(...),
    recycling_rate: int = Form(...),
    utilisation_rate: int = Form(...),
    air_qality_index: int = Form(...),
    issue_reolution_time: int = Form(...),
    resident_count: int = Form(...)
):
    structure_encoding = encode_structure_type(structure_type)
    features = [
        site_area,
        water_consumption,
        recycling_rate,
        utilisation_rate,
        air_qality_index,
        issue_reolution_time,
        resident_count
    ] + structure_encoding

    prediction = model.predict([features])[0]
    return templates.TemplateResponse("front.html", {
        "request": request,
        "prediction": round(prediction, 2)
    })
