from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sys
import os

import logging

# Add parent directory to path to import your scripts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts import call_model
from scripts import prepare_data
import pandas as pd
import json

print("Making FastAPI app...")
app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (if you have CSS/JS files)
if os.path.exists("public"):
    print("Mounting file...")
    app.mount("/static", StaticFiles(directory="public"), name="static")

class PredictionRequest(BaseModel):
    make: str
    model: str
    variant: str
    year: int
    mileage: int
    province: str

class PredictionResponse(BaseModel):
    predicted_price: float
    formatted_price: str

# Serve the index.html at root
@app.get("/")
def read_root():
    print("===========================================")
    print(f"Loading index.html file...")
    print("===========================================\n\n")
    return FileResponse('public/index.html')

# API endpoint to get data (make/model/cities)
@app.get("/api/data")
async def get_data():
    print("===========================================")

    """Return the make/model/variants and cities data"""
    
    # Load make_model_variants
    print(f"Opening file to get variants...")
    with open("data/make_model_variants.json", "r") as f:
        make_model_variants = json.load(f)
    print(f"Variants loaded\n")


    # Load cities from prepared data
    print(f"Loading df...")
    df, _ = prepare_data.prepare_data(pd.DataFrame(), use_old=True)
    print(f"DF loaded\n")
    
    print(f"Setting provinces...")
    provinces = set()
    for province in list(df["province"].unique()):
        if not isinstance(province, float):
            provinces.add(province)
    print(f"Provinces loaded.\n")

    print("===========================================\n\n")
    return {
        "make_model_variants": make_model_variants,
        "provinces": sorted(list(provinces))
    }

# API endpoint for predictions
@app.post("/api/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    

    print("Trying to make a Prediction...")
    try:
        input_data = {
            "title": f"{request.make} {request.model}",
            "variant": request.variant,
            "year": request.year,
            "mileage": request.mileage,
            "province": request.province
        }

        print(f"Got input data: {input_data}")

        prediction = call_model.get_prediction(input_data)
        
        return PredictionResponse(
            predicted_price=float(prediction),
            formatted_price=f"R{prediction:,.2f}"
        )
    except Exception as e:
        print(e)

        return {
            "Unable to find a price. Please try a different variant."
        }

        raise HTTPException(status_code=500, detail=str(e))
        