from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import logging

# Ensure the root directory is in the sys path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import predict_risk

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="CredifyAI API", description="Credit Risk Prediction Backend")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    dti: float
    credit_utilization: float
    emi_to_income: float
    loan_to_income: float
    active_loan_count: int
    delinquency_count: int

import traceback

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        logger.info(f"Prediction request received: {request.dict()}")
        input_data = request.dict()
        result = predict_risk(input_data)
        logger.info(f"Prediction successful: {result}")
        return {"status": "success", "data": result}
    except Exception as e:
        err = traceback.format_exc()
        logger.error(f"Prediction failed: {err}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files correctly
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")
