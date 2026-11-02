from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import sys
import os
import logging
import traceback

# Ensure the root directory is in the sys path for absolute imports
HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
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


# ---------------------------------------------------------------------------
# Static file lookup (works locally and on Vercel, wherever the files live)
# ---------------------------------------------------------------------------
SEARCH_ROOTS = [HERE, HERE.parent, HERE.parent.parent, Path.cwd()]
SEARCH_SUBDIRS = ["static", "static/public", "public", ""]


def find_static(name: str) -> Path:
    for root in SEARCH_ROOTS:
        for sub in SEARCH_SUBDIRS:
            candidate = root / sub / name
            if candidate.is_file():
                return candidate
    raise HTTPException(status_code=404, detail=f"{name} not found")


class PredictionRequest(BaseModel):
    dti: float
    credit_utilization: float
    emi_to_income: float
    loan_to_income: float
    active_loan_count: int
    delinquency_count: int


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


# Keep the /static mount working, but only if the folder exists
for _root in SEARCH_ROOTS:
    _static_dir = _root / "static"
    if _static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
        break


@app.get("/styles.css")
async def serve_css():
    return FileResponse(find_static("styles.css"), media_type="text/css")


@app.get("/script.js")
async def serve_js():
    return FileResponse(find_static("script.js"), media_type="application/javascript")


@app.get("/")
async def serve_frontend():
    return FileResponse(find_static("index.html"), media_type="text/html")
