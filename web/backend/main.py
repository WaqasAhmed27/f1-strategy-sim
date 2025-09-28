"""
FastAPI Backend for F1 Real-Time Prediction System
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add the parent directory to the path to import f1sim
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from f1sim.features.builder import build_features_from_dir
from f1sim.models.ensemble import EnsemblePredictor
from f1sim.models.advanced import XGBoostRegressor, CatBoostRegressorWrapper
from f1sim.models.baseline import PositionRegressor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="F1 Real-Time Prediction API",
    description="Real-time Formula 1 race prediction system",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
active_connections: List[WebSocket] = []
ensemble_predictor: Optional[EnsemblePredictor] = None

# Pydantic models
class PredictionRequest(BaseModel):
    season: int
    race_round: int
    session: str = "R"

class PredictionResponse(BaseModel):
    predictions: List[Dict]
    timestamp: datetime
    model_info: Dict

class LiveUpdate(BaseModel):
    type: str  # "prediction", "status", "error"
    data: Dict
    timestamp: datetime

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                self.disconnect(connection)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize the ensemble predictor on startup."""
    global ensemble_predictor
    try:
        logger.info("Starting F1 Prediction System...")
        ensemble_predictor = EnsemblePredictor()
        await ensemble_predictor.load_models()
        logger.info("Ensemble predictor initialized successfully")
        logger.info("F1 Prediction System ready!")
    except Exception as e:
        logger.error(f"Failed to initialize ensemble predictor: {e}")
        # Don't fail startup if models can't be loaded
        ensemble_predictor = None

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "F1 Real-Time Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "websocket": "/ws",
            "health": "/health",
            "models": "/models/info"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "models_loaded": ensemble_predictor is not None,
            "version": "1.0.0",
            "service": "f1-prediction-api"
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models."""
    if ensemble_predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "models": ensemble_predictor.get_model_info(),
        "ensemble_weights": ensemble_predictor.get_weights(),
        "last_updated": datetime.now()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_race(request: PredictionRequest):
    """Make race predictions using ensemble model."""
    if ensemble_predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Build features for the race
        race_dir = Path(f"data/{request.season}_{request.race_round}_{request.session}")
        if not race_dir.exists():
            raise HTTPException(status_code=404, detail=f"Race data not found: {race_dir}")
        
        X, y, meta = build_features_from_dir(race_dir)
        
        # Make predictions
        predictions = ensemble_predictor.predict(X, meta)
        
        # Format response
        prediction_data = []
        for _, row in predictions.iterrows():
            prediction_data.append({
                "driver_number": row["DriverNumber"],
                "predicted_position": int(row["pred_final_pos"]),
                "confidence": float(row.get("confidence", 0.0)),
                "gap_to_winner": float(row.get("est_gap_to_winner_s", 0.0)),
                "driver_name": row.get("Abbreviation", f"Driver {row['DriverNumber']}"),
                "team": row.get("TeamName", "Unknown")
            })
        
        # Sort by predicted position
        prediction_data.sort(key=lambda x: x["predicted_position"])
        
        return PredictionResponse(
            predictions=prediction_data,
            timestamp=datetime.now(),
            model_info={
                "model_type": "ensemble",
                "models_used": ensemble_predictor.get_model_info(),
                "feature_count": len(X.columns)
            }
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}))
            elif message.get("type") == "subscribe":
                # Handle subscription to specific race updates
                await websocket.send_text(json.dumps({
                    "type": "subscribed",
                    "race": message.get("race"),
                    "timestamp": datetime.now().isoformat()
                }))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.post("/live/predict/{season}/{race_round}")
async def trigger_live_prediction(season: int, race_round: int, session: str = "R"):
    """Trigger live prediction and broadcast to WebSocket clients."""
    try:
        # Make prediction
        request = PredictionRequest(season=season, race_round=race_round, session=session)
        prediction_response = await predict_race(request)
        
        # Broadcast to WebSocket clients
        live_update = LiveUpdate(
            type="prediction",
            data={
                "race": f"{season}_{race_round}_{session}",
                "predictions": prediction_response.predictions,
                "model_info": prediction_response.model_info
            },
            timestamp=datetime.now()
        )
        
        await manager.broadcast(live_update.dict())
        
        return {"message": "Live prediction broadcasted", "timestamp": datetime.now()}
        
    except Exception as e:
        logger.error(f"Live prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Use PORT environment variable for Railway deployment
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

