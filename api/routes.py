from fastapi import APIRouter, HTTPException, Depends
from .schemas import PredictionRequest, PredictionResponse
from models.predictor import DeliveryPredictor
import logging
import uuid

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize predictor (singleton)
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        try:
            predictor = DeliveryPredictor()
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {e}")
            raise HTTPException(status_code=500, detail="Model initialization failed")
    return predictor

@router.post("/predict", response_model=PredictionResponse)
async def predict_delivery_time(
    request: PredictionRequest,
    predictor: DeliveryPredictor = Depends(get_predictor)
):
    """Generate delivery time prediction"""
    try:
        request_id = str(uuid.uuid4())[:8]
        logger.info(f"Processing prediction request {request_id}")
        
        result = await predictor.predict(request)
        result.requestId = request_id
        
        logger.info(f"Prediction completed: {result.estimatedTime} minutes (confidence: {result.confidence})")
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@router.get("/model/info")
async def get_model_info(predictor:  = Depends(get_predictor)):
    """Get model information and statistics"""
    return predictor.get_model_info()
