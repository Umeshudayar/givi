from pydantic import BaseModel, Field
from typing import List, Optional

class PredictionRequest(BaseModel):
    restaurant: str = Field(..., min_length=1, max_length=100)
    location: str = Field(..., min_length=1, max_length=200)
    orderTime: str = Field(..., description="Time in HH:MM format")
    orderType: str = Field(default="standard", pattern="^(standard|priority|scheduled)$")

class Factor(BaseModel):
    name: str
    impact: str  # "high", "medium", "low"
    description: str

class PredictionResponse(BaseModel):
    estimatedTime: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    factors: List[Factor]
    requestId: Optional[str] = None
