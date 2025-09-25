import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from datetime import datetime
import logging
from typing import List
import os

from api.schemas import PredictionRequest, PredictionResponse, Factor
from config import settings
from utils.weather_api import WeatherAPI
from utils.traffic_api import TrafficAPI

logger = logging.getLogger(__name__)

class DeliveryPredictor:
    def __init__(self):
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.feature_columns = None
        self.weather_api = WeatherAPI()
        self.traffic_api = TrafficAPI()
        self._load_model()
    
    def _load_model(self):
        """Load trained model and scalers"""
        try:
            if os.path.exists(settings.MODEL_PATH):
                self.model = tf.keras.models.load_model(settings.MODEL_PATH)
                self.scaler_X = joblib.load(settings.SCALER_X_PATH)
                self.scaler_y = joblib.load(settings.SCALER_Y_PATH)
                self.feature_columns = joblib.load(settings.FEATURES_PATH)
                logger.info("Model loaded successfully")
            else:
                logger.warning("No trained model found. Please train model first.")
                # For demo purposes, we'll use a simple fallback
                self._create_dummy_model()
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create dummy model for demo purposes"""
        logger.info("Creating dummy model for demo")
        self.feature_columns = [
            'hour', 'day_of_week', 'distance', 'weather_impact',
            'traffic_density', 'restaurant_busy', 'prep_time_base', 'order_complexity'
        ]
    
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Generate delivery time prediction"""
        
        # Extract features from request
        features = await self._extract_features(request)
        
        # Make prediction
        if self.model is not None:
            estimated_time = self._predict_with_model(features)
        else:
            estimated_time = self._predict_fallback(features)
        
        # Calculate confidence
        confidence = self._calculate_confidence(features)
        
        # Generate factor analysis
        factors = self._analyze_factors(features)
        
        return PredictionResponse(
            estimatedTime=int(estimated_time),
            confidence=round(confidence, 2),
            factors=factors
        )
    
    def _predict_with_model(self, features: dict) -> float:
        """Make LSTM prediction"""
        # Create feature vector
        feature_vector = np.array([[
            features['hour'], features['day_of_week'], features['distance'],
            features['weather_impact'], features['traffic_density'],
            features['restaurant_busy'], features['prep_time_base'],
            features['order_complexity']
        ]])
        
        # Scale features
        features_scaled = self.scaler_X.transform(feature_vector)
        
        # Create sequence (repeat for LSTM)
        sequence = np.tile(features_scaled, (settings.SEQUENCE_LENGTH, 1)).reshape(1, settings.SEQUENCE_LENGTH, -1)
        
        # Predict
        prediction_scaled = self.model.predict(sequence, verbose=0)
        prediction = self.scaler_y.inverse_transform(prediction_scaled)[0][0]
        
        return max(12, min(90, prediction))
    
    def _predict_fallback(self, features: dict) -> float:
        """Fallback prediction method"""
        base_time = features['prep_time_base'] + (features['distance'] * 2.5)
        
        # Add delays based on conditions
        weather_delay = features['weather_impact'] * 0.8
        traffic_delay = (features['traffic_density'] / 10) * features['distance'] * 1.5
        busy_delay = (features['restaurant_busy'] / 10) * 3
        complexity_delay = features['order_complexity'] * 0.5
        
        total_time = base_time + weather_delay + traffic_delay + busy_delay + complexity_delay
        
        return max(15, min(75, total_time))
    
    async def _extract_features(self, request: PredictionRequest) -> dict:
        """Extract features from request"""
        # Parse time
        try:
            order_hour = int(request.orderTime.split(':')[0])
        except:
            order_hour = datetime.now().hour
        
        # Get current date info
        day_of_week = datetime.now().weekday()
        
        # Get real-time data
        weather_data = await self.weather_api.get_weather(request.location)
        traffic_data = await self.traffic_api.get_traffic_density(request.location)
        
        # Estimate other features
        distance = self._estimate_distance(request.location)
        restaurant_busy = self._estimate_restaurant_busy(order_hour)
        prep_time_base = self._estimate_prep_time(request.restaurant)
        order_complexity = {'standard': 2, 'priority': 3, 'scheduled': 1}[request.orderType]
        
        return {
            'hour': order_hour,
            'day_of_week': day_of_week,
            'distance': distance,
            'weather_impact': weather_data.get('impact', 3.0),
            'traffic_density': traffic_data.get('density', 5.0),
            'restaurant_busy': restaurant_busy,
            'prep_time_base': prep_time_base,
            'order_complexity': order_complexity
        }
    
    def _calculate_confidence(self, features: dict) -> float:
        """Calculate prediction confidence"""
        confidence = 0.85
        
        if features['weather_impact'] > 7:
            confidence -= 0.15
        if features['traffic_density'] > 8:
            confidence -= 0.10
        if features['restaurant_busy'] > 8:
            confidence -= 0.08
        
        return max(0.6, min(0.98, confidence))
    
    def _analyze_factors(self, features: dict) -> List[Factor]:
        """Analyze factors affecting delivery time"""
        factors = []
        
        if features['traffic_density'] > 7:
            factors.append(Factor(
                name="Traffic Conditions",
                impact="high" if features['traffic_density'] > 8.5 else "medium",
                description=f"Current traffic is adding approximately {int(features['traffic_density']/2)} minutes to delivery time."
            ))
        
        if features['weather_impact'] > 5:
            factors.append(Factor(
                name="Weather Impact",
                impact="high" if features['weather_impact'] > 7 else "medium",
                description="Weather conditions may affect delivery speed."
            ))
        
        if features['restaurant_busy'] > 7:
            factors.append(Factor(
                name="Restaurant Load",
                impact="medium",
                description="Restaurant is experiencing high order volume."
            ))
        
        if not factors:
            factors.append(Factor(
                name="Optimal Conditions",
                impact="low",
                description="Favorable conditions for timely delivery."
            ))
        
        return factors
    
    def get_model_info(self) -> dict:
        """Return model information"""
        return {
            "model_type": "LSTM",
            "sequence_length": settings.SEQUENCE_LENGTH,
            "features": self.feature_columns,
            "status": "loaded" if self.model else "fallback"
        }
    
    def _estimate_distance(self, location: str) -> float:
        """Estimate distance based on location"""
        location_distances = {
            'downtown': 2.5, 'midtown': 4.0, 'uptown': 6.5,
            'east': 3.5, 'west': 4.5, 'north': 5.0, 'south': 3.0
        }
        for area, dist in location_distances.items():
            if area.lower() in location.lower():
                return dist + np.random.uniform(-0.5, 0.5)
        return np.random.uniform(2.0, 8.0)
    
    def _estimate_restaurant_busy(self, hour: int) -> float:
        """Estimate restaurant busy level"""
        base_busy = 5.0
        if hour in [12, 13, 18, 19, 20]:  # Peak hours
            base_busy += np.random.uniform(2, 4)
        return base_busy
    
    def _estimate_prep_time(self, restaurant: str) -> float:
        """Estimate preparation time"""
        return np.random.uniform(10, 20)
