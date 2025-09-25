import requests
import logging
from typing import Dict
from config import settings

logger = logging.getLogger(__name__)

class WeatherAPI:
    def __init__(self):
        self.api_key = settings.WEATHER_API_KEY
        self.base_url = "http://api.openweathermap.org/data/2.5"
    
    async def get_weather(self, location: str) -> Dict:
        """Get weather data for location"""
        try:
            if not self.api_key:
                return self._get_mock_weather()
            
            # In production, make actual API call
            # For now, return mock data
            return self._get_mock_weather()
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return self._get_mock_weather()
    
    def _get_mock_weather(self) -> Dict:
        """Return mock weather data"""
        import random
        return {
            'impact': random.uniform(1, 7),  # Weather impact scale 1-10
            'description': 'Clear sky'
        }
