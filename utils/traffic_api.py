import requests
import logging
from typing import Dict
from datetime import datetime
from config import settings

logger = logging.getLogger(__name__)

class TrafficAPI:
    def __init__(self):
        self.api_key = settings.GOOGLE_MAPS_API_KEY
    
    async def get_traffic_density(self, location: str) -> Dict:
        """Get traffic density for location"""
        try:
            if not self.api_key:
                return self._get_mock_traffic()
            
            # In production, make actual API call to Google Maps
            # For now, return mock data based on time
            return self._get_mock_traffic()
        except Exception as e:
            logger.error(f"Traffic API error: {e}")
            return self._get_mock_traffic()
    
    def _get_mock_traffic(self) -> Dict:
        """Return mock traffic data based on current time"""
        import random
        current_hour = datetime.now().hour
        
        # Simulate rush hour traffic
        base_density = 4
        if current_hour in [8, 9, 17, 18, 19]:  # Rush hours
            base_density = 8
        elif current_hour in [12, 13]:  # Lunch hour
            base_density = 6
        
        density = base_density + random.uniform(-1, 2)
        
        return {
            'density': max(1, min(10, density)),
            'description': 'Moderate traffic'
        }
