import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_data(n_samples=5000):
    """Generate realistic sample delivery data"""
    print(f"Generating {n_samples} sample delivery records...")
    
    data = []
    base_date = datetime(2023, 1, 1)
    
    for i in range(n_samples):
        # Generate random datetime
        random_days = random.randint(0, 365)
        order_date = base_date + timedelta(days=random_days)
        
        # Time features
        hour = random.randint(8, 23)
        day_of_week = order_date.weekday()
        
        # Location features
        distance = random.uniform(0.5, 15.0)
        
        # Weather features (0-10 scale)
        weather_impact = random.uniform(0, 8)
        
        # Traffic features
        traffic_density = random.uniform(2, 10)
        if hour in [12, 13, 18, 19, 20]:  # Peak hours
            traffic_density += random.uniform(2, 4)
        
        # Restaurant features
        restaurant_busy = random.uniform(1, 10)
        prep_time_base = random.uniform(8, 25)
        
        # Order features
        order_complexity = random.uniform(1, 5)
        
        # Calculate realistic delivery time
        base_time = prep_time_base + (distance * 2.5)
        
        # Add variations
        weather_delay = weather_impact * 0.8
        traffic_delay = (traffic_density / 10) * distance * 1.5
        busy_delay = (restaurant_busy / 10) * 3
        complexity_delay = order_complexity * 0.5
        
        # Weekend variation
        if day_of_week in [5, 6]:
            base_time += random.uniform(2, 8)
        
        total_time = base_time + weather_delay + traffic_delay + busy_delay + complexity_delay
        total_time = max(12, min(90, total_time))
        
        data.append({
            'hour': hour,
            'day_of_week': day_of_week,
            'distance': distance,
            'weather_impact': weather_impact,
            'traffic_density': traffic_density,
            'restaurant_busy': restaurant_busy,
            'prep_time_base': prep_time_base,
            'order_complexity': order_complexity,
            'delivery_time': total_time
        })
    
    return pd.DataFrame(data)
