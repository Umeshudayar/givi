import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_realistic_training_data(n_samples=5000):
    """
    Generate realistic training data for delivery time prediction.
    Uses real city locations in India with realistic delivery patterns.
    """
    
    # Real restaurant locations in major Indian cities
    restaurants = [
        # Mumbai
        {"name": "Pizza Hut Andheri", "location": "Andheri West, Mumbai, Maharashtra"},
        {"name": "McDonald's Bandra", "location": "Bandra West, Mumbai, Maharashtra"},
        {"name": "Domino's Thane", "location": "Thane West, Thane, Maharashtra"},
        {"name": "KFC Powai", "location": "Powai, Mumbai, Maharashtra"},
        {"name": "Burger King Dadar", "location": "Dadar West, Mumbai, Maharashtra"},
        
        # Delhi
        {"name": "Haldiram's Connaught Place", "location": "Connaught Place, New Delhi"},
        {"name": "Subway Saket", "location": "Saket, New Delhi"},
        {"name": "Pizza Hut Karol Bagh", "location": "Karol Bagh, New Delhi"},
        
        # Bangalore
        {"name": "Truffles Koramangala", "location": "Koramangala, Bangalore, Karnataka"},
        {"name": "Empire Restaurant Indiranagar", "location": "Indiranagar, Bangalore, Karnataka"},
        
        # Pune
        {"name": "Vaishali Restaurant Pune", "location": "FC Road, Pune, Maharashtra"},
        {"name": "McDonald's Hinjewadi", "location": "Hinjewadi Phase 1, Pune, Maharashtra"},
        
        # Hyderabad
        {"name": "Paradise Biryani Secunderabad", "location": "Secunderabad, Hyderabad, Telangana"},
        {"name": "Cafe Bahar Himayatnagar", "location": "Himayatnagar, Hyderabad, Telangana"},
        
        # Chennai
        {"name": "Saravana Bhavan T Nagar", "location": "T Nagar, Chennai, Tamil Nadu"},
        {"name": "Anjappar Velachery", "location": "Velachery, Chennai, Tamil Nadu"}
    ]
    
    # Delivery locations (residential areas in same cities)
    delivery_locations = [
        # Mumbai area
        "Malad West, Mumbai", "Goregaon East, Mumbai", "Vikhroli, Mumbai",
        "Mulund West, Mumbai", "Ghatkopar, Mumbai", "Chembur, Mumbai",
        "Borivali West, Mumbai", "Kandivali, Mumbai", "Santacruz, Mumbai",
        "Vile Parle, Mumbai", "Jogeshwari, Mumbai", "Kurla West, Mumbai",
        
        # Thane area
        "Dombivli, Thane", "Kalyan, Thane", "Ghodbunder Road, Thane",
        "Bhiwandi, Thane", "Naupada, Thane",
        
        # Delhi area
        "Rohini, Delhi", "Dwarka, Delhi", "Lajpat Nagar, Delhi",
        "Mayur Vihar, Delhi", "Janakpuri, Delhi", "Nehru Place, Delhi",
        
        # Bangalore area
        "Whitefield, Bangalore", "Electronic City, Bangalore", "Marathahalli, Bangalore",
        "HSR Layout, Bangalore", "BTM Layout, Bangalore",
        
        # Pune area
        "Aundh, Pune", "Kothrud, Pune", "Wakad, Pune", "Viman Nagar, Pune",
        
        # Hyderabad area
        "Gachibowli, Hyderabad", "Madhapur, Hyderabad", "Kukatpally, Hyderabad",
        
        # Chennai area
        "Anna Nagar, Chennai", "Adyar, Chennai", "Porur, Chennai"
    ]
    
    data = []
    start_date = datetime.now() - timedelta(days=180)  # 6 months of data
    
    for i in range(n_samples):
        # Select random restaurant
        restaurant = random.choice(restaurants)
        
        # Select delivery location (prefer nearby locations)
        restaurant_city = restaurant['location'].split(',')[-2].strip()
        
        # 70% chance of same city, 30% nearby city
        if random.random() < 0.7:
            # Same city deliveries
            city_deliveries = [loc for loc in delivery_locations if restaurant_city in loc]
            if city_deliveries:
                delivery_loc = random.choice(city_deliveries)
            else:
                delivery_loc = random.choice(delivery_locations)
        else:
            # Cross-city or distant deliveries
            delivery_loc = random.choice(delivery_locations)
        
        # Random timestamp in last 6 months
        random_days = random.randint(0, 180)
        random_hours = random.randint(0, 23)
        random_minutes = random.randint(0, 59)
        timestamp = start_date + timedelta(
            days=random_days,
            hours=random_hours,
            minutes=random_minutes
        )
        
        # Calculate realistic delivery time based on various factors
        # This simulates the actual delivery times the model should learn
        
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = day_of_week >= 5
        is_rush_hour = (7 <= hour <= 10) or (17 <= hour <= 20)
        
        # Base time calculation (would normally use real distance)
        # For simulation: estimate based on location names
        base_time = random.uniform(15, 25)  # Base 15-25 minutes
        
        # Distance factor (simulated - in reality comes from geocoding)
        if restaurant_city not in delivery_loc:
            # Cross-city delivery - much longer
            base_time += random.uniform(30, 60)
        else:
            # Same city
            base_time += random.uniform(0, 15)
        
        # Time of day factor
        if is_rush_hour:
            base_time *= random.uniform(1.3, 1.7)
        elif 22 <= hour or hour <= 6:
            base_time *= random.uniform(0.8, 0.9)
        
        # Weekend factor
        if is_weekend and (12 <= hour <= 22):
            base_time *= random.uniform(1.1, 1.3)
        
        # Weather factor (random events)
        if random.random() < 0.15:  # 15% bad weather
            base_time *= random.uniform(1.2, 1.5)
        
        # Restaurant preparation time variation
        prep_variation = random.uniform(0.9, 1.2)
        base_time *= prep_variation
        
        # Add some noise
        noise = random.uniform(-3, 3)
        actual_delivery_time = max(10, base_time + noise)  # Minimum 10 minutes
        
        data.append({
            'restaurant_name': restaurant['name'],
            'restaurant_location': restaurant['location'],
            'delivery_location': delivery_loc,
            'timestamp': timestamp,
            'actual_delivery_time': round(actual_delivery_time, 1),
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'is_rush_hour': is_rush_hour
        })
    
    df = pd.DataFrame(data)
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Generated {n_samples} realistic training samples")
    print(f"\nSample statistics:")
    print(f"Average delivery time: {df['actual_delivery_time'].mean():.1f} minutes")
    print(f"Min delivery time: {df['actual_delivery_time'].min():.1f} minutes")
    print(f"Max delivery time: {df['actual_delivery_time'].max():.1f} minutes")
    print(f"Unique restaurants: {df['restaurant_location'].nunique()}")
    print(f"Unique delivery locations: {df['delivery_location'].nunique()}")
    
    return df

def save_training_data(df, filepath='data/training_data.csv'):
    """Save generated training data to CSV."""
    df.to_csv(filepath, index=False)
    print(f"\nTraining data saved to {filepath}")

if __name__ == "__main__":
    # Generate training data
    df = generate_realistic_training_data(n_samples=5000)
    
    # Save to file
    save_training_data(df)
    
    # Display sample
    print("\nSample data:")
    print(df.head(10))