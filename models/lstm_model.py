import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Concatenate, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import joblib
import logging
from datetime import datetime
import requests
from typing import Dict, Tuple, Optional
from config import settings

logger = logging.getLogger(__name__)

class GeolocationLSTMModel:
    def __init__(self):
        self.model = None
        self.scaler_features = MinMaxScaler()
        self.scaler_target = MinMaxScaler()
        self.restaurant_encoder = LabelEncoder()
        self.geolocator = Nominatim(user_agent="delivery_predictor")
        
        # Cache for geocoded locations
        self.location_cache = {}
        
        # Feature columns (will be computed)
        self.numerical_features = [
            'distance_km', 'hour', 'day_of_week', 'is_weekend',
            'is_rush_hour', 'traffic_multiplier', 'weather_impact'
        ]
        
        self.sequence_length = getattr(settings, 'SEQUENCE_LENGTH', 5)
        
    def geocode_location(self, location_name: str, retry: int = 3) -> Optional[Tuple[float, float]]:
        """
        Geocode a location name to coordinates with caching and retry logic.
        
        Args:
            location_name: Name of the location (address, city, landmark)
            retry: Number of retry attempts
            
        Returns:
            Tuple of (latitude, longitude) or None if failed
        """
        # Check cache first
        cache_key = location_name.lower().strip()
        if cache_key in self.location_cache:
            return self.location_cache[cache_key]
        
        for attempt in range(retry):
            try:
                location = self.geolocator.geocode(location_name, timeout=10)
                if location:
                    coords = (location.latitude, location.longitude)
                    self.location_cache[cache_key] = coords
                    logger.info(f"Geocoded '{location_name}' -> {coords}")
                    return coords
                else:
                    logger.warning(f"Could not geocode '{location_name}'")
                    return None
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                logger.warning(f"Geocoding attempt {attempt + 1} failed: {e}")
                if attempt == retry - 1:
                    return None
        
        return None
    
    def calculate_distance(self, restaurant_location: str, delivery_location: str) -> Optional[float]:
        """
        Calculate distance between restaurant and delivery location.
        
        Args:
            restaurant_location: Restaurant address/name
            delivery_location: Delivery address/name
            
        Returns:
            Distance in kilometers or None if geocoding failed
        """
        restaurant_coords = self.geocode_location(restaurant_location)
        delivery_coords = self.geocode_location(delivery_location)
        
        if not restaurant_coords or not delivery_coords:
            logger.error(f"Failed to geocode: {restaurant_location} or {delivery_location}")
            return None
        
        # Calculate geodesic distance
        distance = geodesic(restaurant_coords, delivery_coords).kilometers
        logger.info(f"Distance from {restaurant_location} to {delivery_location}: {distance:.2f} km")
        
        return distance
    
    def get_traffic_data(self, distance_km: float, hour: int) -> float:
        """
        Estimate traffic multiplier based on distance and time.
        More sophisticated implementations could use Google Maps API or TomTom API.
        
        Args:
            distance_km: Distance in kilometers
            hour: Hour of day (0-23)
            
        Returns:
            Traffic multiplier (1.0 = normal, >1.0 = heavy traffic)
        """
        # Rush hour periods (morning and evening)
        morning_rush = 7 <= hour <= 10
        evening_rush = 17 <= hour <= 20
        
        base_multiplier = 1.0
        
        if morning_rush or evening_rush:
            # Higher impact for longer distances during rush hour
            base_multiplier = 1.3 + (distance_km * 0.01)
        elif 22 <= hour or hour <= 6:
            # Night time - less traffic
            base_multiplier = 0.8
        else:
            # Normal hours
            base_multiplier = 1.0 + (distance_km * 0.005)
        
        return min(base_multiplier, 2.5)  # Cap at 2.5x
    
    def get_weather_impact(self) -> float:
        """
        Get current weather impact on delivery time.
        In production, integrate with OpenWeatherMap API or similar.
        
        Returns:
            Weather multiplier (1.0 = normal, >1.0 = bad weather)
        """
        # Placeholder - in production, call weather API
        # For now, return normal conditions
        return 1.0
    
    def extract_features(self, restaurant_location: str, delivery_location: str, 
                        timestamp: datetime = None) -> Optional[Dict]:
        """
        Extract all features needed for prediction.
        
        Args:
            restaurant_location: Restaurant address/name
            delivery_location: Delivery address/name
            timestamp: Time of order (defaults to now)
            
        Returns:
            Dictionary of features or None if geocoding failed
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate distance
        distance = self.calculate_distance(restaurant_location, delivery_location)
        if distance is None:
            return None
        
        # Time features
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        is_rush_hour = 1 if (7 <= hour <= 10) or (17 <= hour <= 20) else 0
        
        # Traffic and weather
        traffic_multiplier = self.get_traffic_data(distance, hour)
        weather_impact = self.get_weather_impact()
        
        features = {
            'restaurant_location': restaurant_location,
            'delivery_location': delivery_location,
            'distance_km': distance,
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'is_rush_hour': is_rush_hour,
            'traffic_multiplier': traffic_multiplier,
            'weather_impact': weather_impact,
            'timestamp': timestamp
        }
        
        return features
    
    def build_model(self, n_features: int, n_restaurants: int):
        """
        Build enhanced LSTM model with restaurant embeddings.
        
        Args:
            n_features: Number of numerical features
            n_restaurants: Number of unique restaurants for embedding
        """
        # Input layers
        numerical_input = Input(shape=(self.sequence_length, n_features), name='numerical_input')
        restaurant_input = Input(shape=(self.sequence_length,), name='restaurant_input')
        
        # Restaurant embedding
        restaurant_embedding = Embedding(
            input_dim=n_restaurants + 1,
            output_dim=8,
            input_length=self.sequence_length,
            name='restaurant_embedding'
        )(restaurant_input)
        
        # Flatten embedding for concatenation
        restaurant_flat = tf.keras.layers.Reshape((self.sequence_length, 8))(restaurant_embedding)
        
        # Concatenate numerical features with restaurant embedding
        combined = Concatenate(axis=-1)([numerical_input, restaurant_flat])
        
        # LSTM layers
        x = LSTM(128, return_sequences=True)(combined)
        x = Dropout(0.3)(x)
        x = LSTM(64, return_sequences=True)(x)
        x = Dropout(0.3)(x)
        x = LSTM(32, return_sequences=False)(x)
        x = Dense(16, activation='relu')(x)
        output = Dense(1, activation='linear', name='delivery_time')(x)
        
        # Create model
        model = Model(inputs=[numerical_input, restaurant_input], outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',  # More robust to outliers than MAE
            metrics=['mae', 'mse']
        )
        
        return model
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple:
        """
        Prepare training data with proper feature engineering.
        
        Args:
            df: DataFrame with columns: restaurant_location, delivery_location, 
                actual_delivery_time, timestamp
                
        Returns:
            Prepared sequences and encodings
        """
        logger.info("Preparing training data with geolocation features...")
        
        # Extract features for each row
        features_list = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            features = self.extract_features(
                row['restaurant_location'],
                row['delivery_location'],
                row['timestamp'] if 'timestamp' in row else None
            )
            
            if features is not None:
                features['actual_delivery_time'] = row['actual_delivery_time']
                features_list.append(features)
                valid_indices.append(idx)
        
        if len(features_list) == 0:
            raise ValueError("No valid features extracted. Check location data.")
        
        # Create features DataFrame
        features_df = pd.DataFrame(features_list)
        logger.info(f"Successfully extracted features for {len(features_df)}/{len(df)} samples")
        
        # Encode restaurants
        features_df['restaurant_encoded'] = self.restaurant_encoder.fit_transform(
            features_df['restaurant_location']
        )
        
        # Scale numerical features
        numerical_data = features_df[self.numerical_features].values
        scaled_numerical = self.scaler_features.fit_transform(numerical_data)
        
        # Scale target
        target_data = features_df[['actual_delivery_time']].values
        scaled_target = self.scaler_target.fit_transform(target_data)
        
        return features_df, scaled_numerical, scaled_target
    
    def create_sequences(self, numerical_data, restaurant_encoded, target_data):
        """Create sequences for LSTM training."""
        X_numerical, X_restaurant, y = [], [], []
        
        for i in range(len(numerical_data) - self.sequence_length):
            X_numerical.append(numerical_data[i:i + self.sequence_length])
            X_restaurant.append(restaurant_encoded[i:i + self.sequence_length])
            y.append(target_data[i + self.sequence_length])
        
        return (
            np.array(X_numerical),
            np.array(X_restaurant),
            np.array(y)
        )
    
    def train(self, df: pd.DataFrame, validation_split: float = 0.2):
        """
        Train the model with geolocation-based features.
        
        Args:
            df: Training data with restaurant_location, delivery_location, 
                actual_delivery_time, timestamp
            validation_split: Validation data fraction
        """
        logger.info("Starting training with geolocation features...")
        
        # Prepare data
        features_df, scaled_numerical, scaled_target = self.prepare_training_data(df)
        restaurant_encoded = features_df['restaurant_encoded'].values
        
        # Create sequences
        X_num, X_rest, y = self.create_sequences(
            scaled_numerical, restaurant_encoded, scaled_target
        )
        
        logger.info(f"Created {len(X_num)} sequences for training")
        
        # Time-based split
        split_idx = int(len(X_num) * (1 - validation_split))
        
        X_num_train, X_num_val = X_num[:split_idx], X_num[split_idx:]
        X_rest_train, X_rest_val = X_rest[:split_idx], X_rest[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Build model
        n_restaurants = len(self.restaurant_encoder.classes_)
        self.model = self.build_model(
            n_features=len(self.numerical_features),
            n_restaurants=n_restaurants
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_mae',
                patience=15,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_mae',
                factor=0.5,
                patience=7,
                min_lr=1e-6
            )
        ]
        
        # Train
        history = self.model.fit(
            [X_num_train, X_rest_train], y_train,
            validation_data=([X_num_val, X_rest_val], y_val),
            epochs=getattr(settings, 'EPOCHS', 100),
            batch_size=getattr(settings, 'BATCH_SIZE', 32),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        val_loss, val_mae, val_mse = self.model.evaluate(
            [X_num_val, X_rest_val], y_val, verbose=0
        )
        
        actual_mae = val_mae * self.scaler_target.scale_[0]
        logger.info(f"Validation MAE: {actual_mae:.2f} minutes")
        
        return history
    
    def predict(self, restaurant_location: str, delivery_location: str,
                historical_context: Optional[pd.DataFrame] = None) -> Dict:
        """
        Predict delivery time with confidence and affecting factors.
        
        Args:
            restaurant_location: Restaurant name/address
            delivery_location: Delivery address
            historical_context: Recent orders for sequence (optional)
            
        Returns:
            Dictionary with prediction, confidence, and factors
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Extract features for current order
        current_features = self.extract_features(restaurant_location, delivery_location)
        
        if current_features is None:
            return {
                'error': 'Failed to geocode locations',
                'restaurant_location': restaurant_location,
                'delivery_location': delivery_location
            }
        
        # If no historical context, create dummy sequence
        if historical_context is None or len(historical_context) < self.sequence_length:
            # Use current features repeated
            features_sequence = [current_features] * self.sequence_length
        else:
            # Use historical context
            features_sequence = []
            for _, row in historical_context.tail(self.sequence_length).iterrows():
                hist_features = self.extract_features(
                    row['restaurant_location'],
                    row['delivery_location'],
                    row.get('timestamp')
                )
                if hist_features:
                    features_sequence.append(hist_features)
        
        # Prepare input
        numerical_seq = []
        restaurant_seq = []
        
        for features in features_sequence:
            numerical_seq.append([features[col] for col in self.numerical_features])
            try:
                rest_encoded = self.restaurant_encoder.transform([features['restaurant_location']])[0]
            except ValueError:
                # Unknown restaurant, use 0
                rest_encoded = 0
            restaurant_seq.append(rest_encoded)
        
        # Scale and reshape
        numerical_seq = self.scaler_features.transform(np.array(numerical_seq))
        numerical_seq = numerical_seq.reshape(1, self.sequence_length, -1)
        restaurant_seq = np.array(restaurant_seq).reshape(1, self.sequence_length)
        
        # Predict
        prediction_scaled = self.model.predict(
            [numerical_seq, restaurant_seq], 
            verbose=0
        )
        prediction = self.scaler_target.inverse_transform(prediction_scaled)[0, 0]
        
        # Calculate confidence (inverse of prediction uncertainty)
        # Simple heuristic: higher for known restaurants, moderate distances
        confidence = 0.85
        if current_features['distance_km'] > 20:
            confidence -= 0.15
        if current_features['traffic_multiplier'] > 1.5:
            confidence -= 0.10
        if current_features['weather_impact'] > 1.2:
            confidence -= 0.10
        
        # Identify affecting factors
        factors = []
        if current_features['distance_km'] > 15:
            factors.append(f"Long distance: {current_features['distance_km']:.1f} km")
        if current_features['is_rush_hour']:
            factors.append("Rush hour traffic")
        if current_features['traffic_multiplier'] > 1.3:
            factors.append(f"Heavy traffic (×{current_features['traffic_multiplier']:.2f})")
        if current_features['weather_impact'] > 1.1:
            factors.append("Weather conditions")
        if current_features['is_weekend']:
            factors.append("Weekend (different traffic pattern)")
        
        if not factors:
            factors.append("Normal conditions")
        
        result = {
            'predicted_time_minutes': round(prediction, 1),
            'confidence_percentage': round(confidence * 100, 1),
            'distance_km': round(current_features['distance_km'], 2),
            'affecting_factors': factors,
            'traffic_multiplier': round(current_features['traffic_multiplier'], 2),
            'hour': current_features['hour'],
            'is_rush_hour': bool(current_features['is_rush_hour']),
            'restaurant_coords': self.location_cache.get(restaurant_location.lower().strip()),
            'delivery_coords': self.location_cache.get(delivery_location.lower().strip())
        }
        
        return result
    
    def save_model(self):
        """Save model and all preprocessing objects."""
        self.model.save(settings.MODEL_PATH)
        joblib.dump(self.scaler_features, settings.SCALER_X_PATH)
        joblib.dump(self.scaler_target, settings.SCALER_Y_PATH)
        joblib.dump({
            'restaurant_encoder': self.restaurant_encoder,
            'location_cache': self.location_cache,
            'numerical_features': self.numerical_features,
            'sequence_length': self.sequence_length
        }, settings.FEATURES_PATH)
        logger.info("Model and preprocessors saved")
    
    def load_model(self):
        """Load model and preprocessors."""
        self.model = tf.keras.models.load_model(settings.MODEL_PATH)
        self.scaler_features = joblib.load(settings.SCALER_X_PATH)
        self.scaler_target = joblib.load(settings.SCALER_Y_PATH)
        
        config = joblib.load(settings.FEATURES_PATH)
        self.restaurant_encoder = config['restaurant_encoder']
        self.location_cache = config['location_cache']
        self.numerical_features = config['numerical_features']
        self.sequence_length = config['sequence_length']
        
        logger.info("Model and preprocessors loaded")