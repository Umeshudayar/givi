import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import logging
from config import settings

logger = logging.getLogger(__name__)

class LSTMDeliveryModel:
    def __init__(self):
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.feature_columns = [
            'hour', 'day_of_week', 'distance', 'weather_impact',
            'traffic_density', 'restaurant_busy', 'prep_time_base', 'order_complexity'
        ]
        
    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=True),
            Dropout(0.2),
            LSTM(16, return_sequences=False),
            Dense(8, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mae',
            metrics=['mse', 'mae']
        )
        
        return model
    
    def prepare_sequences(self, X, y, sequence_length=None):
        """Prepare data sequences for LSTM"""
        if sequence_length is None:
            sequence_length = settings.SEQUENCE_LENGTH
            
        X_sequences, y_sequences = [], []
        
        for i in range(len(X) - sequence_length):
            X_sequences.append(X[i:(i + sequence_length)])
            y_sequences.append(y[i + sequence_length])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train(self, df):
        """Train the LSTM model"""
        logger.info("Starting model training...")
        
        # Prepare features and target
        X = self.scaler_X.fit_transform(df[self.feature_columns])
        y = self.scaler_y.fit_transform(df[['delivery_time']])
        
        # Create sequences
        X_sequences, y_sequences = self.prepare_sequences(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_sequences, y_sequences, test_size=0.2, random_state=42
        )
        
        # Build and train model
        self.model = self.build_model((X_sequences.shape[1], X_sequences.shape[2]))
        
        history = self.model.fit(
            X_train, y_train,
            epochs=settings.EPOCHS,
            batch_size=settings.BATCH_SIZE,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluate
        test_loss, test_mse, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test MAE: {test_mae:.2f} minutes")
        
        return history
    
    def save_model(self):
        """Save model and scalers"""
        self.model.save(settings.MODEL_PATH)
        joblib.dump(self.scaler_X, settings.SCALER_X_PATH)
        joblib.dump(self.scaler_y, settings.SCALER_Y_PATH)
        joblib.dump(self.feature_columns, settings.FEATURES_PATH)
        logger.info("Model saved successfully")
    
    def load_model(self):
        """Load model and scalers"""
        self.model = tf.keras.models.load_model(settings.MODEL_PATH)
        self.scaler_X = joblib.load(settings.SCALER_X_PATH)
        self.scaler_y = joblib.load(settings.SCALER_Y_PATH)
        self.feature_columns = joblib.load(settings.FEATURES_PATH)
        logger.info("Model loaded successfully")
