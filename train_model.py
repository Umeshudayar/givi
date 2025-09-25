#!/usr/bin/env python3
"""
Train the LSTM model for delivery time prediction
"""

import pandas as pd
import os
from models.lstm_model import LSTMDeliveryModel
from utils.helpers import generate_sample_data
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting LSTM model training...")
    
    # Check if data exists, if not generate it
    data_path = "data/processed/delivery_data.csv"
    if os.path.exists(data_path):
        logger.info("Loading existing data...")
        df = pd.read_csv(data_path)
    else:
        logger.info("Generating sample data...")
        df = generate_sample_data(5000)
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv(data_path, index=False)
    
    # Initialize and train model
    model = LSTMDeliveryModel()
    history = model.train(df)
    
    # Save model
    os.makedirs("saved_models", exist_ok=True)
    model.save_model()
    
    logger.info("Model training completed successfully!")
    logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    logger.info(f"Final validation MAE: {history.history['val_mae'][-1]:.2f} minutes")

if __name__ == "__main__":
    main()
