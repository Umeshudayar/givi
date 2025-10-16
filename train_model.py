#!/usr/bin/env python3
"""
Train the LSTM model for delivery time prediction using geolocation data
"""

import pandas as pd
import os
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models.geolocation_lstm_model import GeolocationLSTMModel
from utils.data_generator import generate_realistic_training_data, save_training_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 60)
    logger.info("DELIVERY TIME PREDICTION MODEL TRAINING")
    logger.info("=" * 60)
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Check if data exists, if not generate it
    data_path = "data/training_data.csv"
    if os.path.exists(data_path):
        logger.info(f"Loading existing data from {data_path}...")
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        logger.info("Generating realistic training data...")
        df = generate_realistic_training_data(n_samples=5000)
        save_training_data(df, data_path)
        logger.info(f"Training data generated and saved to {data_path}")
    
    logger.info(f"\nDataset info:")
    logger.info(f"  - Total samples: {len(df)}")
    logger.info(f"  - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"  - Unique restaurants: {df['restaurant_location'].nunique()}")
    logger.info(f"  - Unique delivery locations: {df['delivery_location'].nunique()}")
    logger.info(f"  - Average delivery time: {df['actual_delivery_time'].mean():.1f} minutes")
    
    # Initialize and train model
    logger.info("\nInitializing model...")
    model = GeolocationLSTMModel()
    
    logger.info("\nStarting training...")
    logger.info("This may take some time as we need to geocode locations...")
    
    try:
        history = model.train(df, validation_split=0.2)
        
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        # Print final metrics
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_mae = history.history['mae'][-1]
        final_val_mae = history.history['val_mae'][-1]
        
        logger.info(f"\nFinal Training Metrics:")
        logger.info(f"  - Loss: {final_loss:.4f}")
        logger.info(f"  - MAE: {final_mae:.4f}")
        logger.info(f"\nFinal Validation Metrics:")
        logger.info(f"  - Loss: {final_val_loss:.4f}")
        logger.info(f"  - MAE: {final_val_mae:.4f}")
        
        # Save model
        logger.info("\nSaving model...")
        model.save_model()
        
        logger.info("\n" + "=" * 60)
        logger.info("MODEL SAVED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info("\nYou can now run the API server with: python app.py")
        
        # Test prediction
        logger.info("\n" + "=" * 60)
        logger.info("TESTING MODEL WITH SAMPLE PREDICTION")
        logger.info("=" * 60)
        
        test_restaurant = "Domino's Thane West, Thane, Maharashtra"
        test_delivery = "Dombivli East, Thane"
        
        logger.info(f"\nTest prediction:")
        logger.info(f"  From: {test_restaurant}")
        logger.info(f"  To: {test_delivery}")
        
        result = model.predict(test_restaurant, test_delivery)
        
        if 'error' not in result:
            logger.info(f"\n✅ Prediction Results:")
            logger.info(f"  - Predicted Time: {result['predicted_time_minutes']:.1f} minutes")
            logger.info(f"  - Distance: {result['distance_km']:.2f} km")
            logger.info(f"  - Confidence: {result['confidence_percentage']:.1f}%")
            logger.info(f"  - Traffic Multiplier: {result['traffic_multiplier']:.2f}x")
            logger.info(f"  - Affecting Factors:")
            for factor in result['affecting_factors']:
                logger.info(f"    • {factor}")
        else:
            logger.error(f"Test prediction failed: {result['error']}")
        
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())