import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Configuration
    API_HOST: str = "0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True
    
    # Model Configuration
    MODEL_PATH: str = "saved_models/givi_lstm_model.h5"
    SCALER_X_PATH: str = "saved_models/scaler_X.pkl"
    SCALER_Y_PATH: str = "saved_models/scaler_y.pkl"
    FEATURES_PATH: str = "saved_models/feature_columns.pkl"
    
    # Data Configuration
    SEQUENCE_LENGTH: int = 10
    BATCH_SIZE: int = 32
    EPOCHS: int = 50
    
    # External APIs
    WEATHER_API_KEY: str = ""
    GOOGLE_MAPS_API_KEY: str = ""
    
    class Config:
        env_file = ".env"

settings = Settings()
