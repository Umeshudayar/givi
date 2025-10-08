import os
import sys
import logging
import argparse
import warnings

# Suppress TensorFlow and Protobuf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_project():
    """Initial project setup"""
    logger.info("Setting up Givi project...")
    
    # Create necessary directories
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    
    # Generate sample data if not exists
    data_file = "data/processed/delivery_data.csv"
    if not os.path.exists(data_file):
        logger.info("Generating sample data...")
        os.system("python data/sample_data.py")
    
    # Train model if not exists
    model_file = "saved_models/givi_lstm_model.h5"
    if not os.path.exists(model_file):
        logger.info("Training LSTM model...")
        os.system("python train_model.py")
    
    logger.info("Setup complete!")

def run_app():
    """Run the FastAPI application with proper reload configuration"""
    logger.info("Starting Givi Smart Delivery Predictions...")
    
    # Use uvicorn directly with reload exclusions
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=[
            "venv/*",
            "env/*",
            ".venv/*",
            "*/site-packages/*",
            "*/__pycache__/*",
            ".git/*",
            "*.pyc",
            "*.pyo",
            "*.log",
            "data/*",
            "saved_models/*",
            "run.py",  # Exclude this launcher script
        ],
        reload_includes=["*.py"],  # Only watch Python files
        reload_dirs=["api", "models"],  # Only watch specific code directories (exclude root)
        log_level="info"
    )

def run_tests():
    """Run test suite"""
    logger.info("Running tests...")
    os.system("pytest tests/ -v")

def main():
    parser = argparse.ArgumentParser(description="Givi Smart Delivery Predictions")
    parser.add_argument("--setup", action="store_true", help="Setup project")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--no-reload", action="store_true", help="Run without auto-reload")
    
    args = parser.parse_args()
    
    if args.setup:
        setup_project()
    elif args.test:
        run_tests()
    elif args.train:
        os.system("python train_model.py")
    else:
        # Default: setup and run
        setup_project()
        
        if args.no_reload:
            # Run without reload
            logger.info("Starting Givi Smart Delivery Predictions (no auto-reload)...")
            import uvicorn
            uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
        else:
            run_app()

if __name__ == "__main__":
    main()