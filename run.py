import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Delivery Time Prediction API",
    description="AI-powered delivery time prediction using geolocation and LSTM",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Import routes
from api.routes import router as api_router
app.include_router(api_router, prefix="/api", tags=["predictions"])


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    try:
        with open("templates/index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html>
            <head><title>Error</title></head>
            <body>
                <h1>Error: Web interface not found</h1>
                <p>Please ensure templates/index.html exists</p>
            </body>
        </html>
        """


@app.get("/docs-redirect")
async def docs_redirect():
    """Redirect to API documentation."""
    return {"message": "Visit /docs for API documentation"}


@app.on_event("startup")
async def startup_event():
    """Startup tasks."""
    logger.info("=" * 60)
    logger.info("DELIVERY TIME PREDICTION API STARTING")
    logger.info("=" * 60)
    logger.info(f"API Documentation: http://localhost:8000/docs")
    logger.info(f"Web Interface: http://localhost:8000")
    logger.info("=" * 60)
    
    # Check if model exists
    if not os.path.exists("saved_models/delivery_model.h5"):
        logger.warning("⚠️  No trained model found!")
        logger.warning("Please run: python train_model.py")
    else:
        logger.info("✅ Trained model found")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Delivery Time Prediction API")


if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )