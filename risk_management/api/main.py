"""
Main FastAPI application for the risk management module.
"""

import logging
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config.settings import settings
from .endpoints import router as risk_router
from ..database.models import init_db

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.logging.level),
    format=settings.logging.format,
    handlers=[
        logging.StreamHandler() if settings.logging.console_output else logging.NullHandler(),
        logging.FileHandler(settings.logging.file_path) if settings.logging.file_path else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Risk Management API",
    description="API for risk management and portfolio optimization",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(risk_router)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

# Add API key validation middleware
@app.middleware("http")
async def validate_api_key(request, call_next):
    """Validate API key middleware."""
    # Skip validation for development environment
    if settings.environment == "development":
        return await call_next(request)
    
    # Skip validation for documentation endpoints
    if request.url.path in ["/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)
    
    # Get API key from header
    api_key = request.headers.get("X-API-Key")
    
    # Validate API key
    if not api_key or api_key not in settings.api.api_keys:
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid API key"}
        )
    
    return await call_next(request)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Risk Management API",
        "version": "1.0.0",
        "description": "API for risk management and portfolio optimization"
    }

def start():
    """Start the FastAPI application."""
    import uvicorn
    uvicorn.run(
        "trading_pipeline.risk_management.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        workers=settings.api.workers,
        timeout_keep_alive=settings.api.timeout_keep_alive,
        access_log=settings.api.access_log
    )

if __name__ == "__main__":
    start()
