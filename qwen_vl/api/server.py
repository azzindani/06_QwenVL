"""
FastAPI server with proper model initialization and lifecycle management.

This module provides the main API server with:
- Model initialization at startup
- Graceful shutdown handling
- CORS configuration for UI communication
- Logging throughout using qwen_vl.utils.logger
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import get_config, load_config
from ..core.model_loader import ModelLoader
from ..utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

# Global model loader instance (shared across endpoints)
_model_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get the global model loader instance."""
    global _model_loader
    if _model_loader is None:
        raise RuntimeError("Model not initialized. Server not started properly.")
    return _model_loader


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    
    Handles:
    - Model loading at startup
    - Cleanup at shutdown
    """
    global _model_loader
    
    logger.info("Starting Qwen VL API server...")
    
    # Load configuration
    config = load_config()
    
    # Setup logging
    setup_logging(
        level=config.logging.level,
        format_type=config.logging.format,
        file_path=config.logging.file_path,
    )
    
    # Initialize model loader
    logger.info(f"Loading model: {config.model.model_id}")
    _model_loader = ModelLoader()
    
    try:
        _model_loader.load(config)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield  # Server is running
    
    # Cleanup
    logger.info("Shutting down Qwen VL API server...")
    if _model_loader:
        _model_loader.unload()
        _model_loader = None
    logger.info("Server shutdown complete")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="Qwen VL Document Processing API",
        description="Vision-Language model API for document extraction and analysis",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # Configure CORS for UI communication
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Import and include routers
    from .endpoints import router
    app.include_router(router)
    
    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
):
    """
    Run the API server.
    
    Args:
        host: Host to bind to
        port: Port to listen on
        reload: Enable auto-reload for development
    """
    logger.info(f"Starting server on {host}:{port}")
    
    app = create_app()
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    run_server()
