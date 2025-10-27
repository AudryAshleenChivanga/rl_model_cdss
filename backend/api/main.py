"""FastAPI main application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.api.routers import sim, models, health
from backend.utils.config import settings
from backend.utils.logging import setup_logger

# Setup logging
setup_logger(
    log_file=f"{settings.logs_dir}/api.log",
    level=settings.log_level,
)

# Create FastAPI app
app = FastAPI(
    title="H. pylori CDSS 3D Endoscopy RL Simulator API",
    description="""
    ⚠️ **RESEARCH PROTOTYPE - NOT A MEDICAL DEVICE**
    
    This API provides endpoints for:
    - Loading 3D GI tract models
    - Running endoscopy simulations with RL policy
    - Streaming live frames and CNN predictions
    - Training and evaluation metrics
    
    **DISCLAIMER**: This system is for research and simulation only.
    It is NOT intended for clinical use or patient care.
    """,
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(models.router, prefix="/api", tags=["models"])
app.include_router(sim.router, prefix="/api", tags=["simulation"])


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with disclaimer."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>H. pylori CDSS RL Simulator API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: #f5f5f5;
            }
            .warning {
                background: #fff3cd;
                border: 2px solid #ffc107;
                border-radius: 5px;
                padding: 20px;
                margin: 20px 0;
            }
            .content {
                background: white;
                border-radius: 5px;
                padding: 20px;
                margin: 20px 0;
            }
            h1 { color: #333; }
            h2 { color: #666; }
            .warning-icon { font-size: 48px; text-align: center; }
        </style>
    </head>
    <body>
        <div class="warning">
            <div class="warning-icon">⚠️</div>
            <h2 style="text-align: center;">RESEARCH PROTOTYPE - NOT A MEDICAL DEVICE</h2>
            <p style="text-align: center;">
                This system is for research and simulation purposes ONLY.<br>
                It is NOT intended for clinical use, diagnosis, or patient care.<br>
                Never use this system to make medical decisions.
            </p>
        </div>
        
        <div class="content">
            <h1>H. pylori CDSS 3D Endoscopy RL Simulator</h1>
            
            <h2>About</h2>
            <p>
                A reinforcement learning simulator that trains an RL policy to navigate 
                a virtual upper GI tract and detect ulcer-like anomalies.
            </p>
            
            <h2>Features</h2>
            <ul>
                <li>3D endoscopy simulation with custom Gymnasium environment</li>
                <li>Synthetic lesion generation and labeling</li>
                <li>CNN-based anomaly detection (ResNet18)</li>
                <li>PPO-based navigation policy</li>
                <li>Real-time WebSocket streaming</li>
            </ul>
            
            <h2>API Documentation</h2>
            <ul>
                <li><a href="/docs">Swagger UI</a></li>
                <li><a href="/redoc">ReDoc</a></li>
            </ul>
            
            <h2>Quick Links</h2>
            <ul>
                <li><a href="/api/health">Health Check</a></li>
                <li><a href="/api/metrics">System Metrics</a></li>
            </ul>
        </div>
    </body>
    </html>
    """


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    from backend.utils.config import ensure_dirs
    ensure_dirs(settings)
    print("="*60)
    print("RESEARCH PROTOTYPE - NOT A MEDICAL DEVICE")
    print("   This system is for research and simulation only.")
    print("   NOT for clinical use or patient care.")
    print("="*60)
    print(f"API server starting on {settings.api_host}:{settings.api_port}")
    print(f"Documentation: http://{settings.api_host}:{settings.api_port}/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    # Cleanup resources
    from backend.api.routers.sim import cleanup_simulation
    cleanup_simulation()
    print("API server shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )

