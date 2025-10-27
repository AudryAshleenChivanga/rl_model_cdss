"""Model loading and management endpoints."""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, HttpUrl
from typing import Optional
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.models.cnn.model import load_model as load_cnn_model
from backend.utils.config import settings

router = APIRouter()

# Global model state
cnn_model = None
ppo_model = None
current_gltf_url = None


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    loaded: bool
    path: str | None = None
    device: str | None = None


class LoadModelRequest(BaseModel):
    """Request to load a model."""
    gltf_url: Optional[HttpUrl] = None


@router.post("/load_model")
async def load_model(gltf_url: str = Query(..., description="GLTF model URL")):
    """Load 3D GI tract model from URL.
    
    Args:
        gltf_url: Public URL to GLTF/GLB file
        
    Returns:
        Status of model loading
        
    **Important**: Ensure the GLTF model is properly licensed.
    The user is responsible for model licensing compliance.
    """
    global current_gltf_url
    
    try:
        # Validate URL
        if not gltf_url.startswith(('http://', 'https://')):
            raise HTTPException(
                status_code=400,
                detail="Invalid URL. Must start with http:// or https://"
            )
        
        # Store URL for environment initialization
        current_gltf_url = gltf_url
        
        return {
            "status": "success",
            "message": f"Model URL set: {gltf_url}",
            "gltf_url": gltf_url,
            "disclaimer": "Ensure model is properly licensed for your use case"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@router.get("/models/info")
async def get_models_info():
    """Get information about loaded models.
    
    Returns status of CNN and PPO models.
    """
    cnn_info = ModelInfo(
        name="CNN Anomaly Detector",
        loaded=cnn_model is not None,
        path=settings.cnn_checkpoint if cnn_model is not None else None,
        device=settings.device if cnn_model is not None else None,
    )
    
    ppo_info = ModelInfo(
        name="PPO Navigation Policy",
        loaded=ppo_model is not None,
        path=settings.ppo_checkpoint if ppo_model is not None else None,
        device=settings.device if ppo_model is not None else None,
    )
    
    return {
        "cnn": cnn_info,
        "ppo": ppo_info,
        "gltf_url": current_gltf_url,
    }


@router.post("/models/load_cnn")
async def load_cnn(checkpoint_path: Optional[str] = None):
    """Load CNN model from checkpoint.
    
    Args:
        checkpoint_path: Path to CNN checkpoint (optional)
        
    Returns:
        Status of model loading
    """
    global cnn_model
    
    try:
        device = settings.device
        if checkpoint_path is None:
            checkpoint_path = settings.cnn_checkpoint
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            return {
                "status": "warning",
                "message": f"CNN checkpoint not found: {checkpoint_path}",
                "note": "CNN will not be available for inference. Train CNN first."
            }
        
        cnn_model = load_cnn_model(str(checkpoint_path), device=device)
        
        return {
            "status": "success",
            "message": "CNN model loaded",
            "checkpoint": str(checkpoint_path),
            "device": device,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load CNN: {str(e)}")


@router.post("/models/load_ppo")
async def load_ppo(checkpoint_path: Optional[str] = None):
    """Load PPO model from checkpoint.
    
    Args:
        checkpoint_path: Path to PPO checkpoint (optional)
        
    Returns:
        Status of model loading
    """
    global ppo_model
    
    try:
        from stable_baselines3 import PPO
        
        if checkpoint_path is None:
            checkpoint_path = settings.ppo_checkpoint
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            return {
                "status": "warning",
                "message": f"PPO checkpoint not found: {checkpoint_path}",
                "note": "PPO will not be available. Train RL policy first."
            }
        
        ppo_model = PPO.load(str(checkpoint_path), device=settings.device)
        
        return {
            "status": "success",
            "message": "PPO model loaded",
            "checkpoint": str(checkpoint_path),
            "device": settings.device,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load PPO: {str(e)}")


def get_current_gltf_url() -> Optional[str]:
    """Get current GLTF URL."""
    return current_gltf_url


def get_cnn_model():
    """Get loaded CNN model."""
    return cnn_model


def get_ppo_model():
    """Get loaded PPO model."""
    return ppo_model

