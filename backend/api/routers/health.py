"""Health check endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict
import psutil
import torch
from datetime import datetime

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    disclaimer: str


class SystemMetrics(BaseModel):
    """System metrics response."""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    gpu_available: bool
    gpu_name: str | None = None
    gpu_memory_gb: float | None = None


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status.
    
    Returns basic health check with timestamp and disclaimer.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        disclaimer="RESEARCH PROTOTYPE - NOT A MEDICAL DEVICE - FOR SIMULATION ONLY"
    )


@router.get("/metrics", response_model=SystemMetrics)
async def get_metrics():
    """Get system resource metrics.
    
    Returns CPU, memory, and GPU usage information.
    """
    # CPU and Memory
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    memory_available_gb = memory.available / (1024 ** 3)
    
    # GPU info
    gpu_available = torch.cuda.is_available()
    gpu_name = None
    gpu_memory_gb = None
    
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    
    return SystemMetrics(
        cpu_percent=cpu_percent,
        memory_percent=memory_percent,
        memory_available_gb=memory_available_gb,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
    )


@router.get("/disclaimer")
async def get_disclaimer():
    """Get full research disclaimer.
    
    Returns comprehensive disclaimer about research use only.
    """
    return {
        "title": "RESEARCH PROTOTYPE DISCLAIMER",
        "warnings": [
            "This is a RESEARCH PROTOTYPE for simulation and educational purposes ONLY",
            "This system is NOT a medical device",
            "This system has NOT been validated for clinical use",
            "This system is NOT intended for diagnosis, treatment, or patient care",
            "NEVER use this system to make clinical decisions",
            "NEVER use this system with real patient data",
            "Users must comply with all applicable laws and regulations",
            "Users are responsible for obtaining properly licensed 3D models",
        ],
        "license": "MIT License - See LICENSE file",
        "contact": "For research questions only - not for medical advice",
    }

