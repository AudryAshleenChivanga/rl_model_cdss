"""Configuration management utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment variables."""

    # API Configuration
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_workers: int = Field(default=1, alias="API_WORKERS")
    cors_origins: list[str] = Field(
        default=["http://localhost:8080", "http://localhost:3000"],
        alias="CORS_ORIGINS"
    )

    # Paths
    data_dir: str = Field(default="./data", alias="DATA_DIR")
    checkpoints_dir: str = Field(default="./checkpoints", alias="CHECKPOINTS_DIR")
    logs_dir: str = Field(default="./logs", alias="LOGS_DIR")
    reports_dir: str = Field(default="./reports", alias="REPORTS_DIR")

    # Model Configuration
    cnn_checkpoint: Optional[str] = Field(
        default="./checkpoints/cnn_best.pt", alias="CNN_CHECKPOINT"
    )
    ppo_checkpoint: Optional[str] = Field(
        default="./checkpoints/ppo_best.zip", alias="PPO_CHECKPOINT"
    )
    device: str = Field(default="cuda", alias="DEVICE")

    # Simulation Configuration
    default_gltf_url: Optional[str] = Field(default=None, alias="DEFAULT_GLTF_URL")
    render_width: int = Field(default=224, alias="RENDER_WIDTH")
    render_height: int = Field(default=224, alias="RENDER_HEIGHT")
    render_fps: int = Field(default=10, alias="RENDER_FPS")

    # Training Configuration
    cnn_batch_size: int = Field(default=32, alias="CNN_BATCH_SIZE")
    cnn_epochs: int = Field(default=20, alias="CNN_EPOCHS")
    cnn_lr: float = Field(default=3e-4, alias="CNN_LR")

    rl_total_timesteps: int = Field(default=2000000, alias="RL_TOTAL_TIMESTEPS")
    rl_n_steps: int = Field(default=2048, alias="RL_N_STEPS")
    rl_learning_rate: float = Field(default=3e-4, alias="RL_LEARNING_RATE")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    tensorboard_dir: str = Field(default="./logs/tensorboard", alias="TENSORBOARD_DIR")

    # Security
    max_upload_size: int = Field(default=100, alias="MAX_UPLOAD_SIZE")  # MB
    request_timeout: int = Field(default=300, alias="REQUEST_TIMEOUT")  # seconds

    # Warning Banner
    show_research_disclaimer: bool = Field(default=True, alias="SHOW_RESEARCH_DISCLAIMER")

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def save_yaml_config(config: Dict[str, Any], config_path: str | Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        config_path: Path to output YAML file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def get_device(device: Optional[str] = None) -> str:
    """Get the appropriate device for PyTorch operations.

    Args:
        device: Requested device ('cuda', 'cpu', or None for auto)

    Returns:
        Device string ('cuda' or 'cpu')
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU")
        device = "cpu"

    return device


def ensure_dirs(settings: Settings) -> None:
    """Ensure all required directories exist.

    Args:
        settings: Application settings
    """
    dirs = [
        settings.data_dir,
        settings.checkpoints_dir,
        settings.logs_dir,
        settings.reports_dir,
        settings.tensorboard_dir,
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()

