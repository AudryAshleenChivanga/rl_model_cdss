#!/usr/bin/env python3
"""
Convenience script to run H. pylori RL Simulator.

Usage:
    python run.py --help
    python run.py api
    python run.py frontend
    python run.py generate-data --episodes 1000
    python run.py train-cnn
    python run.py train-rl
    python run.py test
"""

import argparse
import subprocess
import sys
from pathlib import Path


def print_banner():
    """Print application banner."""
    print("=" * 70)
    print("  H. pylori CDSS 3D Endoscopy RL Simulator")
    print("=" * 70)
    print("  WARNING: RESEARCH PROTOTYPE - NOT A MEDICAL DEVICE")
    print("  For research and simulation purposes ONLY")
    print("=" * 70)
    print()


def run_api(host="0.0.0.0", port=8000, reload=True):
    """Run API server."""
    print(f"Starting API server on {host}:{port}...")
    cmd = ["uvicorn", "backend.api.main:app", "--host", host, "--port", str(port)]
    if reload:
        cmd.append("--reload")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nAPI server stopped.")


def run_frontend(port=8080):
    """Run frontend server."""
    print(f"Starting frontend server on port {port}...")
    print(f"Open http://localhost:{port} in your browser")
    cmd = ["python", "-m", "http.server", str(port), "--directory", "frontend"]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nFrontend server stopped.")


def generate_data(episodes=1000, frames_per_episode=10, output="data/synth"):
    """Generate synthetic training data."""
    print(f"Generating {episodes} episodes with {frames_per_episode} frames each...")
    cmd = [
        "python", "backend/sim/renderer.py",
        "--export-dataset", output,
        "--episodes", str(episodes),
        "--frames-per-episode", str(frames_per_episode)
    ]
    
    subprocess.run(cmd)
    print(f"Data saved to: {output}")


def train_cnn(config="configs/train_cnn.yaml"):
    """Train CNN model."""
    print("Training CNN anomaly detector...")
    cmd = ["python", "backend/models/cnn/train_cnn.py", "--config", config]
    subprocess.run(cmd)


def train_rl(config="configs/train_rl.yaml"):
    """Train RL policy."""
    print("Training RL policy...")
    cmd = ["python", "backend/models/rl/train_rl.py", "--config", config]
    subprocess.run(cmd)


def run_tests(verbose=True):
    """Run test suite."""
    print("Running tests...")
    cmd = ["pytest", "backend/tests/"]
    if verbose:
        cmd.append("-v")
    
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def check_setup():
    """Check if setup is correct."""
    print("Checking setup...")
    
    issues = []
    warnings = []
    
    # Check if virtual environment exists
    venv_path = Path("venv")
    if not venv_path.exists():
        warnings.append("Virtual environment not found at ./venv")
        warnings.append("Run './setup.sh' (Linux/Mac) or 'setup.bat' (Windows) to create it")
    
    # Check if running in virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    if not in_venv and venv_path.exists():
        warnings.append("Virtual environment exists but not activated")
        warnings.append("Activate with: source venv/bin/activate (Linux/Mac)")
        warnings.append("            or: venv\\Scripts\\activate (Windows)")
    
    # Check Python version
    if sys.version_info < (3, 11):
        issues.append(f"Python 3.11+ required (you have {sys.version_info.major}.{sys.version_info.minor})")
    else:
        print(f"[OK] Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check if dependencies installed
    try:
        import torch
        import fastapi
        import gymnasium
        print(f"[OK] PyTorch {torch.__version__}")
        print(f"[OK] FastAPI {fastapi.__version__}")
        print(f"[OK] Gymnasium {gymnasium.__version__}")
        
        # Check GPU
        if torch.cuda.is_available():
            print(f"[OK] GPU Available: {torch.cuda.get_device_name(0)}")
        else:
            print("[INFO] GPU: Not available (using CPU)")
    except ImportError as e:
        issues.append(f"Missing dependencies: {e}")
        issues.append("Run setup script or: pip install -r backend/requirements.txt")
    
    # Check directories
    for dir_name in ["configs", "backend", "frontend"]:
        if not Path(dir_name).exists():
            issues.append(f"Missing directory: {dir_name}")
    
    # Check if necessary directories exist
    for dir_name in ["data", "checkpoints", "logs", "reports"]:
        if not Path(dir_name).exists():
            warnings.append(f"Directory '{dir_name}' will be created when needed")
    
    # Print warnings
    if warnings:
        print("\n[WARNING] Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    # Print issues
    if issues:
        print("\n[ERROR] Setup issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\n[TIP] Quick fix:")
        print("  Run the setup script to configure everything automatically:")
        print("    ./setup.sh        (Linux/Mac)")
        print("    setup.bat         (Windows)")
        return False
    else:
        print("\n[SUCCESS] Setup looks good!")
        if not in_venv:
            print("\n[TIP] For best results, run from within the virtual environment")
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="H. pylori RL Simulator - Run Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py check              # Check setup
  python run.py api                # Run API server
  python run.py frontend           # Run frontend server
  python run.py generate-data      # Generate 1000 episodes
  python run.py train-cnn          # Train CNN
  python run.py train-rl           # Train RL
  python run.py test               # Run tests

Quick Start:
  1. python run.py check
  2. python run.py generate-data --episodes 100
  3. python run.py api
  4. (In another terminal) python run.py frontend
  5. Open http://localhost:8080 in browser

Note: This is a RESEARCH PROTOTYPE. NOT for clinical use.
        """
    )
    
    parser.add_argument(
        "command",
        choices=["check", "api", "frontend", "generate-data", "train-cnn", "train-rl", "test"],
        help="Command to run"
    )
    
    # API options
    parser.add_argument("--host", default="0.0.0.0", help="API host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    
    # Data generation options
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes (default: 1000)")
    parser.add_argument("--frames-per-episode", type=int, default=10, help="Frames per episode (default: 10)")
    parser.add_argument("--output", default="data/synth", help="Output directory (default: data/synth)")
    
    # Training options
    parser.add_argument("--config", help="Config file path")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Execute command
    if args.command == "check":
        check_setup()
    
    elif args.command == "api":
        run_api(host=args.host, port=args.port, reload=not args.no_reload)
    
    elif args.command == "frontend":
        run_frontend(port=args.port if args.port != 8000 else 8080)
    
    elif args.command == "generate-data":
        generate_data(
            episodes=args.episodes,
            frames_per_episode=args.frames_per_episode,
            output=args.output
        )
    
    elif args.command == "train-cnn":
        config = args.config or "configs/train_cnn.yaml"
        train_cnn(config=config)
    
    elif args.command == "train-rl":
        config = args.config or "configs/train_rl.yaml"
        train_rl(config=config)
    
    elif args.command == "test":
        run_tests()


if __name__ == "__main__":
    main()

