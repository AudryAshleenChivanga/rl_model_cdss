#!/bin/bash
# Setup script for H. pylori RL Simulator (Linux/Mac)

set -e  # Exit on error

echo "============================================================"
echo "  H. pylori CDSS 3D Endoscopy RL Simulator - Setup"
echo "============================================================"
echo "  ⚠️  RESEARCH PROTOTYPE - NOT A MEDICAL DEVICE"
echo "============================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.11+ is required (found Python $python_version)"
    echo "   Please install Python 3.11 or higher"
    exit 1
fi

echo "✓ Python $python_version found"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists at ./venv"
    read -p "   Remove and recreate? (y/N): " response
    if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
        rm -rf venv
        python3 -m venv venv
        echo "✓ Virtual environment recreated"
    else
        echo "  Using existing virtual environment"
    fi
else
    python3 -m venv venv
    echo "✓ Virtual environment created at ./venv"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
echo "This may take several minutes..."
pip install -r backend/requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "❌ Error installing dependencies"
    exit 1
fi
echo ""

# Create necessary directories
echo "Creating project directories..."
mkdir -p data checkpoints logs reports
echo "✓ Directories created"
echo ""

# Copy .env.example to .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << 'EOF'
# H. pylori RL Simulator Environment Configuration

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=["http://localhost:8080", "http://localhost:3000"]

# Paths
DATA_DIR=./data
CHECKPOINTS_DIR=./checkpoints
LOGS_DIR=./logs
REPORTS_DIR=./reports

# Device (change to 'cuda' if you have a GPU)
DEVICE=cpu

# Model Configuration
CNN_CHECKPOINT=./checkpoints/cnn_best.pt
PPO_CHECKPOINT=./checkpoints/ppo_best.zip

# Simulation Configuration
RENDER_WIDTH=224
RENDER_HEIGHT=224
RENDER_FPS=10

# Logging
LOG_LEVEL=INFO

# Warning
SHOW_RESEARCH_DISCLAIMER=true
EOF
    echo "✓ .env file created"
else
    echo "  .env file already exists"
fi
echo ""

# Run quick health check
echo "Running health check..."
python3 -c "
import sys
try:
    import torch
    import fastapi
    import gymnasium
    import numpy as np
    print('✓ Core packages imported successfully')
    print(f'  - PyTorch: {torch.__version__}')
    print(f'  - FastAPI: {fastapi.__version__}')
    print(f'  - Gymnasium: {gymnasium.__version__}')
    print(f'  - NumPy: {np.__version__}')
    
    if torch.cuda.is_available():
        print(f'  - GPU Available: {torch.cuda.get_device_name(0)}')
    else:
        print('  - GPU: Not available (using CPU)')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"
echo ""

# Summary
echo "============================================================"
echo "  ✅ Setup Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate virtual environment (if not already active):"
echo "   source venv/bin/activate"
echo ""
echo "2. Generate synthetic training data (optional):"
echo "   python run.py generate-data --episodes 1000"
echo ""
echo "3. Train models (optional):"
echo "   python run.py train-cnn"
echo "   python run.py train-rl"
echo ""
echo "4. Start the API server:"
echo "   python run.py api"
echo ""
echo "5. In another terminal, open the frontend:"
echo "   python run.py frontend"
echo "   # Or open frontend/index.html in your browser"
echo ""
echo "For more information, see:"
echo "  - README.md (full documentation)"
echo "  - QUICKSTART.md (quick start guide)"
echo ""
echo "⚠️  Remember: This is a RESEARCH PROTOTYPE"
echo "   NOT for clinical use or patient care"
echo "============================================================"

