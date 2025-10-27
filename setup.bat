@echo off
REM Setup script for H. pylori RL Simulator (Windows)

echo ============================================================
echo   H. pylori CDSS 3D Endoscopy RL Simulator - Setup
echo ============================================================
echo   WARNING: RESEARCH PROTOTYPE - NOT A MEDICAL DEVICE
echo ============================================================
echo.

REM Check if Python is installed
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.11 or higher.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%
echo.

REM Create virtual environment
echo Creating virtual environment...
if exist venv (
    echo WARNING: Virtual environment already exists at .\venv
    set /p response="Remove and recreate? (y/N): "
    if /i "%response%"=="y" (
        rmdir /s /q venv
        python -m venv venv
        echo Virtual environment recreated
    ) else (
        echo Using existing virtual environment
    )
) else (
    python -m venv venv
    echo Virtual environment created at .\venv
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel
echo pip upgraded
echo.

REM Install dependencies
echo Installing dependencies...
echo This may take several minutes...
pip install -r backend\requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies installed successfully
echo.

REM Create necessary directories
echo Creating project directories...
if not exist data mkdir data
if not exist checkpoints mkdir checkpoints
if not exist logs mkdir logs
if not exist reports mkdir reports
echo Directories created
echo.

REM Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file...
    (
        echo # H. pylori RL Simulator Environment Configuration
        echo.
        echo # API Configuration
        echo API_HOST=0.0.0.0
        echo API_PORT=8000
        echo CORS_ORIGINS=["http://localhost:8080", "http://localhost:3000"]
        echo.
        echo # Paths
        echo DATA_DIR=./data
        echo CHECKPOINTS_DIR=./checkpoints
        echo LOGS_DIR=./logs
        echo REPORTS_DIR=./reports
        echo.
        echo # Device ^(change to 'cuda' if you have a GPU^)
        echo DEVICE=cpu
        echo.
        echo # Model Configuration
        echo CNN_CHECKPOINT=./checkpoints/cnn_best.pt
        echo PPO_CHECKPOINT=./checkpoints/ppo_best.zip
        echo.
        echo # Simulation Configuration
        echo RENDER_WIDTH=224
        echo RENDER_HEIGHT=224
        echo RENDER_FPS=10
        echo.
        echo # Logging
        echo LOG_LEVEL=INFO
        echo.
        echo # Warning
        echo SHOW_RESEARCH_DISCLAIMER=true
    ) > .env
    echo .env file created
) else (
    echo .env file already exists
)
echo.

REM Run health check
echo Running health check...
python -c "import sys; import torch; import fastapi; import gymnasium; import numpy as np; print('Core packages imported successfully'); print(f'  - PyTorch: {torch.__version__}'); print(f'  - FastAPI: {fastapi.__version__}'); print(f'  - Gymnasium: {gymnasium.__version__}'); print(f'  - NumPy: {np.__version__}'); print(f'  - GPU Available: {torch.cuda.is_available()}')"
echo.

REM Summary
echo ============================================================
echo   Setup Complete!
echo ============================================================
echo.
echo Next steps:
echo.
echo 1. Activate virtual environment (if not already active):
echo    venv\Scripts\activate
echo.
echo 2. Generate synthetic training data (optional):
echo    python run.py generate-data --episodes 1000
echo.
echo 3. Train models (optional):
echo    python run.py train-cnn
echo    python run.py train-rl
echo.
echo 4. Start the API server:
echo    python run.py api
echo.
echo 5. In another terminal, open the frontend:
echo    python run.py frontend
echo    # Or open frontend\index.html in your browser
echo.
echo For more information, see:
echo   - README.md (full documentation)
echo   - QUICKSTART.md (quick start guide)
echo.
echo WARNING: This is a RESEARCH PROTOTYPE
echo    NOT for clinical use or patient care
echo ============================================================
echo.
pause

