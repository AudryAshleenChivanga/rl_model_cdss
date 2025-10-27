@echo off
REM Quick activation script for virtual environment (Windows)

if not exist venv (
    echo ERROR: Virtual environment not found!
    echo    Run setup.bat first to create it
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Virtual environment activated
echo.
echo You can now run:
echo   python run.py api       # Start API server
echo   python run.py frontend  # Start frontend server
echo   python run.py --help    # See all commands
echo.
echo To deactivate later, run: deactivate

