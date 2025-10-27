#!/bin/bash
# Quick activation script for virtual environment (Linux/Mac)

if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Run ./setup.sh first to create it"
    exit 1
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "✓ Virtual environment activated"
echo ""
echo "You can now run:"
echo "  python run.py api       # Start API server"
echo "  python run.py frontend  # Start frontend server"
echo "  python run.py --help    # See all commands"
echo ""
echo "To deactivate later, run: deactivate"

