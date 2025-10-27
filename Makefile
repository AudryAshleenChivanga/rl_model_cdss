# Makefile for H. pylori RL Simulator

.PHONY: help install install-dev test clean run-api run-frontend docker-build docker-up docker-down train-cnn train-rl generate-data lint format

help:
	@echo "H. pylori RL Simulator - Makefile Commands"
	@echo "==========================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install dependencies"
	@echo "  make install-dev      Install dev dependencies"
	@echo ""
	@echo "Running:"
	@echo "  make run-api          Start API server"
	@echo "  make run-frontend     Start frontend server"
	@echo ""
	@echo "Training:"
	@echo "  make generate-data    Generate synthetic training data"
	@echo "  make train-cnn        Train CNN model"
	@echo "  make train-rl         Train RL policy"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     Build Docker images"
	@echo "  make docker-up        Start Docker containers"
	@echo "  make docker-down      Stop Docker containers"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests"
	@echo "  make test-env         Test environment only"
	@echo "  make test-api         Test API only"
	@echo "  make test-cnn         Test CNN only"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run linters"
	@echo "  make format           Format code with black"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Remove generated files"
	@echo ""

# Installation
install:
	pip install -r backend/requirements.txt

install-dev:
	pip install -r backend/requirements.txt
	pip install pytest pytest-asyncio pytest-cov black flake8 mypy

# Running
run-api:
	uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

run-frontend:
	python -m http.server 8080 --directory frontend

# Training
generate-data:
	python backend/sim/renderer.py --export-dataset data/synth --episodes 1000 --frames-per-episode 10

train-cnn:
	python backend/models/cnn/train_cnn.py --config configs/train_cnn.yaml

train-rl:
	python backend/models/rl/train_rl.py --config configs/train_rl.yaml

# Docker
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Testing
test:
	pytest backend/tests/ -v

test-env:
	pytest backend/tests/test_env.py -v

test-api:
	pytest backend/tests/test_api.py -v

test-cnn:
	pytest backend/tests/test_cnn.py -v

test-coverage:
	pytest backend/tests/ --cov=backend --cov-report=html --cov-report=term

# Code Quality
lint:
	flake8 backend/ --exclude=__pycache__,venv --max-line-length=100
	mypy backend/ --ignore-missing-imports

format:
	black backend/ --line-length=100

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/

# Setup environment
setup:
	python -m venv venv
	@echo "Virtual environment created. Activate with:"
	@echo "  Windows: venv\\Scripts\\activate"
	@echo "  Linux/Mac: source venv/bin/activate"
	@echo "Then run: make install"

# Quick start for demo
demo:
	@echo "Starting demo (no trained models required)..."
	@echo "1. Starting API server..."
	@make run-api &
	@sleep 5
	@echo "2. Open http://localhost:8000 for API"
	@echo "3. Open frontend/index.html in your browser"
	@echo "4. Load a GLTF model and start simulation"

