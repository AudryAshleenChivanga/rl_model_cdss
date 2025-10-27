# Quick Start Guide

⚠️ **DISCLAIMER**: This is a research prototype. NOT for clinical use.

## Prerequisites

- Python 3.11+
- CUDA-capable GPU (optional, CPU fallback supported)
- 8GB+ RAM
- Docker (optional)

## Option 1: Local Installation (Recommended for Development)

### 1. Run Setup Script

**Linux/Mac:**
```bash
cd rl_model_cdss
chmod +x setup.sh
./setup.sh
```

**Windows:**
```bash
cd rl_model_cdss
setup.bat
```

This automatically:
- Creates virtual environment at `./venv`
- Installs all dependencies
- Creates directories (data, checkpoints, logs, reports)
- Sets up `.env` configuration file
- Runs health checks

**Or Manual Setup:**
```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r backend/requirements.txt
```

### 2. Generate Synthetic Training Data

```bash
# Generate 1000 episodes (adjust for your needs)
python backend/sim/renderer.py --export-dataset data/synth --episodes 1000 --frames-per-episode 10
```

This creates `data/synth/` with images and labels.

### 3. Train CNN (Optional)

```bash
# Train anomaly detection CNN
python backend/models/cnn/train_cnn.py --config configs/train_cnn.yaml
```

This will:
- Train ResNet18 on synthetic data
- Save checkpoints to `checkpoints/cnn_best.pt`
- Export TorchScript and ONNX models
- Generate evaluation reports

**Time**: ~20-30 minutes on GPU, longer on CPU

### 4. Train RL Policy (Optional)

```bash
# Train PPO policy
python backend/models/rl/train_rl.py --config configs/train_rl.yaml
```

This will:
- Train PPO agent in the environment
- Save checkpoints to `checkpoints/ppo_best.zip`
- Export ONNX model
- Log to TensorBoard

**Time**: 2-4 hours depending on timesteps

**Monitor training**:
```bash
tensorboard --logdir logs/rl
```

### 5. Start API Server

```bash
# In one terminal
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at: http://localhost:8000

### 6. Open Frontend

```bash
# In another terminal (optional static server)
python -m http.server 8080 --directory frontend
```

Or simply open `frontend/index.html` in your browser.

### 7. Use the Simulator

1. **Paste GLTF URL**: Enter a public GLTF model URL
   - Example: `https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Cube/glTF/Cube.gltf`
   - Or use a GI tract model from Sketchfab (ensure proper licensing!)

2. **Load Model**: Click "Load Model"

3. **Start Simulation**: Click "Start Simulation"

4. **Watch**: Live frames, CNN predictions, and RL actions will stream

## Option 2: Docker (Production)

### 1. Build and Run

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d
```

### 2. Access

- **API**: http://localhost:8000
- **Frontend**: http://localhost:8080
- **API Docs**: http://localhost:8000/docs

### 3. Stop

```bash
docker-compose down
```

## Option 3: Quick Demo (No Training)

If you want to see the system without training models:

1. Start the API server
2. Open frontend
3. Load a GLTF model
4. Start simulation

The system will work with **random actions** and **zero CNN predictions** (since models aren't trained yet). This lets you test the infrastructure.

## Testing

```bash
# Run all tests
pytest backend/tests/ -v

# Run specific test file
pytest backend/tests/test_env.py -v

# Run with coverage
pytest backend/tests/ --cov=backend --cov-report=html
```

## Common Issues

### Issue: "CUDA not available"
**Solution**: System will automatically fall back to CPU. To use GPU, ensure PyTorch CUDA is installed:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "pyrender cannot initialize"
**Solution**: For headless servers, ensure EGL is available:
```bash
# Ubuntu
sudo apt-get install libegl1-mesa-dev
```

### Issue: "WebSocket connection failed"
**Solution**: 
- Check backend is running on port 8000
- Check CORS settings in `.env`
- Try using `localhost` instead of `127.0.0.1`

### Issue: "Model not found"
**Solution**: Train models first or skip model loading to use random policy

## Configuration

### Environment Variables

Create `.env` file (copy from `.env.example`):

```bash
# API
API_HOST=0.0.0.0
API_PORT=8000

# Device
DEVICE=cuda  # or 'cpu'

# Paths
CHECKPOINTS_DIR=./checkpoints
DATA_DIR=./data
```

### YAML Configs

Edit configs in `configs/`:
- `sim.yaml`: Environment parameters
- `train_cnn.yaml`: CNN training settings
- `train_rl.yaml`: RL training settings

## Next Steps

1. **Experiment**: Try different GLTF models
2. **Tune Hyperparameters**: Adjust configs for better performance
3. **Visualize**: Use TensorBoard to monitor training
4. **Evaluate**: Check reports in `reports/` directory
5. **Export**: Use trained models for inference

## API Endpoints

Key endpoints:

- `GET /api/health` - Health check
- `GET /api/metrics` - System metrics
- `POST /api/load_model?gltf_url=...` - Load 3D model
- `POST /api/sim/start` - Start simulation
- `POST /api/sim/stop` - Stop simulation
- `WS /api/stream` - WebSocket stream

Full docs: http://localhost:8000/docs

## Resources

- **README.md**: Full documentation
- **NOTICE.md**: Licensing and disclaimers
- **configs/**: Configuration examples
- **backend/tests/**: Test examples

## Support

This is a research prototype. For issues:
1. Check logs in `logs/` directory
2. Review test cases for examples
3. Check API documentation

## Remember

⚠️ **This is NOT a medical device. For research and simulation only.**

