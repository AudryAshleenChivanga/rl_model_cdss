# H. pylori CDSS 3D Endoscopy RL Simulator

⚠️ **RESEARCH PROTOTYPE - NOT A MEDICAL DEVICE**

This system is a research prototype for simulation and educational purposes only. It is NOT intended for clinical use, diagnostic purposes, or patient care. This software has not been validated for medical use and should never be used to make clinical decisions.

---

## Overview

A reinforcement learning simulator that trains an RL policy to navigate a virtual upper GI tract, detect ulcer-like anomalies using a CNN, and stream live guidance to a web interface. The system combines:

- **3D Endoscopy Simulation**: Custom Gymnasium environment with pyrender/trimesh
- **Synthetic Lesion Generation**: Procedural texture synthesis for training data
- **Anomaly Detection CNN**: ResNet18 fine-tuned on synthetic frames
- **RL Policy**: PPO agent that maximizes coverage and anomaly detection
- **Live Streaming API**: FastAPI + WebSocket for real-time visualization
- **Web Dashboard**: Three.js viewer with metrics and controls

### 🎯 NEW: Enhanced Multi-Disease Detection System

The enhanced version includes:

- **Multi-Class CNN**: Detects 5 GI conditions (normal, H. pylori gastritis, peptic ulcer, tumor, inflammation)
- **Enhanced Action Space**: 12 actions including diagnostic capabilities (FLAG, BIOPSY, AI REQUEST)
- **Clinical Metrics**: Sensitivity, Precision, F1 Score tracking
- **Sophisticated Rewards**: True/false positive/negative aware reward system

See `ENHANCED_SYSTEM.md` for full details on the advanced clinical decision support capabilities.

## Tech Stack

- **Python 3.11** with PyTorch, Stable-Baselines3, Gymnasium
- **Rendering**: pyrender + trimesh + pyglet (offscreen)
- **CNN**: torchvision ResNet18 with PyTorch Lightning
- **Backend**: FastAPI + Uvicorn + WebSocket
- **Frontend**: HTML/CSS/JS + Three.js (no frameworks)
- **Deployment**: Docker + docker-compose

## ⚠️ **IMPORTANT: Windows Users**

**For full 3D rendering and functional RL training, you MUST use Docker or WSL2.**

Native Windows setup will run but with **placeholder images only** - the RL agent will be "blind" and cannot learn. See `DOCKER_QUICKSTART.md` for proper setup.

## Quick Start

### Prerequisites

- Python 3.11+
- **Docker Desktop (REQUIRED for full 3D rendering)** - See `DOCKER_QUICKSTART.md`
- CUDA-capable GPU (optional, CPU fallback supported)

### Option 1: Docker (Recommended - Full Functionality)

```bash
# Install Docker Desktop for Windows
# https://www.docker.com/products/docker-desktop/

# Build and run
cd rl_model_cdss
docker-compose up --build

# Access at:
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
```

See `DOCKER_QUICKSTART.md` for detailed Docker instructions.

### Option 2: Native Windows (Development/Testing Only)

### Installation

#### Automated Setup (Recommended)

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

This will:
- ✅ Create a virtual environment
- ✅ Install all dependencies
- ✅ Create necessary directories
- ✅ Set up .env configuration
- ✅ Run health checks

#### Manual Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Create directories
mkdir data checkpoints logs reports
```

### Generate Synthetic Training Data

```bash
# Generate 3000 episodes of synthetic frames with lesion labels
python backend/sim/renderer.py --export-dataset data/synth --episodes 3000
```

### Train the CNN

```bash
# Train anomaly detection CNN on synthetic data
python backend/models/cnn/train_cnn.py --config configs/train_cnn.yaml
```

### Train the RL Policy

```bash
# Train PPO policy for navigation and detection
python backend/models/rl/train_rl.py --config configs/train_rl.yaml
```

### Run the API Server

```bash
# Start FastAPI backend
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Open the Frontend

```bash
# Open in browser
# Navigate to: frontend/index.html
# Or serve with: python -m http.server 8080 --directory frontend
```

Then:
1. Paste a public GLTF URL from Sketchfab (e.g., a stomach/GI tract model)
2. Click "Load Model"
3. Click "Start Simulation"
4. Watch live frames, CNN probabilities, and RL actions

## Docker Deployment

```bash
# Build and run with docker-compose
docker-compose up --build

# Access at http://localhost:8000 (API) and http://localhost:8080 (frontend)
```

## Project Structure

```
hpylori-rl-sim/
├── backend/
│   ├── api/                 # FastAPI endpoints
│   │   ├── main.py
│   │   └── routers/
│   │       ├── sim.py       # Simulation control
│   │       ├── models.py    # Model loading
│   │       └── health.py    # Health checks
│   ├── sim/                 # Simulation environment
│   │   ├── env.py           # Gymnasium EndoscopyEnv
│   │   ├── renderer.py      # Pyrender 3D rendering
│   │   └── lesion_synth.py  # Synthetic lesion generator
│   ├── models/
│   │   ├── cnn/             # Anomaly detection CNN
│   │   │   ├── train_cnn.py
│   │   │   ├── dataset.py
│   │   │   ├── model.py
│   │   │   └── eval.py
│   │   └── rl/              # Reinforcement learning
│   │       ├── train_rl.py
│   │       ├── callbacks.py
│   │       └── export_onnx.py
│   ├── utils/
│   │   ├── config.py
│   │   └── logging.py
│   └── tests/
│       ├── test_env.py
│       ├── test_api.py
│       └── test_cnn.py
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── configs/
│   ├── sim.yaml
│   ├── train_cnn.yaml
│   └── train_rl.yaml
├── README.md
├── LICENSE
├── .env.example
└── docker-compose.yml
```

## Configuration

Edit YAML files in `configs/`:

- **sim.yaml**: Environment parameters (lesions, reward coefficients)
- **train_cnn.yaml**: CNN training hyperparameters
- **train_rl.yaml**: PPO training configuration

## Testing

```bash
# Run all tests
pytest backend/tests/ -v

# Run specific test
pytest backend/tests/test_env.py -v
```

## Model Export

Models are automatically exported during training:

- **CNN**: TorchScript format (`checkpoints/cnn_best.pt`)
- **PPO**: Stable-Baselines3 ZIP + ONNX (`checkpoints/ppo_best.zip`, `checkpoints/ppo_policy.onnx`)

## API Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

- `POST /load_model?gltf_url={url}` - Load 3D GI tract model
- `POST /sim/start` - Start simulation
- `POST /sim/stop` - Stop simulation
- `WS /stream` - WebSocket for live frame streaming
- `GET /metrics` - Training and inference metrics
- `GET /health` - System health status

## Evaluation Metrics

### CNN Performance
- ROC-AUC, PR-AUC, F1 score
- Confusion matrix
- Per-class precision/recall

### RL Performance
- Average episode coverage
- Anomaly detection recall
- Collision rate
- Mean episode reward

Reports are saved to `reports/` with HTML visualizations.

## Licensing and Attribution

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

### IMPORTANT: 3D Model Licensing

This simulator requires you to provide a 3D model of the GI tract (GLTF format). When using models from Sketchfab or other sources:

1. Ensure the model license permits your use case
2. Provide proper attribution to the original creator
3. Respect any restrictions (commercial use, derivatives, etc.)

**The developers of this simulator are not responsible for model licensing compliance. Users must obtain properly licensed models.**

## Research Use Only

**DISCLAIMER**: This software is provided for research, education, and simulation purposes only. It is NOT a medical device and has NOT been evaluated or approved by any regulatory authority (FDA, CE, etc.). 

Do NOT use this system:
- For clinical diagnosis or treatment decisions
- With real patient data (no PHI)
- As a substitute for professional medical advice
- In any clinical or patient care setting

## Citation

If you use this simulator in your research, please cite:

```bibtex
@software{hpylori_rl_sim_2025,
  title={H. pylori CDSS 3D Endoscopy RL Simulator},
  year={2025},
  note={Research prototype - not for clinical use}
}
```

## Contributing

This is a research prototype. For issues or enhancements, please open an issue on the repository.

## Support

For questions about the simulator, consult the documentation or open an issue. For medical questions, consult a qualified healthcare professional.

---

**Remember: This is a simulation tool for research only. Never use it for real patient care.**

