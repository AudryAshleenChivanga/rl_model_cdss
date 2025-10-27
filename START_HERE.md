# 🚀 START HERE

## H. pylori CDSS 3D Endoscopy RL Simulator

⚠️ **RESEARCH PROTOTYPE - NOT A MEDICAL DEVICE - FOR SIMULATION ONLY**

---

## ⚡ Super Quick Start (3 Commands)

### 1️⃣ **Run Setup**

**Linux/Mac:**
```bash
chmod +x setup.sh && ./setup.sh
```

**Windows:**
```bash
setup.bat
```

*Takes 5-10 minutes. Installs everything automatically.*

---

### 2️⃣ **Activate Virtual Environment**

**Linux/Mac:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

*You'll see `(venv)` in your prompt*

---

### 3️⃣ **Start the System**

```bash
# Terminal 1: Start API
python run.py api

# Terminal 2: Start Frontend (with venv activated)
python run.py frontend
```

Then open **http://localhost:8080** in your browser! 🎉

---

## 📖 What to Read

- **New users?** → Read `INSTALL.md` (installation troubleshooting)
- **Want details?** → Read `QUICKSTART.md` (step-by-step guide)
- **Need everything?** → Read `README.md` (full documentation)

---

## 🎯 What You Can Do

✅ **Demo Mode** (no training needed):
- Load a GLTF model
- Watch RL agent navigate (random policy)
- See CNN predictions (will be zero without training)

✅ **Full Training Pipeline**:
```bash
# 1. Generate synthetic training data
python run.py generate-data --episodes 1000

# 2. Train CNN (20-60 min)
python run.py train-cnn

# 3. Train RL policy (2-6 hours)
python run.py train-rl

# 4. Run with trained models
python run.py api
python run.py frontend
```

✅ **Docker** (if you prefer):
```bash
docker-compose up --build
# Access at http://localhost:8080
```

---

## 🆘 Quick Troubleshooting

**Problem:** "Python 3.11+ required"
- **Fix:** Install Python 3.11 or higher

**Problem:** "Virtual environment not found"
- **Fix:** Run `setup.sh` or `setup.bat`

**Problem:** "Module not found"
- **Fix:** Activate venv: `source venv/bin/activate`

**Problem:** "CUDA not available" (but you have GPU)
- **Fix:** Install PyTorch with CUDA support (see INSTALL.md)

For more help → See `INSTALL.md`

---

## 🧪 Verify Installation

```bash
python run.py check
```

Should show:
```
✓ Python 3.11.x
✓ PyTorch x.x.x
✓ FastAPI x.x.x
✓ Gymnasium x.x.x
✅ Setup looks good!
```

---

## 📁 File Guide

- `START_HERE.md` ← **You are here!**
- `INSTALL.md` - Installation and troubleshooting
- `QUICKSTART.md` - Detailed quick start guide
- `README.md` - Complete documentation
- `CONTRIBUTING.md` - How to contribute
- `run.py` - Main command script

---

## ⚡ Common Commands

```bash
python run.py check           # Check if everything is set up
python run.py api             # Start API server
python run.py frontend        # Start frontend server
python run.py generate-data   # Generate training data
python run.py train-cnn       # Train anomaly detector
python run.py train-rl        # Train RL policy
python run.py test            # Run tests
python run.py --help          # See all options
```

---

## 🎓 Learning Path

**Day 1**: Setup → Demo Mode
- Run setup scripts
- Start API and frontend
- Load a simple GLTF model
- See the system working (random policy)

**Day 2-3**: Generate Data → Train CNN
- Generate synthetic dataset
- Train anomaly detection CNN
- Evaluate performance

**Day 4-7**: Train RL → Full System
- Train RL navigation policy
- Run full pipeline with trained models
- Experiment with parameters

---

## 🚨 Remember

This is a **RESEARCH PROTOTYPE**:
- ❌ NOT for clinical use
- ❌ NOT for patient care
- ❌ NOT for medical decisions
- ✅ FOR research and simulation ONLY

---

## 🤝 Need Help?

1. Check `INSTALL.md` for detailed troubleshooting
2. Run `python run.py --help` for command options
3. Look at `backend/tests/` for usage examples
4. Review configs in `configs/` directory

---

## 🎉 You're Ready!

Now run:
```bash
./setup.sh              # Or setup.bat on Windows
source venv/bin/activate
python run.py check
python run.py api
```

Then open http://localhost:8080 and start exploring! 🚀

---

**Built with:** PyTorch • Stable-Baselines3 • FastAPI • Three.js

**For:** Research • Education • Simulation

**Not for:** Clinical use • Patient care • Medical decisions

