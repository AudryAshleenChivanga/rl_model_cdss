# Installation Guide

⚠️ **RESEARCH PROTOTYPE - NOT A MEDICAL DEVICE**

---

## Prerequisites

✅ **Python 3.11 or higher**
✅ **8GB+ RAM recommended**
✅ **GPU optional** (CUDA-capable for faster training)

---

## Quick Install (3 Steps)

### Step 1: Run Setup Script

Choose your platform:

#### 🐧 **Linux / 🍎 Mac**

```bash
cd rl_model_cdss
chmod +x setup.sh
./setup.sh
```

#### 🪟 **Windows**

```bash
cd rl_model_cdss
setup.bat
```

**What this does:**
- ✅ Creates virtual environment (`./venv`)
- ✅ Installs all Python dependencies (~2GB download)
- ✅ Creates project directories
- ✅ Generates `.env` configuration file
- ✅ Runs health checks

**Time:** 5-10 minutes depending on internet speed

---

### Step 2: Activate Virtual Environment

**Every time you start a new terminal**, activate the venv:

#### Linux/Mac:
```bash
source venv/bin/activate

# Or use the helper script:
source activate.sh
```

#### Windows:
```bash
venv\Scripts\activate

# Or use the helper script:
activate.bat
```

**You'll see `(venv)` in your terminal prompt when active**

---

### Step 3: Verify Installation

```bash
python run.py check
```

Should output:
```
✓ PyTorch X.X.X
✓ FastAPI X.X.X
✓ Gymnasium X.X.X
✓ Setup looks good!
```

---

## ✅ You're Done!

Now you can:

```bash
# Start the system
python run.py api

# In another terminal (with venv activated):
python run.py frontend
```

Or see all options:
```bash
python run.py --help
```

---

## Troubleshooting

### Issue: "Python 3.11+ required"

**Solution:** Install Python 3.11 or newer:
- **Windows:** https://www.python.org/downloads/
- **Mac:** `brew install python@3.11`
- **Ubuntu:** 
  ```bash
  sudo add-apt-repository ppa:deadsnakes/ppa
  sudo apt update
  sudo apt install python3.11 python3.11-venv
  ```

### Issue: "pip: command not found" or "No module named pip"

**Solution:**
```bash
python -m ensurepip --upgrade
```

### Issue: "venv creation failed"

**Solution on Ubuntu/Debian:**
```bash
sudo apt install python3.11-venv
```

### Issue: "pyrender cannot initialize"

**Solution:** Install OpenGL dependencies:

**Ubuntu/Debian:**
```bash
sudo apt-get install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libegl1-mesa-dev
```

**Mac:**
```bash
brew install mesa
```

**Windows:** Usually works out of the box. If not, update graphics drivers.

### Issue: "CUDA not available" (but you have GPU)

**Solution:** Install PyTorch with CUDA:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Check CUDA availability:
```python
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Issue: "ModuleNotFoundError"

**Solution:** Make sure virtual environment is activated:
```bash
# Check if activated
which python   # Linux/Mac
where python   # Windows

# Should show path to venv/bin/python or venv\Scripts\python
```

If not activated, run:
```bash
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### Issue: "Permission denied: setup.sh"

**Solution (Linux/Mac):**
```bash
chmod +x setup.sh
chmod +x activate.sh
./setup.sh
```

---

## Manual Installation (Alternative)

If automated setup fails, install manually:

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# 3. Upgrade pip
pip install --upgrade pip setuptools wheel

# 4. Install dependencies
pip install -r backend/requirements.txt

# 5. Create directories
mkdir -p data checkpoints logs reports

# 6. Create .env file (copy from .env.example or see docs)
cp .env.example .env  # Linux/Mac
copy .env.example .env  # Windows

# 7. Test installation
python -c "import torch; import fastapi; import gymnasium; print('OK')"
```

---

## Docker Installation (Alternative)

Don't want to deal with Python/venv? Use Docker:

```bash
# Build and run
docker-compose up --build

# Access at:
# - API: http://localhost:8000
# - Frontend: http://localhost:8080
```

No virtual environment needed with Docker!

---

## Next Steps

After installation:

1. **Read the docs:**
   - `README.md` - Full documentation
   - `QUICKSTART.md` - Quick start guide

2. **Try the demo:**
   ```bash
   python run.py api
   # Open frontend/index.html in browser
   ```

3. **Generate training data** (optional):
   ```bash
   python run.py generate-data --episodes 1000
   ```

4. **Train models** (optional):
   ```bash
   python run.py train-cnn
   python run.py train-rl
   ```

---

## Uninstallation

To completely remove:

```bash
# 1. Deactivate virtual environment (if active)
deactivate

# 2. Remove virtual environment
rm -rf venv  # Linux/Mac
rmdir /s venv  # Windows

# 3. Remove generated data (optional)
rm -rf data checkpoints logs reports

# 4. Remove the project folder
cd ..
rm -rf rl_model_cdss
```

---

## Need Help?

1. Check `README.md` for detailed documentation
2. Run `python run.py --help` to see all commands
3. Look at test files in `backend/tests/` for usage examples
4. Check logs in `logs/` directory if things go wrong

---

## Remember

⚠️ **This is a RESEARCH PROTOTYPE**
- NOT for clinical use
- NOT for patient care
- For research and simulation ONLY

---

**Estimated disk space needed:**
- Virtual environment: ~2-3 GB
- Dependencies: ~2 GB
- Generated data (optional): ~1-10 GB depending on dataset size
- **Total: ~5-15 GB**

**Estimated time:**
- Setup: 5-10 minutes
- Data generation: 10-60 minutes (optional)
- CNN training: 20-120 minutes (optional)
- RL training: 2-6 hours (optional)

