# Docker Quick Start - Full 3D Rendering

## ⚠️ **Why Docker is REQUIRED for Full Functionality**

### **The Vision: What This System Does**

This simulator is designed to:

1. **Load a 3D anatomical model** (digestive system from Sketchfab)
2. **Procedurally generate synthetic diseases** (ulcers, H. pylori, tumors) ON the 3D mesh surfaces
3. **Navigate a virtual camera INSIDE the 3D model** (like a real endoscope)
4. **Render realistic endoscopy frames** showing mucosal tissue and lesions
5. **Train an RL agent** to detect diseases based on what it "sees"

### **The Problem: Windows Limitations**

❌ **Windows Cannot Run Full 3D Rendering**
- pyrender requires OpenGL/OSMesa libraries
- These are not available on Windows natively
- Without them:
  - No 3D model rendering
  - No lesion visualization  
  - No realistic camera views
  - RL agent is "blind" (placeholder images)
  - **The simulation is non-functional**

✅ **Docker Solves This**
- Runs Linux container with all required libraries
- Full OpenGL/OSMesa support
- Proper 3D rendering
- RL agent can actually see and learn

---

## 🚀 Quick Start with Docker

### Step 1: Install Docker Desktop

```bash
# Download Docker Desktop for Windows:
https://www.docker.com/products/docker-desktop/

# Install and start Docker Desktop
# Wait for it to fully start (Docker icon in system tray should be green)
```

### Step 2: Build and Run

```bash
# Navigate to project directory
cd C:\Users\Audry\Downloads\rl_model_cdss

# Build and start all services
docker-compose up --build
```

**What this does:**
- Builds backend with OpenGL/OSMesa libraries
- Installs all Python dependencies
- Starts backend API (port 8000)
- Starts frontend server (port 3000)
- Creates Docker network for communication

**First build takes:** 5-10 minutes (downloads base images, installs libraries)

**Subsequent starts:** ~30 seconds

### Step 3: Access the System

Once you see:
```
backend   | INFO:     Uvicorn running on http://0.0.0.0:8000
frontend  | Serving HTTP on :: port 3000 (http://[::]:3000/) ...
```

Open your browser:
```
http://localhost:3000
```

---

## 🎯 What You'll See (With Docker)

### 1. Load Model
```
- URL: models/digestive/source/scene.gltf
- Click: "Load Model"
- Wait: 5-10 seconds
```

**Expected:**
- ✅ Model loads in frontend 3D viewer (Three.js)
- ✅ Backend loads model with trimesh
- ✅ pyrender scene initialized
- ✅ Ready for simulation

### 2. Start Simulation
```
- Click: "Start Simulation"
```

**Expected:**
- ✅ Simulation starts successfully
- ✅ **REAL 3D frames** in video feed (not placeholders!)
- ✅ Camera navigating inside GI tract
- ✅ Synthetic lesions visible on tissue
- ✅ Metrics update: Step, Reward, Coverage
- ✅ RL agent learns from visual frames

### 3. What You're Seeing

**Video Feed:**
- Real-time rendered frames from virtual endoscope
- Mucosal tissue texture
- Synthetic lesions (red/inflamed regions)
- Depth and lighting effects
- Realistic endoscopy appearance

**3D Viewer:**
- Camera trajectory through GI tract
- Current camera position marked
- Full digestive system model

**Metrics:**
- Step count increases
- Coverage % grows as agent explores
- Reward based on:
  - New areas explored (+coverage)
  - Lesions detected (+detection)
  - Collisions avoided (-penalty)

---

## 🔧 Docker Commands

### Start Services
```bash
# Start (detached mode)
docker-compose up -d

# Start with rebuild
docker-compose up --build

# View logs
docker-compose logs -f

# View backend logs only
docker-compose logs -f backend
```

### Stop Services
```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Restart After Code Changes
```bash
# Rebuild backend only
docker-compose build backend

# Restart backend
docker-compose restart backend
```

### Access Container Shell
```bash
# Backend shell
docker exec -it hpylori-rl-backend bash

# Check if pyrender works
docker exec -it hpylori-rl-backend python -c "import pyrender; print('pyrender OK')"
```

---

## 📊 Comparison: Windows vs Docker

| Feature | Windows (Native) | Docker (Linux) |
|---------|------------------|----------------|
| **3D Model Loading** | ✅ Frontend only | ✅ Frontend + Backend |
| **3D Rendering** | ❌ Placeholder images | ✅ Real rendered frames |
| **Lesion Synthesis** | ❌ Not visible | ✅ Visible on mesh |
| **Camera Navigation** | ⚠️ Logic only | ✅ Visual + Logic |
| **RL Training** | ❌ Blind agent | ✅ Agent sees frames |
| **Research Usable** | ❌ No | ✅ Yes |

---

## 🧪 Verify Full Functionality

### Test 1: Check pyrender Availability

```bash
# Inside Docker backend
docker exec -it hpylori-rl-backend python -c "
import pyrender
print('pyrender: OK')
import trimesh
print('trimesh: OK')
import cv2
print('cv2: OK')
"
```

**Expected Output:**
```
pyrender: OK
trimesh: OK
cv2: OK
```

### Test 2: Check Model Loading

```bash
# Check backend logs after loading model
docker-compose logs backend | grep -i "loaded"
```

**Expected Output:**
```
Successfully loaded GLTF model: 45234 vertices, 89032 faces
```

### Test 3: Check Rendering

```bash
# After starting simulation, check logs
docker-compose logs backend | grep -i "render"
```

**Expected Output:**
```
Rendering frame at step 1
Rendering frame at step 2
...
```

(Should NOT see: "3D Rendering Unavailable" or "pyrender not available")

---

## 🐛 Troubleshooting

### Error: "Cannot connect to Docker daemon"

**Solution:**
```bash
# Start Docker Desktop
# Wait for green icon in system tray
# Try again
```

### Error: "Port 8000 already in use"

**Solution:**
```bash
# Stop native Python servers
taskkill /F /IM python.exe

# Try docker-compose again
docker-compose up
```

### Error: "pyrender not available" (in Docker logs)

**Solution:**
```bash
# Rebuild with updated Dockerfile
docker-compose down
docker-compose build --no-cache backend
docker-compose up
```

### Container Runs But Can't Load Model

**Solution:**
```bash
# Check if models directory is accessible
docker exec -it hpylori-rl-backend ls -la /app/frontend/models/

# If not found, model files may not be mounted
# Ensure frontend volume is mounted correctly in docker-compose.yml
```

---

## 📈 Training with Docker

### Train CNN (Inside Container)

```bash
# Enter backend container
docker exec -it hpylori-rl-backend bash

# Generate synthetic data
python -m backend.utils.bootstrap_data --n-frames 10000

# Train CNN
python -m backend.models.cnn.train_cnn --epochs 30

# Exit container
exit
```

### Train RL Policy (Inside Container)

```bash
docker exec -it hpylori-rl-backend bash

# Train basic PPO
python -m backend.models.rl.train_rl --total-timesteps 500000

# Train enhanced PPO (multi-disease)
python -m backend.models.rl.train_rl_enhanced \
    --env-id EndoscopyEnhanced-v0 \
    --total-timesteps 1000000 \
    --use-cnn
```

### Access Trained Models

Models are saved to mounted volumes:
```
C:\Users\Audry\Downloads\rl_model_cdss\checkpoints\
```

---

## 🎓 Why This Matters for Research

### Without Docker (Windows Native)
- ❌ Cannot train RL agents (no visual input)
- ❌ Cannot test navigation algorithms
- ❌ Cannot evaluate detection performance
- ❌ Cannot publish meaningful results

### With Docker (Full Rendering)
- ✅ Train RL agents on realistic frames
- ✅ Test navigation in 3D anatomical space
- ✅ Evaluate detection sensitivity/precision
- ✅ Publish research-grade results
- ✅ Compare to real endoscopy data

---

## 🚀 Alternative: WSL2 (Without Docker Desktop)

If you don't want to use Docker Desktop:

```bash
# 1. Enable WSL2 on Windows
wsl --install

# 2. Install Ubuntu
wsl --install -d Ubuntu

# 3. Enter WSL
wsl

# 4. Navigate to project (from WSL)
cd /mnt/c/Users/Audry/Downloads/rl_model_cdss

# 5. Install dependencies
sudo apt-get update
sudo apt-get install python3.11 python3-pip libosmesa6

# 6. Create venv and install requirements
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

# 7. Run backend (with full rendering)
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000

# 8. Access from Windows browser
# Frontend: http://localhost:3000 (run separately on Windows)
# Backend:  http://localhost:8000 (running in WSL)
```

---

## 📝 Summary

| Approach | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| **Windows Native** | Easy setup | ❌ No 3D rendering | ⚠️ Development only |
| **Docker** | Full functionality | Requires Docker Desktop | ✅ **Recommended** |
| **WSL2** | Native Linux | More complex setup | ✅ Good alternative |

---

## 🎯 Final Checklist

Before starting research/training, verify:

- [ ] Docker containers running (`docker ps`)
- [ ] Backend has pyrender (`docker exec ... python -c "import pyrender"`)
- [ ] Model loads successfully (check logs)
- [ ] Simulation starts without errors
- [ ] Video feed shows **real rendered frames** (not placeholders)
- [ ] Lesions are visible on tissue
- [ ] Metrics update correctly

**If all checked:** ✅ System ready for research!

---

**Last Updated:** 2025-10-27  
**Version:** 2.0 (Full 3D Rendering with Docker)

