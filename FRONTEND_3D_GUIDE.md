# 3D Frontend Guide

## Overview
The frontend now has **full 3D visualization** capabilities using Three.js, which runs directly in your browser. This bypasses the Windows OpenGL limitations on the backend.

## What's Working

### ✅ Full 3D Visualization
- The frontend loads and displays GLTF/GLB 3D models directly in the browser
- Interactive camera controls (mouse to rotate, zoom, pan)
- Real-time camera path visualization
- Live metrics and anomaly detection from the backend

### ✅ Backend Integration
- Backend provides RL guidance and CNN anomaly predictions
- WebSocket streaming for live data
- API endpoints for model management and simulation control

## How to Use

### 1. Start Both Servers

**Backend (in one terminal):**
```powershell
cd C:\Users\Audry\Downloads\rl_model_cdss
.\venv\Scripts\activate
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

**Frontend (in another terminal):**
```powershell
cd C:\Users\Audry\Downloads\rl_model_cdss
python -m http.server 3000 --directory frontend
```

### 2. Access the Dashboard

Open your browser to: **http://localhost:3000**

### 3. Load a 3D Model

**Option A: Use the default cube (for testing)**
- The input field already has a default cube model URL
- Just click "Load Model"

**Option B: Use a custom model**
- Find a GLTF/GLB model (e.g., from Sketchfab with proper license)
- Paste the URL into the "GLTF Model URL" field
- Click "Load Model"

**Good Test Models:**
- Cube: `https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Cube/glTF/Cube.gltf`
- Duck: `https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Duck/glTF/Duck.gltf`
- Brain Stem: `https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/BrainStem/glTF/BrainStem.gltf`

### 4. Start Simulation

- Once the model loads, you'll see it in the 3D viewport
- Click "▶️ Start Simulation" to begin
- The backend will stream:
  - CNN anomaly probabilities
  - RL action suggestions
  - Camera position and coverage metrics

### 5. Interact with 3D View

- **Rotate**: Left mouse button + drag
- **Zoom**: Mouse wheel
- **Pan**: Right mouse button + drag (if OrbitControls is enabled)

## Features

### 3D Viewport (Left Side)
- Interactive 3D model visualization
- Camera path visualization (blue line showing trajectory)
- Current camera position marker (red sphere)
- Grid and axes for reference

### Video Feed (Top Right)
- Live frames from backend (placeholder images on Windows without OpenGL)
- Can be replaced with rendered views when running on Linux

### Metrics Panels (Right Side)
- **CNN Anomaly Detection**: Probability gauge and status
- **RL Guidance**: Suggested actions and rewards
- **Camera Pose**: Position and rotation
- **Performance**: FPS and frame count
- **Coverage**: How much of the surface has been explored

## Known Limitations

### Windows Backend
- `pyrender` doesn't work on Windows without proper OpenGL drivers
- The video feed shows placeholder images
- However, the 3D visualization in the browser works perfectly!

### Workarounds
1. **Use browser 3D** (current solution): All 3D visualization happens in the browser
2. **Use Docker**: Run backend in Linux container with GPU support
3. **Use WSL2**: Run backend in Windows Subsystem for Linux

## Architecture

```
┌─────────────────┐
│   Browser       │
│  (Three.js)     │ ← Full 3D rendering here
│                 │
│  - Loads GLTF   │
│  - Shows path   │
│  - Interactive  │
└────────┬────────┘
         │ WebSocket
         │ (metrics only)
┌────────▼────────┐
│   Backend API   │
│                 │
│  - RL Policy    │ ← Provides guidance
│  - CNN Model    │ ← Detects anomalies
│  - Simulation   │
└─────────────────┘
```

## Troubleshooting

### "Cannot connect to API server"
- Make sure backend is running on port 8000
- Check: http://localhost:8000/docs

### "Model failed to load"
- Ensure the URL is accessible
- Check CORS permissions
- Try one of the test models above

### "Three.js not defined"
- Check browser console for errors
- Ensure you're accessing via http://localhost:3000 (not file://)

### No 3D model visible
- Check browser console (F12)
- Try zooming out (mouse wheel)
- Try the default cube model first

## Next Steps

1. **Train CNN Model**: Generate synthetic frames and train anomaly detector
   ```powershell
   python backend/models/cnn/train_cnn.py
   ```

2. **Train RL Policy**: Train navigation policy
   ```powershell
   python backend/models/rl/train_rl.py
   ```

3. **Test with Medical Models**: Use anatomically accurate GI tract models

## Support

For issues or questions, check:
- `README.md` - Main documentation
- `QUICKSTART.md` - Quick setup guide
- Browser console (F12) - JavaScript errors
- Backend logs - Python errors

---

**⚠️ RESEARCH PROTOTYPE - NOT A MEDICAL DEVICE - FOR SIMULATION ONLY ⚠️**

