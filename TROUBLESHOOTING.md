# Troubleshooting Guide

This guide covers all common errors and their solutions.

---

## Table of Contents

1. [Model Loading Errors](#model-loading-errors)
2. [Simulation Start Errors](#simulation-start-errors)
3. [WebSocket Connection Issues](#websocket-connection-issues)
4. [Windows-Specific Issues](#windows-specific-issues)
5. [Port Conflicts](#port-conflicts)

---

## Model Loading Errors

### Error: "Failed to load model: HTTP error! status: 500"

**Symptoms**: Clicking "Load Model" button shows HTTP 500 error

**Cause**: Backend tried to download model from invalid URL or file doesn't exist

**Solution**:
```
1. Use the Duck model first to test:
   - Click "Click here" link below URL field
   - This loads: https://raw.githubusercontent.com/.../Duck.gltf
   - Should work in ~2 seconds

2. For local models:
   - Ensure file exists: frontend/models/digestive/source/scene.gltf
   - Ensure textures folder exists in same directory
   - URL should be: models/digestive/source/scene.gltf (no leading slash)

3. Check browser console (F12) for detailed errors
```

**Fixed in Version 2.0**: Frontend now loads models directly with Three.js, no longer depends on backend for model loading.

---

### Error: "Failed to load model: undefined"

**Symptoms**: Generic error message when loading model

**Cause**: CORS issue or Three.js GLTFLoader not initialized

**Solution**:
```
1. Check browser console (F12) for CORS errors
2. Ensure GLTFLoader script is loaded:
   - View page source
   - Look for: <script src="...GLTFLoader.js"></script>
   
3. Clear browser cache: Ctrl+Shift+R

4. Verify model file accessibility:
   - Open: http://localhost:3000/models/digestive/source/scene.gltf
   - Should show JSON or download file
```

**Prevention**: Always serve models from the same origin as the frontend (no cross-origin loading).

---

## Simulation Start Errors

### Error: "Failed to start simulation: HTTP error! status: 400"

**Symptoms**: Model loads successfully, but clicking "Start Simulation" shows 400 error

**Cause**: Backend wasn't notified about loaded model

**Solution**:
```
1. REFRESH browser (Ctrl+Shift+R) to get latest code

2. Load model again:
   - Click "Load Model"
   - Wait for success message
   - Check browser console for: "Backend notified about model"

3. Now click "Start Simulation"
   - Should work without 400 error

4. If still failing:
   - Open: http://localhost:8000/api/models/info
   - Check "gltf_url" field is not null
   - If null, backend didn't receive notification
```

**Technical Details**:
```javascript
// Frontend now converts relative URLs to absolute:
models/digestive/source/scene.gltf
  → http://localhost:3000/models/digestive/source/scene.gltf

// Backend receives and stores this URL
// Simulation can now start successfully
```

**Fixed in Version 2.0**: Frontend always notifies backend about model loading, converting relative URLs to absolute.

---

### Error: "Failed to start simulation: HTTP error! status: 500"

**Symptoms**: Start simulation returns 500 error with detailed message

**Cause**: Backend environment initialization failed

**Common Causes**:
1. Missing configuration file
2. Invalid GLTF URL
3. Renderer initialization failure

**Solution**:
```bash
# 1. Check configuration exists
ls configs/sim.yaml

# 2. Check backend logs (in terminal where backend is running)
# Look for detailed error message

# 3. Verify renderer status
curl http://localhost:8000/api/health

# 4. On Windows, expect this warning (it's OK):
# "Warning: pyrender not available... 3D rendering will be disabled"
```

**On Windows**: Simulation will use placeholder images instead of 3D rendering. This is expected and OK for testing!

---

## WebSocket Connection Issues

### Error: "WebSocket connection failed"

**Symptoms**: Simulation starts but no video feed appears

**Cause**: WebSocket not connecting to backend

**Solution**:
```
1. Check backend is running:
   curl http://localhost:8000/api/health

2. Check WebSocket URL in browser console:
   Should be: ws://localhost:8000/api/stream

3. Verify no firewall blocking WebSocket

4. Check browser DevTools → Network → WS tab
   - Should show WebSocket connection
   - Status: 101 Switching Protocols

5. If using Docker:
   - Ensure ports are mapped: "8000:8000"
```

**Test WebSocket manually**:
```javascript
// In browser console:
const ws = new WebSocket('ws://localhost:8000/api/stream');
ws.onopen = () => console.log('Connected!');
ws.onerror = (e) => console.error('Error:', e);
```

---

## Windows-Specific Issues

### Warning: "pyrender not available... 3D rendering will be disabled"

**Status**: ⚠️ Expected on Windows (not an error!)

**Explanation**:
- pyrender requires OpenGL/OSMesa libraries
- These are not available on Windows by default
- Backend falls back to placeholder images

**Impact**:
- ✅ Frontend 3D viewer still works (Three.js)
- ✅ Simulation still runs
- ✅ WebSocket streaming still works
- ⚠️ Backend renders placeholder images instead of actual 3D frames

**Solutions**:

**Option 1: Use Docker (Recommended)**
```bash
docker-compose up
# Linux container has all required libraries
```

**Option 2: Use WSL2**
```bash
# In WSL2 terminal:
cd /mnt/c/Users/Audry/Downloads/rl_model_cdss
source venv/bin/activate
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

**Option 3: Accept Limitation**
```
For development/testing, placeholder images are OK!
For production/research, use Linux deployment.
```

---

### Error: "UnicodeEncodeError: 'charmap' codec can't encode characters"

**Symptoms**: Backend crashes on startup with Unicode error

**Cause**: Windows console doesn't support Unicode characters (⚠️, ✓, etc.)

**Solution**: Already fixed in Version 2.0!

**If you see this error**:
```bash
# Pull latest code:
git pull origin main

# Or manually edit backend/api/main.py:
# Replace Unicode characters with ASCII equivalents
```

**Fixed Files**:
- `backend/api/main.py` (startup banner)
- `run.py` (print statements)

---

### Error: "error: Microsoft Visual C++ 14.0 or greater is required"

**Symptoms**: `pip install` fails for `noise` package

**Cause**: `noise` requires C++ compiler

**Solution**: Already fixed in Version 2.0!

**Technical Details**:
- Removed `noise` from `requirements.txt`
- Implemented `simple_perlin()` function in Python
- No C++ compiler needed

---

## Port Conflicts

### Error: "error while attempting to bind... address already in use"

**Full Error**:
```
[Errno 10048] error while attempting to bind on address ('0.0.0.0', 8000):
only one usage of each socket address is normally permitted
```

**Cause**: Another process is using port 8000 or 3000

**Solution**:

**Find and Kill Process**:
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill process (replace <PID> with actual PID)
taskkill /PID <PID> /F

# Or kill all Python processes:
taskkill /IM python.exe /F
```

**Change Port**:
```bash
# Use different port for backend
python -m uvicorn backend.api.main:app --port 8001

# Update frontend API_BASE_URL in app.js:
const API_BASE_URL = 'http://localhost:8001/api';
```

**Prevention**:
```bash
# Always stop servers properly with Ctrl+C
# Don't close terminal window while server running
```

---

## Quick Diagnostic Commands

### Check System Status

```bash
# 1. Check backend health
curl http://localhost:8000/api/health

# Expected: {"status":"ok","version":"1.0.0",...}

# 2. Check frontend accessibility
curl http://localhost:3000/index.html

# Expected: HTML content

# 3. Check model file
curl http://localhost:3000/models/digestive/source/scene.gltf

# Expected: JSON with "asset", "scene", etc.

# 4. Check backend logs
# Look at terminal where backend is running
# Should show: "INFO: Uvicorn running on..."

# 5. Check if ports are listening
netstat -ano | findstr :8000
netstat -ano | findstr :3000
```

---

## Error Priority Checklist

When encountering errors, check in this order:

### 1. Servers Running?
```bash
curl http://localhost:8000/api/health  # Backend
curl http://localhost:3000             # Frontend
```

### 2. Virtual Environment Activated?
```bash
# Should see (venv) in prompt
# If not:
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 3. Dependencies Installed?
```bash
pip list | findstr torch
pip list | findstr fastapi
# Should show installed packages
```

### 4. Browser Cache Cleared?
```
Press: Ctrl+Shift+R (or Cmd+Shift+R on Mac)
Or: Open DevTools (F12) → Network → Check "Disable cache"
```

### 5. Files in Correct Location?
```bash
# Check project structure
ls frontend/index.html
ls frontend/models/digestive/source/scene.gltf
ls configs/sim.yaml
ls backend/requirements.txt
```

---

## Common Error Messages Reference

| Error Message | Location | Fix |
|---------------|----------|-----|
| "HTTP error! status: 500" | Frontend (model loading) | Check model file exists, try Duck model |
| "HTTP error! status: 400" | Frontend (start sim) | Refresh browser, reload model |
| "pyrender not available" | Backend (startup) | Expected on Windows, use Docker for full rendering |
| "address already in use" | Backend (startup) | Kill process on port 8000 |
| "No GLTF model loaded" | Backend (start sim) | Load model first via frontend |
| "UnicodeEncodeError" | Backend (startup) | Update to Version 2.0 |
| "CORS policy" | Browser console | Serve model from same origin (localhost:3000) |
| "WebSocket connection failed" | Frontend (simulation) | Check backend running, check firewall |

---

## Getting Help

### Self-Help Resources
1. **`SYSTEM_STATUS.md`** - Current status & quick reference
2. **`ENHANCED_SYSTEM.md`** - Feature documentation
3. **`FIXES_AND_ENHANCEMENTS.md`** - Detailed change log
4. **`README.md`** - Setup instructions

### Diagnostic Information to Provide

When asking for help, provide:

```bash
# 1. System info
echo "OS: Windows/Linux/Mac"
echo "Python: $(python --version)"

# 2. Backend status
curl http://localhost:8000/api/health

# 3. Browser console errors (F12 → Console)
# Screenshot or copy/paste errors

# 4. Backend terminal output
# Last 20 lines from terminal where backend is running

# 5. What you were trying to do
# Step-by-step actions taken
```

---

## Known Limitations

### Windows
- ❌ Full 3D rendering (pyrender) not available
- ✅ Frontend 3D viewer works
- ✅ Simulation works with placeholder frames
- **Recommended**: Use Docker or WSL2 for full functionality

### Model Loading
- ⚠️ Very large models (>100MB) may take 30+ seconds
- ⚠️ Some Sketchfab models have complex texture setups
- **Recommended**: Test with Duck model first

### Performance
- ⚠️ Without GPU: Training will be slow
- ⚠️ CPU inference: ~10 FPS max
- **Recommended**: Use GPU for training, CPU OK for testing

---

## Version-Specific Issues

### Version 1.0 (Original)
- ❌ Model loading failed with local files (CORS)
- ❌ Simulation required backend model loading
- ❌ Unicode errors on Windows
- ❌ Required C++ compiler (noise library)

### Version 2.0 (Current)
- ✅ Model loading works with local files
- ✅ Frontend loads models directly
- ✅ No Unicode characters
- ✅ No C++ compiler needed
- ✅ Enhanced multi-disease detection system

**Recommended**: Always use Version 2.0 or later!

---

## Emergency Reset

If nothing works, try a clean restart:

```bash
# 1. Stop all processes
taskkill /IM python.exe /F

# 2. Clear Python cache
rm -rf backend/__pycache__
rm -rf backend/**/__pycache__

# 3. Restart servers
# Terminal 1:
cd C:\Users\Audry\Downloads\rl_model_cdss
.\venv\Scripts\activate
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000

# Terminal 2:
cd C:\Users\Audry\Downloads\rl_model_cdss
python -m http.server 3000 --directory frontend

# 4. Open fresh browser session
# Incognito/Private mode to avoid cache issues
start chrome --incognito http://localhost:3000
```

---

**Last Updated**: 2025-10-27  
**Version**: 2.0

For more help, see `SYSTEM_STATUS.md` or open an issue on GitHub.

