# Troubleshooting HTTP 500 Error

## 🔴 Current Issue

**Error**: "Failed to load model: HTTP error! status: 500"  
**Cause**: The Python HTTP server is having trouble serving the model file

## 🧪 Testing Strategy

### Step 1: Test with Simple Model (CURRENT)

I've temporarily switched the frontend to load a **Duck model** from GitHub to test if the Three.js loading mechanism works.

**Action**: 
1. Refresh your browser (F5)
2. Click "Load Model"
3. The Duck should load in ~2 seconds

**If Duck Loads**: ✅ Three.js is working! Issue is with local file serving.  
**If Duck Fails**: ❌ Problem with Three.js setup or browser.

### Step 2: Check Browser Console

Open Developer Tools to see detailed errors:
1. Press `F12` in your browser
2. Click the **Console** tab
3. Look for red error messages
4. Copy any error text you see

### Step 3: Test Direct File Access

Try accessing the model file directly in your browser:

**Open this URL**: http://localhost:3000/models/digestive/source/scene.gltf

**Expected**: You should see JSON text (the GLTF data)  
**If 500 Error**: The server can't serve this file

## 🔧 Possible Fixes

### Fix 1: Restart Frontend Server

The Python HTTP server might be having issues:

```powershell
# Stop the current server (Ctrl+C in the terminal)
# Then restart it:
cd C:\Users\Audry\Downloads\rl_model_cdss
python -m http.server 3000 --directory frontend
```

### Fix 2: Check File Permissions

Windows might be blocking the files:

```powershell
# Right-click the rl_model_cdss folder
# Properties → Security → Edit
# Make sure your user has "Read" permissions
```

### Fix 3: Use a Different Port

Sometimes port 3000 has issues:

```powershell
# Try port 8080 instead:
python -m http.server 8080 --directory frontend

# Then open: http://localhost:8080
```

### Fix 4: Simplify Model Location

Move model to root of frontend:

```powershell
cd C:\Users\Audry\Downloads\rl_model_cdss
Copy-Item "frontend\models\digestive\source\scene.gltf" "frontend\scene.gltf" -Force
Copy-Item "frontend\models\digestive\source\scene.bin" "frontend\scene.bin" -Force
Copy-Item "frontend\models\digestive\source\textures" "frontend\textures" -Recurse -Force
```

Then change the URL in the browser to: `scene.gltf`

## 🔍 Common Causes of HTTP 500

### 1. File Path Issues
- **Symptom**: Can't find file
- **Fix**: Verify file exists at exact path
- **Check**: http://localhost:3000/models/digestive/source/scene.gltf

### 2. Special Characters in Filenames
- **Symptom**: Files with `@` or special chars fail
- **Fix**: Rename files to simpler names
- **Example**: Our textures have `@channels=` in names

### 3. File Size Too Large
- **Symptom**: Large files timeout
- **Fix**: Use smaller model or different server
- **Our case**: scene.gltf is only 7KB, should be fine

### 4. Server Permissions
- **Symptom**: Server can't read files
- **Fix**: Check Windows file permissions
- **Check**: Right-click folder → Properties → Security

### 5. Server Process Issues
- **Symptom**: Server is confused or stuck
- **Fix**: Restart the server
- **Action**: Ctrl+C then restart

## 📊 Diagnostic Checklist

Run these checks:

### Check 1: File Exists
```powershell
Test-Path "C:\Users\Audry\Downloads\rl_model_cdss\frontend\models\digestive\source\scene.gltf"
# Should return: True
```

### Check 2: File is Readable
```powershell
Get-Content "C:\Users\Audry\Downloads\rl_model_cdss\frontend\models\digestive\source\scene.gltf" -TotalCount 1
# Should show JSON text
```

### Check 3: Server is Running
Open http://localhost:3000 in browser - should show the dashboard

### Check 4: Port is Correct
Check the terminal where you started the server - it should say "Serving HTTP on :: port 3000"

## 🎯 Quick Fix Attempts

### Attempt 1: Restart Everything
1. Stop frontend server (Ctrl+C)
2. Close browser
3. Restart frontend server
4. Open browser fresh
5. Try loading Duck model first

### Attempt 2: Use External Model
Since the Duck model from GitHub should work, use it:
1. URL: `https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Duck/glTF/Duck.gltf`
2. This proves Three.js works
3. Then we can debug the local file issue

### Attempt 3: Alternative Model Host
Use GLB instead of GLTF (single file, no external textures):
1. Download GLB version from Sketchfab
2. Place in `frontend/models/`
3. Load with `.glb` extension

## 🚀 Alternative: Use Online Models

While we debug the local files, you can use these working URLs:

### Anatomical Models
```
Brain Stem:
https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/BrainStem/glTF/BrainStem.gltf

Avocado (test):
https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Avocado/glTF/Avocado.gltf
```

### Test Models
```
Duck (simple):
https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Duck/glTF/Duck.gltf

Box (minimal):
https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Box/glTF/Box.gltf
```

## 📝 What to Report

If issues persist, please share:

1. **Browser console errors** (F12 → Console tab)
2. **Server terminal output** (any error messages)
3. **Which models work/don't work**:
   - Duck from GitHub: ❓
   - Local digestive system: ❓
4. **Direct file access test**: Does http://localhost:3000/models/digestive/source/scene.gltf work?

## ✅ Expected Working State

Once fixed, you should see:
- ✅ Duck model loads (proof Three.js works)
- ✅ Can access scene.gltf directly in browser
- ✅ Digestive system model loads from local files
- ✅ Model appears in 3D viewport
- ✅ Can rotate/zoom the model

## 🔄 Next Steps

**Right Now**:
1. Refresh browser
2. Try loading the Duck model
3. Report if it works or not

**If Duck Works**:
- We'll fix the local file serving issue
- Might need to restructure the model files
- Or use a different serving method

**If Duck Fails Too**:
- Check browser console for JavaScript errors
- Verify Three.js and GLTFLoader are loading
- Check browser compatibility

---

**Current Status**: Testing with Duck model to isolate the issue.  
**Action Required**: Refresh browser and try loading the Duck model.

