# Using Sketchfab 3D Models

## Digestive System Model

The recommended model for this simulator is the [Digestive System model by unlim3d](https://skfb.ly/6zqtp) on Sketchfab.

## How to Use Sketchfab Models

### Option 1: Download Model (Recommended)

1. Visit the model page: https://skfb.ly/6zqtp
2. Click "Download 3D Model" (requires free Sketchfab account)
3. Select "glTF" format
4. Download and extract the files
5. Host the files on a web server or use a service like:
   - GitHub Pages
   - Netlify
   - Your own web server
6. Copy the URL to the `.gltf` file
7. Paste it into the simulator

### Option 2: Use Alternative Models

If you can't access the Sketchfab model, try these free GLTF models:

**Anatomical Models:**
- Brain Stem: `https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/BrainStem/glTF/BrainStem.gltf`
- Avocado (simple test): `https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Avocado/glTF/Avocado.gltf`

**Test Models:**
- Duck: `https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Duck/glTF/Duck.gltf`
- Box: `https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Box/glTF/Box.gltf`

### Option 3: Embedded Sketchfab Viewer

While the simulator loads the model in Three.js, you can also:

1. Use Sketchfab's embed API (requires API key)
2. Reference: https://sketchfab.com/developers

## Model Requirements

- **Format**: GLTF (.gltf) or GLB (.glb)
- **Size**: Under 50MB recommended for web viewing
- **Licensing**: Ensure you have rights to use the model
- **CORS**: Model must be hosted on a CORS-enabled server

## Hosting Your Own Models

### Using GitHub Pages (Free)

1. Create a new GitHub repository
2. Upload your GLTF files to a folder (e.g., `models/`)
3. Enable GitHub Pages in repository settings
4. Access via: `https://yourusername.github.io/yourrepo/models/model.gltf`

### Using Python HTTP Server (Local Testing)

```bash
# Navigate to your model directory
cd path/to/models

# Start server
python -m http.server 8080

# Access via: http://localhost:8080/model.gltf
```

## Troubleshooting

### CORS Errors
If you see CORS errors in the browser console:
- Ensure the server hosting the model has CORS enabled
- Use a CORS proxy for testing: `https://cors-anywhere.herokuapp.com/`
- Host the model on your own server with proper CORS headers

### Model Not Loading
- Check browser console (F12) for errors
- Verify the URL is accessible in a new browser tab
- Ensure the file is actual GLTF format
- Try a simpler model first to test

### Model Appears Black or Invisible
- The model may not have materials/textures
- Try rotating the camera (left click + drag)
- Try zooming out (mouse wheel)
- Check if model has proper lighting

## License Compliance

The Digestive System model on Sketchfab has specific licensing:
- Check the model's license on Sketchfab
- Attribute the creator (unlim3d) if required
- Follow CC license terms if applicable
- For research use only (as per this simulator's purpose)

## Model Information

**Digestive System by unlim3d:**
- Polygons: 35.6k triangles
- Vertices: 17.8k
- High-resolution 4K textures
- Sub-surface scattering materials
- Anatomically accurate for medical presentations

---

**Need Help?**
- See `FRONTEND_3D_GUIDE.md` for general 3D visualization help
- See `README.md` for complete setup instructions

