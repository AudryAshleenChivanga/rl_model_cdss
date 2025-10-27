// Frontend Application Logic
// API Configuration
const API_BASE_URL = 'http://localhost:8000/api';
const WS_URL = 'ws://localhost:8000/api/stream';

// Global State
let ws = null;
let isStreaming = false;
let frameCount = 0;
let lastFrameTime = Date.now();
let fps = 0;
let sessionMetrics = [];
let threeScene = null;
let threeRenderer = null;
let threeCamera = null;
let cameraPath = [];
let loadedModel = null;
let gltfLoader = null;
let orbitControls = null;

// DOM Elements
const elements = {
    // Buttons
    loadModelBtn: document.getElementById('load-model-btn'),
    startBtn: document.getElementById('start-btn'),
    stopBtn: document.getElementById('stop-btn'),
    resetBtn: document.getElementById('reset-btn'),
    downloadBtn: document.getElementById('download-btn'),
    
    // Inputs
    gltfUrl: document.getElementById('gltf-url'),
    
    // Status
    connectionStatus: document.getElementById('connection-status'),
    simStatus: document.getElementById('sim-status'),
    modelStatus: document.getElementById('model-status'),
    
    // Video
    videoCanvas: document.getElementById('video-canvas'),
    loadingOverlay: document.getElementById('loading-overlay'),
    
    // Metrics
    stepValue: document.getElementById('step-value'),
    fpsValue: document.getElementById('fps-value'),
    coverageValue: document.getElementById('coverage-value'),
    cnnProb: document.getElementById('cnn-prob'),
    cnnGaugeFill: document.getElementById('cnn-gauge-fill'),
    cnnStatus: document.getElementById('cnn-status'),
    actionName: document.getElementById('action-name'),
    actionId: document.getElementById('action-id'),
    stepReward: document.getElementById('step-reward'),
    totalReward: document.getElementById('total-reward'),
    positionValue: document.getElementById('position-value'),
    rotationValue: document.getElementById('rotation-value'),
    collisionValue: document.getElementById('collision-value'),
    framesCount: document.getElementById('frames-count'),
};

// Initialize Three.js Scene
function initThreeJS() {
    const container = document.getElementById('threejs-container');
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Scene
    threeScene = new THREE.Scene();
    threeScene.background = new THREE.Color(0x1a1a1a);
    
    // Camera
    threeCamera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    threeCamera.position.set(2, 2, 2);
    threeCamera.lookAt(0, 0, 0);
    
    // Renderer
    threeRenderer = new THREE.WebGLRenderer({ antialias: true });
    threeRenderer.setSize(width, height);
    container.appendChild(threeRenderer.domElement);
    
    // Orbit controls for interaction
    if (typeof THREE.OrbitControls !== 'undefined') {
        orbitControls = new THREE.OrbitControls(threeCamera, threeRenderer.domElement);
        orbitControls.enableDamping = true;
        orbitControls.dampingFactor = 0.05;
    }
    
    // Lights
    const ambientLight = new THREE.AmbientLight(0x666666);
    threeScene.add(ambientLight);
    
    const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight1.position.set(1, 1, 1);
    threeScene.add(directionalLight1);
    
    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
    directionalLight2.position.set(-1, -1, -1);
    threeScene.add(directionalLight2);
    
    // Grid helper
    const gridHelper = new THREE.GridHelper(2, 20, 0x444444, 0x222222);
    threeScene.add(gridHelper);
    
    // Axes helper
    const axesHelper = new THREE.AxesHelper(1);
    threeScene.add(axesHelper);
    
    // Initialize GLTF loader
    gltfLoader = new THREE.GLTFLoader();
    
    // Render loop
    function animate() {
        requestAnimationFrame(animate);
        if (orbitControls) {
            orbitControls.update();
        }
        threeRenderer.render(threeScene, threeCamera);
    }
    animate();
    
    console.log('Three.js initialized');
}

// Update camera path visualization
function updateCameraPath(position) {
    if (!threeScene) return;
    
    cameraPath.push(new THREE.Vector3(position[0], position[1], position[2]));
    
    // Limit path length
    if (cameraPath.length > 100) {
        cameraPath.shift();
    }
    
    // Remove old path line
    const oldLine = threeScene.getObjectByName('cameraPath');
    if (oldLine) {
        threeScene.remove(oldLine);
    }
    
    // Create new path line
    if (cameraPath.length > 1) {
        const geometry = new THREE.BufferGeometry().setFromPoints(cameraPath);
        const material = new THREE.LineBasicMaterial({ 
            color: 0x667eea,
            linewidth: 2
        });
        const line = new THREE.Line(geometry, material);
        line.name = 'cameraPath';
        threeScene.add(line);
    }
    
    // Add current camera position marker
    const oldMarker = threeScene.getObjectByName('currentCamera');
    if (oldMarker) {
        threeScene.remove(oldMarker);
    }
    
    const markerGeometry = new THREE.SphereGeometry(0.02, 16, 16);
    const markerMaterial = new THREE.MeshBasicMaterial({ color: 0xff6b6b });
    const marker = new THREE.Mesh(markerGeometry, markerMaterial);
    marker.position.set(position[0], position[1], position[2]);
    marker.name = 'currentCamera';
    threeScene.add(marker);
}

// Load Model
async function loadModel() {
    const url = elements.gltfUrl.value.trim();
    
    if (!url) {
        alert('Please enter a GLTF model URL');
        return;
    }
    
    elements.loadModelBtn.disabled = true;
    elements.loadModelBtn.textContent = 'Loading...';
    
    try {
        // Load model in frontend for 3D visualization
        await loadModelInThreeJS(url);
        
        // Notify backend about the model (required for simulation to start)
        try {
            // Convert relative URL to absolute URL for backend
            let backendUrl = url;
            if (!url.startsWith('http://') && !url.startsWith('https://')) {
                // For local files, construct absolute URL
                const baseUrl = window.location.origin; // e.g., http://localhost:3000
                backendUrl = `${baseUrl}/${url.replace(/^\//, '')}`; // Remove leading slash if present
            }
            
            const response = await fetch(`${API_BASE_URL}/load_model?gltf_url=${encodeURIComponent(backendUrl)}`, {
                method: 'POST',
            });
            
            if (response.ok) {
                console.log('Backend notified about model:', backendUrl);
            } else {
                const errorData = await response.json().catch(() => ({}));
                console.warn('Backend model loading failed:', errorData);
                // Continue anyway - frontend visualization still works
            }
        } catch (backendError) {
            console.warn('Backend notification failed (frontend visualization still works):', backendError);
            // Continue anyway - frontend can still display the model
        }
        
        elements.modelStatus.textContent = 'Loaded';
        elements.modelStatus.style.color = '#51cf66';
        elements.startBtn.disabled = false;
        
        console.log('Model loaded in both backend and frontend');
        
    } catch (error) {
        console.error('Error loading model:', error);
        alert(`Failed to load model: ${error.message}`);
        elements.modelStatus.textContent = 'Error';
        elements.modelStatus.style.color = '#ff6b6b';
    } finally {
        elements.loadModelBtn.disabled = false;
        elements.loadModelBtn.textContent = 'Load Model';
    }
}

// Load model into Three.js scene
async function loadModelInThreeJS(url) {
    return new Promise((resolve, reject) => {
        if (!gltfLoader) {
            reject(new Error('GLTF Loader not initialized'));
            return;
        }
        
        // Remove existing model
        if (loadedModel) {
            threeScene.remove(loadedModel);
            loadedModel = null;
        }
        
        console.log('Loading GLTF model into Three.js:', url);
        
        gltfLoader.load(
            url,
            (gltf) => {
                loadedModel = gltf.scene;
                
                // Center and scale the model
                const box = new THREE.Box3().setFromObject(loadedModel);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                
                // Center the model
                loadedModel.position.sub(center);
                
                // Scale to fit in view (optional)
                const maxDim = Math.max(size.x, size.y, size.z);
                if (maxDim > 2) {
                    const scale = 2 / maxDim;
                    loadedModel.scale.multiplyScalar(scale);
                }
                
                // Add to scene
                threeScene.add(loadedModel);
                
                console.log('Model loaded successfully into Three.js');
                resolve(gltf);
            },
            (progress) => {
                const percent = (progress.loaded / progress.total * 100).toFixed(0);
                elements.loadModelBtn.textContent = `Loading... ${percent}%`;
                console.log(`Loading progress: ${percent}%`);
            },
            (error) => {
                console.error('Error loading GLTF:', error);
                reject(error);
            }
        );
    });
}

// Start Simulation
async function startSimulation() {
    try {
        elements.startBtn.disabled = true;
        
        // Start simulation on backend
        const response = await fetch(`${API_BASE_URL}/sim/start`, {
            method: 'POST',
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Simulation started:', data);
        
        elements.simStatus.textContent = 'Running';
        elements.simStatus.style.color = '#51cf66';
        elements.stopBtn.disabled = false;
        
        // Connect WebSocket
        connectWebSocket();
        
    } catch (error) {
        console.error('Error starting simulation:', error);
        alert(`Failed to start simulation: ${error.message}`);
        elements.startBtn.disabled = false;
    }
}

// Stop Simulation
async function stopSimulation() {
    try {
        elements.stopBtn.disabled = true;
        
        // Close WebSocket
        if (ws) {
            ws.close();
            ws = null;
        }
        
        isStreaming = false;
        
        // Stop simulation on backend
        const response = await fetch(`${API_BASE_URL}/sim/stop`, {
            method: 'POST',
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        elements.simStatus.textContent = 'Stopped';
        elements.simStatus.style.color = '#868e96';
        elements.startBtn.disabled = false;
        elements.loadingOverlay.classList.remove('hidden');
        
    } catch (error) {
        console.error('Error stopping simulation:', error);
        alert(`Failed to stop simulation: ${error.message}`);
    }
}

// Reset
function reset() {
    frameCount = 0;
    sessionMetrics = [];
    cameraPath = [];
    
    // Clear canvas
    const ctx = elements.videoCanvas.getContext('2d');
    ctx.clearRect(0, 0, elements.videoCanvas.width, elements.videoCanvas.height);
    
    // Reset metrics
    elements.framesCount.textContent = '0';
    elements.stepValue.textContent = '0';
    elements.cnnProb.textContent = '0.00';
    elements.cnnGaugeFill.style.width = '0%';
    elements.totalReward.textContent = '0.00';
    elements.coverageValue.textContent = '0.0%';
    
    // Clear Three.js scene
    if (threeScene) {
        const pathLine = threeScene.getObjectByName('cameraPath');
        const marker = threeScene.getObjectByName('currentCamera');
        if (pathLine) threeScene.remove(pathLine);
        if (marker) threeScene.remove(marker);
    }
}

// Connect WebSocket
function connectWebSocket() {
    console.log('Connecting to WebSocket...');
    
    ws = new WebSocket(WS_URL);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        elements.connectionStatus.textContent = 'Connected';
        elements.connectionStatus.style.color = '#51cf66';
        isStreaming = true;
        elements.loadingOverlay.classList.add('hidden');
    };
    
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleStreamData(data);
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        elements.connectionStatus.textContent = 'Error';
        elements.connectionStatus.style.color = '#ff6b6b';
    };
    
    ws.onclose = () => {
        console.log('WebSocket closed');
        elements.connectionStatus.textContent = 'Disconnected';
        elements.connectionStatus.style.color = '#868e96';
        isStreaming = false;
        elements.stopBtn.disabled = true;
        elements.startBtn.disabled = false;
    };
}

// Handle Stream Data
function handleStreamData(data) {
    // Update frame
    if (data.frame_base64) {
        const img = new Image();
        img.onload = () => {
            const canvas = elements.videoCanvas;
            const ctx = canvas.getContext('2d');
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
        };
        img.src = 'data:image/jpeg;base64,' + data.frame_base64;
    }
    
    // Update frame count and FPS
    frameCount++;
    elements.framesCount.textContent = frameCount;
    
    const now = Date.now();
    const deltaTime = (now - lastFrameTime) / 1000;
    if (deltaTime > 0) {
        fps = Math.round(1 / deltaTime);
        elements.fpsValue.textContent = fps;
    }
    lastFrameTime = now;
    
    // Update step
    elements.stepValue.textContent = data.step || 0;
    
    // Update CNN probability
    const cnnProb = data.cnn_prob || 0;
    elements.cnnProb.textContent = cnnProb.toFixed(2);
    elements.cnnGaugeFill.style.width = `${cnnProb * 100}%`;
    
    // Update CNN status
    if (cnnProb < 0.4) {
        elements.cnnStatus.textContent = 'Normal';
        elements.cnnStatus.className = 'status-indicator normal';
    } else if (cnnProb < 0.6) {
        elements.cnnStatus.textContent = 'Suspicious';
        elements.cnnStatus.className = 'status-indicator warning';
    } else {
        elements.cnnStatus.textContent = 'Lesion Detected';
        elements.cnnStatus.className = 'status-indicator danger';
    }
    
    // Update action
    elements.actionName.textContent = data.action_name || '-';
    elements.actionId.textContent = data.action_suggested !== undefined ? data.action_suggested : '-';
    
    // Update reward
    elements.stepReward.textContent = (data.reward || 0).toFixed(2);
    elements.totalReward.textContent = (data.total_reward || 0).toFixed(2);
    
    // Update coverage
    elements.coverageValue.textContent = `${((data.coverage || 0) * 100).toFixed(1)}%`;
    
    // Update pose
    if (data.pose) {
        const pos = data.pose.position || [0, 0, 0];
        const rot = data.pose.rotation || [0, 0, 0];
        
        elements.positionValue.textContent = 
            `x:${pos[0].toFixed(2)} y:${pos[1].toFixed(2)} z:${pos[2].toFixed(2)}`;
        elements.rotationValue.textContent = 
            `r:${rot[0].toFixed(2)} p:${rot[1].toFixed(2)} y:${rot[2].toFixed(2)}`;
        
        // Update Three.js visualization
        updateCameraPath(pos);
    }
    
    // Update collision
    elements.collisionValue.textContent = data.collision ? 'Yes' : 'No';
    elements.collisionValue.style.color = data.collision ? '#ff6b6b' : '#51cf66';
    
    // Store metrics
    sessionMetrics.push({
        timestamp: Date.now(),
        step: data.step,
        cnn_prob: cnnProb,
        action: data.action_suggested,
        reward: data.reward,
        total_reward: data.total_reward,
        coverage: data.coverage,
        collision: data.collision,
        pose: data.pose,
    });
}

// Download Metrics
function downloadMetrics() {
    if (sessionMetrics.length === 0) {
        alert('No metrics to download');
        return;
    }
    
    const data = JSON.stringify(sessionMetrics, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `session_metrics_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Event Listeners
elements.loadModelBtn.addEventListener('click', loadModel);
elements.startBtn.addEventListener('click', startSimulation);
elements.stopBtn.addEventListener('click', stopSimulation);
elements.resetBtn.addEventListener('click', reset);
elements.downloadBtn.addEventListener('click', downloadMetrics);

// Initialize on load
window.addEventListener('DOMContentLoaded', () => {
    console.log('Application initialized');
    initThreeJS();
    
    // Check API health
    fetch(`${API_BASE_URL}/health`)
        .then(response => response.json())
        .then(data => {
            console.log('API Health:', data);
        })
        .catch(error => {
            console.error('API not reachable:', error);
            alert('Warning: Cannot connect to API server. Make sure the backend is running.');
        });
});

// Handle window resize for Three.js
window.addEventListener('resize', () => {
    if (threeRenderer && threeCamera) {
        const container = document.getElementById('threejs-container');
        const width = container.clientWidth;
        const height = container.clientHeight;
        
        threeCamera.aspect = width / height;
        threeCamera.updateProjectionMatrix();
        threeRenderer.setSize(width, height);
    }
});

