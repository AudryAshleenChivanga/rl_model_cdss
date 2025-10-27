"""Simulation control and streaming endpoints."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from pydantic import BaseModel
import asyncio
import base64
import io
import numpy as np
from PIL import Image
import torch
import json
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.sim.env import EndoscopyEnv
from backend.sim.scenarios import get_scenario_list, SCENARIOS
from backend.api.routers.models import get_current_gltf_url, get_cnn_model, get_ppo_model
from backend.utils.config import settings

router = APIRouter()

# Global simulation state
simulation_running = False
simulation_env = None
simulation_task = None
current_scenario = "healthy"  # Default scenario


class SimulationStatus(BaseModel):
    """Simulation status response."""
    running: bool
    step: int | None = None
    episode: int | None = None
    total_reward: float | None = None


@router.get("/sim/scenarios")
async def get_scenarios():
    """Get list of available clinical scenarios.
    
    Returns:
        List of available scenarios with descriptions
    """
    return {
        "scenarios": get_scenario_list(),
        "current_scenario": current_scenario
    }


@router.post("/sim/set_scenario")
async def set_scenario(scenario_id: str = Query(..., description="Scenario ID")):
    """Set the clinical scenario for next simulation.
    
    Args:
        scenario_id: Scenario identifier
        
    Returns:
        Success message with scenario info
    """
    global current_scenario
    
    if scenario_id not in SCENARIOS:
        available = list(SCENARIOS.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scenario. Available scenarios: {available}"
        )
    
    current_scenario = scenario_id
    scenario = SCENARIOS[scenario_id]
    
    return {
        "status": "success",
        "message": f"Scenario set to: {scenario.name}",
        "scenario": {
            "id": scenario.id,
            "name": scenario.name,
            "description": scenario.description,
            "difficulty": scenario.difficulty,
        }
    }


@router.post("/sim/start")
async def start_simulation():
    """Start the simulation.
    
    Initializes the environment and begins the simulation loop.
    Frames will be available via WebSocket at /api/stream
    """
    global simulation_running, simulation_env, current_scenario
    
    if simulation_running:
        raise HTTPException(status_code=400, detail="Simulation already running")
    
    # Get GLTF URL
    gltf_url = get_current_gltf_url()
    if not gltf_url:
        raise HTTPException(
            status_code=400,
            detail="No GLTF model loaded. Call /api/load_model first."
        )
    
    try:
        # Initialize environment with current scenario
        config_path = "configs/sim.yaml"
        simulation_env = EndoscopyEnv(
            config_path=config_path,
            gltf_path=gltf_url,
            scenario_id=current_scenario,  # Pass scenario
        )
        
        # Reset environment
        obs, info = simulation_env.reset()
        
        simulation_running = True
        
        scenario = SCENARIOS[current_scenario]
        
        return {
            "status": "started",
            "message": "Simulation started successfully",
            "gltf_url": gltf_url,
            "scenario": {
                "id": scenario.id,
                "name": scenario.name,
                "description": scenario.description,
            },
            "info": {
                "max_steps": simulation_env.max_steps,
                "observation_shape": obs.shape,
            }
        }
        
    except Exception as e:
        simulation_running = False
        raise HTTPException(status_code=500, detail=f"Failed to start simulation: {str(e)}")


@router.post("/sim/stop")
async def stop_simulation():
    """Stop the simulation.
    
    Stops the simulation loop and cleans up resources.
    """
    global simulation_running, simulation_env
    
    if not simulation_running:
        raise HTTPException(status_code=400, detail="Simulation not running")
    
    simulation_running = False
    
    if simulation_env:
        simulation_env.close()
        simulation_env = None
    
    return {
        "status": "stopped",
        "message": "Simulation stopped successfully"
    }


@router.get("/sim/status", response_model=SimulationStatus)
async def get_simulation_status():
    """Get current simulation status.
    
    Returns whether simulation is running and current metrics.
    """
    if not simulation_running or simulation_env is None:
        return SimulationStatus(running=False)
    
    return SimulationStatus(
        running=True,
        step=simulation_env.current_step,
        episode=0,  # Would track episodes if needed
        total_reward=simulation_env.total_reward,
    )


@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for streaming simulation frames.
    
    Streams frames with CNN predictions and RL action suggestions.
    
    Message format:
    ```json
    {
        "frame_base64": "...",
        "cnn_prob": 0.85,
        "action_suggested": 3,
        "action_name": "pitch_pos",
        "reward": 0.42,
        "pose": {
            "position": [x, y, z],
            "rotation": [roll, pitch, yaw]
        },
        "step": 42,
        "coverage": 0.23,
        "collision": false
    }
    ```
    """
    await websocket.accept()
    
    global simulation_running, simulation_env
    
    # Action names for display
    action_names = [
        "yaw_pos", "yaw_neg", "pitch_pos", "pitch_neg",
        "forward", "backward", "zoom_in", "zoom_out", "done"
    ]
    
    try:
        # Load models if available
        cnn_model = get_cnn_model()
        ppo_model = get_ppo_model()
        
        if cnn_model:
            cnn_model.eval()
        
        # Wait for simulation to start
        while not simulation_running:
            await asyncio.sleep(0.1)
        
        # Get initial observation
        obs = simulation_env.render()
        
        # Simulation loop
        while simulation_running and simulation_env:
            try:
                # Get current observation
                obs = simulation_env.render()
                
                # Convert frame to base64
                pil_image = Image.fromarray(obs)
                buffer = io.BytesIO()
                pil_image.save(buffer, format="JPEG", quality=85)
                frame_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                # CNN prediction
                cnn_prob = 0.0
                if cnn_model:
                    try:
                        # Preprocess for CNN
                        from torchvision import transforms
                        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]
                            )
                        ])
                        img_tensor = transform(pil_image).unsqueeze(0).to(settings.device)
                        
                        with torch.no_grad():
                            probs = cnn_model.predict_proba(img_tensor)
                            cnn_prob = float(probs[0, 1].cpu().numpy())  # Probability of lesion
                    except Exception as e:
                        print(f"CNN inference error: {e}")
                
                # RL action suggestion
                action_suggested = 0
                action_name = action_names[0]
                if ppo_model:
                    try:
                        # Prepare observation for PPO
                        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                        obs_tensor = obs_tensor.permute(0, 3, 1, 2)  # BHWC -> BCHW
                        obs_tensor = obs_tensor / 255.0  # Normalize
                        
                        action, _states = ppo_model.predict(obs_tensor.numpy(), deterministic=True)
                        action_suggested = int(action)
                        action_name = action_names[action_suggested]
                    except Exception as e:
                        print(f"PPO inference error: {e}")
                        # Fall back to random action
                        action_suggested = np.random.randint(0, 5)
                        action_name = action_names[action_suggested]
                else:
                    # Random action if PPO not loaded
                    action_suggested = np.random.randint(0, 5)
                    action_name = action_names[action_suggested]
                
                # Step environment
                obs, reward, terminated, truncated, info = simulation_env.step(action_suggested)
                
                # Build message (convert numpy types to native Python types)
                message = {
                    "frame_base64": frame_base64,
                    "cnn_prob": round(cnn_prob, 4),
                    "action_suggested": int(action_suggested),
                    "action_name": action_name,
                    "reward": round(float(reward), 4),
                    "pose": {
                        "position": [float(x) for x in info.get("camera_position", [0, 0, 0])],
                        "rotation": [float(x) for x in info.get("camera_rotation", [0, 0, 0])],
                    },
                    "step": int(info.get("step", 0)),
                    "coverage": round(float(info.get("coverage", 0.0)), 4),
                    "collision": bool(info.get("collision", False)),
                    "total_reward": round(float(info.get("total_reward", 0.0)), 4),
                }
                
                # Send message
                await websocket.send_json(message)
                
                # Reset if episode finished
                if terminated or truncated:
                    obs, info = simulation_env.reset()
                
                # Control frame rate
                await asyncio.sleep(1.0 / settings.render_fps)
                
            except WebSocketDisconnect:
                print("WebSocket client disconnected")
                break
            except Exception as e:
                print(f"Error in simulation loop: {e}")
                await asyncio.sleep(0.1)
        
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.send_json({"error": str(e)})
    finally:
        try:
            await websocket.close()
        except:
            pass


def cleanup_simulation():
    """Cleanup simulation resources."""
    global simulation_running, simulation_env
    
    simulation_running = False
    if simulation_env:
        try:
            simulation_env.close()
        except:
            pass
        simulation_env = None

