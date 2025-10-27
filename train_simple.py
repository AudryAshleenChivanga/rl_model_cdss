# -*- coding: utf-8 -*-
"""Simple RL training starter script."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import torch

from backend.sim.env import EndoscopyEnv

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print("\n" + "="*60)
print("  RL TRAINING - PURE REINFORCEMENT LEARNING")
print("="*60 + "\n")
print(f"Device: {device}")
print("Scenario: Peptic Ulcer Disease")
print("Algorithm: PPO (learns from rewards only)")
print("\nWhat the agent will learn:")
print("  * Navigate inside digestive system")
print("  * Recognize lesion patterns (visual)")
print("  * Maximize coverage")
print("  * Avoid collisions")
print("\n" + "="*60 + "\n")

# Create environment
print("Creating environment...")
env = EndoscopyEnv(
    config_path="configs/sim.yaml",
    gltf_path="frontend/models/digestive/source/scene.gltf",
    scenario_id="peptic_ulcer",  # Start with peptic ulcers
)
env = Monitor(env)
print("[OK] Environment created\n")

# Create directories
Path("checkpoints").mkdir(exist_ok=True)
Path("logs/rl").mkdir(parents=True, exist_ok=True)

# Create PPO agent
print("Initializing PPO agent...")
print("  * Policy: CnnPolicy (learns from pixels)")
print("  * No pre-training needed")
print("  * Will discover patterns through trial & error\n")

model = PPO(
    "CnnPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    tensorboard_log="./logs/rl",
    device=device,
)

print("[OK] PPO agent initialized\n")

# Setup callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./checkpoints",
    name_prefix="ppo_peptic_ulcer",
    save_replay_buffer=False,
)

# Train!
print("="*60)
print("  STARTING TRAINING")
print("="*60 + "\n")
print("Total timesteps: 50,000 (quick trial)")
print("Checkpoints every: 10,000 steps")
print("\nProgress will appear below...")
print("(This will take 10-30 minutes depending on your CPU)\n")

try:
    model.learn(
        total_timesteps=50000,
        callback=checkpoint_callback,
        log_interval=10,
        progress_bar=True,
    )
    
    print("\n" + "="*60)
    print("  TRAINING COMPLETE!")
    print("="*60 + "\n")
    
    # Save final model
    model.save("checkpoints/ppo_peptic_ulcer_final")
    print("[OK] Model saved to: checkpoints/ppo_peptic_ulcer_final.zip")
    
    print("\nNext steps:")
    print("  1. Test the trained agent in the UI")
    print("  2. Train for more timesteps (500k-1M for expert)")
    print("  3. Try other scenarios (gastric cancer, H. pylori, etc.)")
    
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user")
    model.save("checkpoints/ppo_peptic_ulcer_interrupted")
    print("[OK] Progress saved to: checkpoints/ppo_peptic_ulcer_interrupted.zip")

except Exception as e:
    print(f"\n\nError during training: {e}")
    import traceback
    traceback.print_exc()

finally:
    env.close()
    print("\n[OK] Environment closed")
