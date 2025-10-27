# -*- coding: utf-8 -*-
"""Check training progress."""
import os
from pathlib import Path
import time
from datetime import datetime

print("\n" + "="*60)
print("  RL TRAINING PROGRESS MONITOR")
print("="*60 + "\n")

# Check if training process exists
print("Checking status...\n")

# Check for checkpoints
checkpoint_dir = Path("checkpoints")
if checkpoint_dir.exists():
    checkpoints = list(checkpoint_dir.glob("*.zip"))
    if checkpoints:
        print(f"[OK] CHECKPOINTS FOUND: {len(checkpoints)}")
        for cp in sorted(checkpoints, key=lambda x: x.stat().st_mtime):
            size_mb = cp.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(cp.stat().st_mtime)
            print(f"   * {cp.name} ({size_mb:.1f}MB) - {mtime.strftime('%H:%M:%S')}")
    else:
        print("[WAIT] No checkpoints yet (normal in first 5-10 minutes)")
else:
    print("[WAIT] Checkpoint directory not created yet")

print()

# Check for logs
log_dir = Path("logs/rl")
if log_dir.exists():
    log_files = list(log_dir.rglob("*"))
    if log_files:
        print(f"[OK] LOG FILES FOUND: {len(log_files)}")
        for log in log_files[:5]:
            if log.is_file():
                print(f"   * {log.relative_to('logs/rl')}")
    else:
        print("[WAIT] No log files yet")
else:
    print("[WAIT] Log directory not created yet")

print("\n" + "="*60)
print("  WHAT'S HAPPENING NOW")
print("="*60 + "\n")

print("""
The RL agent is currently:

1. EXPLORING
   - Moving randomly through the digestive system
   - Taking actions: rotate, move forward/back
   - Observing what it sees (224x224 pixel images)

2. COLLECTING EXPERIENCE
   - Recording: (state, action, reward, next_state)
   - Building up memory of what works

3. LEARNING (Every 2048 steps)
   - Analyzing which actions led to high rewards
   - Updating neural network weights
   - Improving policy: "Do more of what worked!"

4. SAVING PROGRESS (Every 10,000 steps)
   - Checkpoint files will appear in checkpoints/
   - Can resume training anytime

EXPECTED TIMELINE:
  0-5 min:  Random exploration, low rewards
  5-15 min: Starting to learn patterns
  15-25 min: Getting better at navigation
  25-30 min: First checkpoint saved!

""")

print("="*60)
print("  NEXT STEPS")
print("="*60 + "\n")

print("""
While training runs:
1. Let it run for at least 30 minutes
2. Monitor progress with: python check_training.py
3. Test trained model in UI when complete
4. Train longer (500k-1M steps) for expert performance

Press Ctrl+C in the training window to stop early
(Progress will be saved automatically!)
""")
