"""RL training script using PPO."""

import argparse
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
import torch

from backend.sim.env import EndoscopyEnv
from backend.models.rl.callbacks import (
    TensorboardCallback,
    CurriculumCallback,
    BestModelCallback,
)
from backend.utils.config import load_yaml_config, get_device


def create_env(config_path: str, curriculum_stage: str = "easy"):
    """Create and wrap environment.

    Args:
        config_path: Path to sim config
        curriculum_stage: Curriculum stage

    Returns:
        Wrapped environment
    """
    def _init():
        env = EndoscopyEnv(config_path=config_path, curriculum_stage=curriculum_stage)
        env = Monitor(env)
        return env

    return _init


def train_rl(config_path: str) -> None:
    """Main RL training function.

    Args:
        config_path: Path to training configuration YAML
    """
    # Load configuration
    config = load_yaml_config(config_path)

    # Extract config sections
    algo_config = config.get("algorithm", {})
    training_config = config.get("training", {})
    env_config = config.get("environment", {})
    curriculum_config = config.get("curriculum", {})
    evaluation_config = config.get("evaluation", {})
    callbacks_config = config.get("callbacks", {})
    export_config = config.get("export", {})

    # Setup device
    device = get_device()
    print(f"Using device: {device}")

    # Create directories
    checkpoint_dir = Path(callbacks_config.get("checkpoint", {}).get("save_path", "./checkpoints/rl"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(callbacks_config.get("tensorboard", {}).get("log_dir", "./logs/rl"))
    log_dir.mkdir(parents=True, exist_ok=True)

    # Environment config
    sim_config_path = env_config.get("config_path", "./configs/sim.yaml")
    n_envs = training_config.get("n_envs", 8)

    # Create vectorized environments
    print(f"Creating {n_envs} parallel environments...")
    env = make_vec_env(
        create_env(sim_config_path, curriculum_stage="easy"),
        n_envs=n_envs,
        seed=config.get("seed", 42),
    )

    # Wrap with VecNormalize for observation/reward normalization
    if training_config.get("normalize_rewards", True):
        env = VecNormalize(
            env,
            training=True,
            norm_obs=True,
            norm_reward=True,
            gamma=algo_config.get("gamma", 0.99),
        )

    # Frame stacking
    frame_stack = training_config.get("frame_stack", 4)
    if frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack, channels_order="last")

    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = make_vec_env(
        create_env(sim_config_path, curriculum_stage="easy"),
        n_envs=1,
        seed=config.get("seed", 42) + 1000,
    )
    eval_env = VecNormalize(
        eval_env,
        training=False,
        norm_obs=True,
        norm_reward=False,
    )

    # Policy kwargs
    policy_kwargs = algo_config.get("policy_kwargs", {})
    if "activation_fn" in policy_kwargs:
        # Convert string to torch activation
        activation_name = policy_kwargs["activation_fn"]
        if activation_name == "relu":
            policy_kwargs["activation_fn"] = torch.nn.ReLU
        elif activation_name == "tanh":
            policy_kwargs["activation_fn"] = torch.nn.Tanh

    # Create PPO model
    print("Creating PPO model...")
    model = PPO(
        policy=algo_config.get("policy", "CnnPolicy"),
        env=env,
        learning_rate=algo_config.get("learning_rate", 3e-4),
        n_steps=algo_config.get("n_steps", 2048),
        batch_size=algo_config.get("batch_size", 64),
        n_epochs=algo_config.get("n_epochs", 10),
        gamma=algo_config.get("gamma", 0.99),
        gae_lambda=algo_config.get("gae_lambda", 0.95),
        clip_range=algo_config.get("clip_range", 0.2),
        ent_coef=algo_config.get("ent_coef", 0.01),
        vf_coef=algo_config.get("vf_coef", 0.5),
        max_grad_norm=algo_config.get("max_grad_norm", 0.5),
        use_sde=algo_config.get("use_sde", False),
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir),
        device=device,
        verbose=config.get("verbose", 1),
        seed=config.get("seed", 42),
    )

    # Setup callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_config = callbacks_config.get("checkpoint", {})
    if checkpoint_config:
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_config.get("save_freq", 50000),
            save_path=str(checkpoint_dir),
            name_prefix=checkpoint_config.get("name_prefix", "ppo"),
            save_replay_buffer=checkpoint_config.get("save_replay_buffer", False),
            save_vecnormalize=checkpoint_config.get("save_vecnormalize", True),
        )
        callbacks.append(checkpoint_callback)

    # Evaluation callback
    eval_freq = evaluation_config.get("eval_freq", 10000)
    n_eval_episodes = evaluation_config.get("n_eval_episodes", 10)
    if eval_freq > 0:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(checkpoint_dir.parent),
            log_path=str(log_dir / "eval"),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=evaluation_config.get("deterministic", True),
            render=False,
        )
        callbacks.append(eval_callback)

    # Curriculum callback
    if curriculum_config.get("enabled", True):
        curriculum_callback = CurriculumCallback(
            difficulty_schedule=curriculum_config.get("difficulty_schedule", []),
            sim_config_path=sim_config_path,
        )
        callbacks.append(curriculum_callback)

    # TensorBoard callback
    tb_callback = TensorboardCallback()
    callbacks.append(tb_callback)

    callback_list = CallbackList(callbacks)

    # Train
    total_timesteps = training_config.get("total_timesteps", 2000000)
    print(f"\nStarting training for {total_timesteps} timesteps...")
    print(f"This is approximately {total_timesteps // (n_envs * algo_config.get('n_steps', 2048))} updates")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        log_interval=config.get("log_interval", 10),
        progress_bar=True,
    )

    # Save final model
    final_model_path = checkpoint_dir.parent / "ppo_final"
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Save VecNormalize stats
    if isinstance(env, VecNormalize):
        env.save(str(checkpoint_dir.parent / "vec_normalize.pkl"))
        print("VecNormalize stats saved")

    # Export to ONNX
    if export_config.get("formats") and "onnx" in export_config.get("formats", []):
        print("\nExporting to ONNX...")
        try:
            from backend.models.rl.export_onnx import export_ppo_to_onnx
            
            export_dir = Path(export_config.get("output_dir", "./checkpoints"))
            onnx_path = export_dir / "ppo_policy.onnx"
            
            export_ppo_to_onnx(
                model=model,
                output_path=str(onnx_path),
                observation_shape=env.observation_space.shape,
            )
        except Exception as e:
            print(f"ONNX export failed: {e}")

    print("\nTraining complete!")

    # Cleanup
    env.close()
    eval_env.close()


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description="Train RL policy with PPO")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_rl.yaml",
        help="Path to configuration YAML",
    )
    args = parser.parse_args()

    train_rl(args.config)


if __name__ == "__main__":
    main()

