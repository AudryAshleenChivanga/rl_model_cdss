"""Custom callbacks for RL training."""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, List


class TensorboardCallback(BaseCallback):
    """Custom callback for tensorboard logging."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        """Called at each step."""
        # Log episode info
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if "r" in info:
                    self.episode_rewards.append(info["r"])
                if "l" in info:
                    self.episode_lengths.append(info["l"])

        return True

    def _on_rollout_end(self) -> None:
        """Called at end of rollout."""
        if len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards)
            mean_length = np.mean(self.episode_lengths)

            self.logger.record("rollout/ep_rew_mean", mean_reward)
            self.logger.record("rollout/ep_len_mean", mean_length)

            self.episode_rewards = []
            self.episode_lengths = []


class CurriculumCallback(BaseCallback):
    """Callback for curriculum learning."""

    def __init__(
        self,
        difficulty_schedule: List[Dict],
        sim_config_path: str,
        verbose: int = 0,
    ):
        """Initialize curriculum callback.

        Args:
            difficulty_schedule: List of dicts with 'timesteps' and 'stage'
            sim_config_path: Path to sim config
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.difficulty_schedule = sorted(
            difficulty_schedule, key=lambda x: x["timesteps"]
        )
        self.sim_config_path = sim_config_path
        self.current_stage_idx = 0

    def _on_step(self) -> bool:
        """Check if we should increase difficulty."""
        if self.current_stage_idx >= len(self.difficulty_schedule):
            return True

        next_stage = self.difficulty_schedule[self.current_stage_idx]
        if self.num_timesteps >= next_stage["timesteps"]:
            stage_name = next_stage["stage"]

            if self.verbose > 0:
                print(f"\n{'='*50}")
                print(f"Curriculum Update: Switching to '{stage_name}' stage")
                print(f"Timesteps: {self.num_timesteps}")
                print(f"{'='*50}\n")

            # Note: Actually updating the environment difficulty would require
            # recreating environments, which is complex. For simplicity,
            # we just log the transition here.
            self.logger.record("curriculum/stage", self.current_stage_idx)

            self.current_stage_idx += 1

        return True


class BestModelCallback(BaseCallback):
    """Callback to save best model based on custom metric."""

    def __init__(
        self,
        save_path: str,
        metric: str = "mean_reward",
        verbose: int = 1,
    ):
        """Initialize callback.

        Args:
            save_path: Path to save best model
            metric: Metric to monitor
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_path = save_path
        self.metric = metric
        self.best_metric = -np.inf

    def _on_step(self) -> bool:
        """Check if current model is best."""
        # This would need to be called after evaluation
        # For simplicity, using mean episode reward from buffer
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])

            if mean_reward > self.best_metric:
                self.best_metric = mean_reward
                self.model.save(self.save_path)

                if self.verbose > 0:
                    print(f"New best model saved with {self.metric}={mean_reward:.2f}")

        return True


class MetricsCallback(BaseCallback):
    """Callback to track custom environment metrics."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_coverages = []
        self.episode_collisions = []

    def _on_step(self) -> bool:
        """Extract custom metrics from info dict."""
        # Check if episode is done
        if "infos" in self.locals:
            for info in self.locals["infos"]:
                if "episode" in info:
                    # Episode finished, extract metrics
                    if "coverage" in info:
                        self.episode_coverages.append(info["coverage"])
                    if "collision" in info:
                        self.episode_collisions.append(int(info["collision"]))

        return True

    def _on_rollout_end(self) -> None:
        """Log metrics at end of rollout."""
        if len(self.episode_coverages) > 0:
            mean_coverage = np.mean(self.episode_coverages)
            self.logger.record("metrics/mean_coverage", mean_coverage)
            self.episode_coverages = []

        if len(self.episode_collisions) > 0:
            collision_rate = np.mean(self.episode_collisions)
            self.logger.record("metrics/collision_rate", collision_rate)
            self.episode_collisions = []

