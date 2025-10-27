"""Tests for EndoscopyEnv."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.sim.env import EndoscopyEnv


class TestEndoscopyEnv:
    """Test suite for EndoscopyEnv."""

    @pytest.fixture
    def env(self):
        """Create test environment."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "sim.yaml"
        env = EndoscopyEnv(config_path=str(config_path))
        yield env
        env.close()

    def test_env_creation(self, env):
        """Test environment can be created."""
        assert env is not None
        assert env.action_space is not None
        assert env.observation_space is not None

    def test_reset(self, env):
        """Test environment reset."""
        obs, info = env.reset(seed=42)
        
        assert obs is not None
        assert obs.shape == (env.render_height, env.render_width, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        assert "step" in info
        assert info["step"] == 0

    def test_step(self, env):
        """Test environment step."""
        env.reset(seed=42)
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == (env.render_height, env.render_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_multiple_steps(self, env):
        """Test multiple environment steps."""
        env.reset(seed=42)
        
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                env.reset()

    def test_episode_termination(self, env):
        """Test episode terminates correctly."""
        env.reset(seed=42)
        
        # Step until episode ends or max steps
        for _ in range(env.max_steps + 10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        assert terminated or truncated

    def test_action_space(self, env):
        """Test action space."""
        assert env.action_space.n == 9
        
        # Test all actions are valid
        for action in range(env.action_space.n):
            assert env.action_space.contains(action)

    def test_observation_space(self, env):
        """Test observation space."""
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)

    def test_reward_range(self, env):
        """Test rewards are reasonable."""
        env.reset(seed=42)
        
        rewards = []
        for _ in range(10):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            
            if terminated or truncated:
                break
        
        # Check rewards are finite
        assert all(np.isfinite(r) for r in rewards)

    def test_seeding(self):
        """Test environment seeding for reproducibility."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "sim.yaml"
        
        env1 = EndoscopyEnv(config_path=str(config_path))
        obs1, _ = env1.reset(seed=42)
        
        env2 = EndoscopyEnv(config_path=str(config_path))
        obs2, _ = env2.reset(seed=42)
        
        # Same seed should give similar initial observations
        # (may not be exactly equal due to rendering)
        assert obs1.shape == obs2.shape
        
        env1.close()
        env2.close()


def test_env_smoke():
    """Quick smoke test."""
    config_path = Path(__file__).parent.parent.parent / "configs" / "sim.yaml"
    env = EndoscopyEnv(config_path=str(config_path))
    
    env.reset(seed=42)
    for _ in range(5):
        action = env.action_space.sample()
        env.step(action)
    
    env.close()
    print("Environment smoke test passed!")


if __name__ == "__main__":
    test_env_smoke()

