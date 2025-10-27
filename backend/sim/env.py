"""Custom Gymnasium environment for endoscopy simulation."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from backend.sim.renderer import EndoscopyRenderer
from backend.sim.lesion_synth import LesionSynthesizer
from backend.sim.scenarios import ScenarioGenerator
from backend.utils.config import load_yaml_config


class EndoscopyEnv(gym.Env):
    """Gymnasium environment for 3D endoscopy navigation and anomaly detection."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    # Action space indices
    ACTION_YAW_POS = 0
    ACTION_YAW_NEG = 1
    ACTION_PITCH_POS = 2
    ACTION_PITCH_NEG = 3
    ACTION_FORWARD = 4
    ACTION_BACKWARD = 5
    ACTION_ZOOM_IN = 6
    ACTION_ZOOM_OUT = 7
    ACTION_DONE = 8

    def __init__(
        self,
        config_path: Optional[str] = None,
        gltf_path: Optional[str] = None,
        render_mode: Optional[str] = "rgb_array",
        curriculum_stage: str = "easy",
        scenario_id: str = "healthy",
    ):
        """Initialize endoscopy environment.

        Args:
            config_path: Path to YAML configuration file
            gltf_path: Path or URL to GLTF model
            render_mode: Rendering mode
            curriculum_stage: Curriculum difficulty stage
            scenario_id: Clinical scenario identifier
        """
        super().__init__()
        
        # Store scenario
        self.scenario_id = scenario_id

        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "sim.yaml"
        self.config = load_yaml_config(config_path)

        # Environment settings
        env_config = self.config.get("environment", {})
        self.max_steps = env_config.get("max_steps", 500)
        self.render_width = env_config.get("render_width", 224)
        self.render_height = env_config.get("render_height", 224)
        self.fps = env_config.get("fps", 10)

        # Camera settings
        self.fov = env_config.get("fov", 70)
        self.near_clip = env_config.get("near_clip", 0.01)
        self.far_clip = env_config.get("far_clip", 10.0)
        self.initial_position = np.array(env_config.get("initial_position", [0.0, 0.0, 0.5]))
        self.initial_rotation = np.array(env_config.get("initial_rotation", [0.0, 0.0, 0.0]))

        # Action deltas
        actions_config = env_config.get("actions", {})
        self.yaw_delta = actions_config.get("yaw_delta", 0.1)
        self.pitch_delta = actions_config.get("pitch_delta", 0.1)
        self.forward_delta = actions_config.get("forward_delta", 0.05)
        self.zoom_delta = actions_config.get("zoom_delta", 0.02)

        # Lesion settings
        lesion_config = env_config.get("lesions_per_episode", {})
        self.lesions_min = lesion_config.get("min", 1)
        self.lesions_max = lesion_config.get("max", 4)

        # Domain randomization
        self.lighting_config = env_config.get("lighting", {})
        self.camera_noise_config = env_config.get("camera_noise", {})

        # Collision settings
        self.collision_threshold = env_config.get("collision_threshold", 0.02)
        self.collision_normal_dot = env_config.get("collision_normal_dot", 0.9)

        # Coverage tracking
        self.coverage_grid_size = env_config.get("coverage_grid_size", 32)
        self.coverage_ray_samples = env_config.get("coverage_ray_samples", 100)

        # Reward coefficients
        reward_config = self.config.get("reward", {})
        self.alpha = reward_config.get("alpha", 0.5)  # Coverage
        self.beta = reward_config.get("beta", 0.4)  # Anomaly detection
        self.gamma = reward_config.get("gamma", 0.2)  # Collision penalty
        self.delta = reward_config.get("delta", 0.05)  # Jerk penalty
        self.coverage_per_new_cell = reward_config.get("coverage_per_new_cell", 0.1)
        self.anomaly_detection_bonus = reward_config.get("anomaly_detection_bonus", 1.0)
        self.collision_penalty = reward_config.get("collision_penalty", -0.5)
        self.jerk_penalty_factor = reward_config.get("jerk_penalty_factor", -0.01)
        self.episode_completion_bonus = reward_config.get("episode_completion_bonus", 5.0)

        # Curriculum learning
        self.curriculum_stage = curriculum_stage
        self._apply_curriculum()

        # Gymnasium spaces
        self.action_space = spaces.Discrete(9)  # 8 movement actions + done
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.render_height, self.render_width, 3),
            dtype=np.uint8,
        )

        # Initialize renderer and lesion synthesizer
        self.renderer = EndoscopyRenderer(
            width=self.render_width,
            height=self.render_height,
            fov=self.fov,
            near_clip=self.near_clip,
            far_clip=self.far_clip,
        )
        self.lesion_synth = LesionSynthesizer()
        self.scenario_generator = ScenarioGenerator()

        # Load GLTF model if provided
        self.gltf_path = gltf_path
        if gltf_path:
            self.load_model(gltf_path)

        # Episode state
        self.current_step = 0
        self.camera_position = self.initial_position.copy()
        self.camera_rotation = self.initial_rotation.copy()
        self.lesions = []
        self.visited_cells = set()
        self.camera_trajectory = []
        self.previous_action = None
        self.total_reward = 0.0

        self.render_mode = render_mode

    def _apply_curriculum(self):
        """Apply curriculum learning difficulty adjustments."""
        curriculum_config = self.config.get("curriculum", {})
        if not curriculum_config.get("enabled", True):
            return

        stages = curriculum_config.get("stages", [])
        for stage in stages:
            if stage["name"] == self.curriculum_stage:
                difficulty = stage.get("difficulty", {})
                # Apply difficulty settings (could modify collision threshold, etc.)
                if "lesions_per_episode" in difficulty:
                    self.lesions_max = difficulty["lesions_per_episode"]
                break

    def load_model(self, gltf_path: str) -> None:
        """Load GLTF model into the environment.

        Args:
            gltf_path: Path or URL to GLTF file
        """
        self.gltf_path = gltf_path
        self.renderer.load_gltf(gltf_path)
        self.renderer.setup_scene()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed
            options: Additional reset options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Reset episode state
        self.current_step = 0
        self.camera_position = self.initial_position.copy()
        self.camera_rotation = self.initial_rotation.copy()
        self.visited_cells = set()
        self.camera_trajectory = [self.camera_position.copy()]
        self.previous_action = None
        self.total_reward = 0.0

        # Generate lesions based on scenario
        if self.renderer.vertices is not None:
            self.lesions = self.scenario_generator.generate_scenario(
                scenario_id=self.scenario_id,
                vertices=self.renderer.vertices,
                faces=self.renderer.faces,
            )
            print(f"Generated {len(self.lesions)} lesions for scenario '{self.scenario_id}'")
        else:
            self.lesions = []

        # Randomize lighting
        if self.lighting_config:
            intensity_range = self.lighting_config.get("intensity_range", [0.5, 1.5])
            self.renderer.ambient_light = self.np_random.uniform(*intensity_range)

        # Set camera pose and render
        self.renderer.set_camera_pose(self.camera_position, self.camera_rotation)
        observation = self._get_observation()

        # Build info dict
        info = self._get_info()

        return observation, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: Action index

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1

        # Apply action
        self._apply_action(action)

        # Update camera trajectory
        self.camera_trajectory.append(self.camera_position.copy())

        # Check collision
        collision = self.renderer.check_collision(
            self.camera_position,
            threshold=self.collision_threshold,
        )

        # Update coverage
        new_coverage = self._update_coverage()

        # Compute reward
        reward = self._compute_reward(action, collision, new_coverage)
        self.total_reward += reward

        # Check termination
        terminated = action == self.ACTION_DONE or collision
        truncated = self.current_step >= self.max_steps

        # Get observation
        observation = self._get_observation()

        # Build info dict
        info = self._get_info()
        info["collision"] = collision
        info["new_coverage"] = new_coverage
        info["step_reward"] = reward

        self.previous_action = action

        return observation, reward, terminated, truncated, info

    def _apply_action(self, action: int) -> None:
        """Apply action to update camera pose.

        Args:
            action: Action index
        """
        # Store old position for rollback if needed
        old_position = self.camera_position.copy()
        
        if action == self.ACTION_YAW_POS:
            self.camera_rotation[2] += self.yaw_delta
        elif action == self.ACTION_YAW_NEG:
            self.camera_rotation[2] -= self.yaw_delta
        elif action == self.ACTION_PITCH_POS:
            self.camera_rotation[1] += self.pitch_delta
        elif action == self.ACTION_PITCH_NEG:
            self.camera_rotation[1] -= self.pitch_delta
        elif action == self.ACTION_FORWARD:
            # Move forward in camera's forward direction
            forward = self._get_forward_vector()
            new_position = self.camera_position + forward * self.forward_delta
            
            # Check if new position is valid (inside mesh)
            if self._is_position_valid(new_position):
                self.camera_position = new_position
            # else: stay at current position (movement blocked by boundary)
            
        elif action == self.ACTION_BACKWARD:
            forward = self._get_forward_vector()
            new_position = self.camera_position - forward * self.forward_delta
            
            # Check if new position is valid (inside mesh)
            if self._is_position_valid(new_position):
                self.camera_position = new_position
            # else: stay at current position (movement blocked by boundary)
            
        elif action == self.ACTION_ZOOM_IN:
            # Zoom is handled by FOV adjustment (not implemented here)
            pass
        elif action == self.ACTION_ZOOM_OUT:
            pass
        elif action == self.ACTION_DONE:
            pass

        # Clamp rotation angles
        self.camera_rotation[1] = np.clip(self.camera_rotation[1], -np.pi/2, np.pi/2)

        # Update renderer
        self.renderer.set_camera_pose(self.camera_position, self.camera_rotation)
    
    def _is_position_valid(self, position: np.ndarray) -> bool:
        """Check if a position is inside the mesh boundaries.
        
        Args:
            position: Position to check (x, y, z)
            
        Returns:
            True if position is valid (inside mesh), False otherwise
        """
        if self.renderer.mesh is None or self.renderer.vertices is None:
            # If no mesh loaded, use simple sphere constraint
            return np.linalg.norm(position) < 0.8
        
        # Check if position is inside the mesh using ray casting
        # Cast rays in multiple directions and count intersections
        # Odd number = inside, even number = outside
        try:
            # Simple bounds check first (fast)
            bounds_min = self.renderer.vertices.min(axis=0)
            bounds_max = self.renderer.vertices.max(axis=0)
            margin = 0.2  # Stay away from edges
            
            if not np.all((position >= bounds_min + margin) & (position <= bounds_max - margin)):
                return False
            
            # Additional check: stay away from mesh surface
            min_distance = self.renderer.check_distance_to_surface(position)
            if min_distance < self.collision_threshold * 2:
                return False
                
            return True
            
        except Exception as e:
            # Fallback to simple sphere constraint
            return np.linalg.norm(position) < 0.8

    def _get_forward_vector(self) -> np.ndarray:
        """Get camera's forward direction vector.

        Returns:
            Forward vector (unit length)
        """
        roll, pitch, yaw = self.camera_rotation

        # Forward direction in camera space (negative Z)
        forward = np.array([
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
            np.cos(pitch) * np.cos(yaw),
        ])

        return forward / np.linalg.norm(forward)

    def _update_coverage(self) -> float:
        """Update coverage tracking and return new coverage gained.

        Returns:
            New coverage fraction
        """
        # Simple voxel-based coverage
        voxel_idx = tuple(
            ((self.camera_position + 1) * self.coverage_grid_size / 2).astype(int)
        )

        if voxel_idx not in self.visited_cells:
            self.visited_cells.add(voxel_idx)
            return 1.0 / (self.coverage_grid_size ** 3)
        else:
            return 0.0

    def _compute_reward(
        self, action: int, collision: bool, new_coverage: float
    ) -> float:
        """Compute reward for current step.

        Args:
            action: Action taken
            collision: Whether collision occurred
            new_coverage: New coverage gained

        Returns:
            Reward value
        """
        reward = 0.0

        # Coverage reward
        if new_coverage > 0:
            reward += self.alpha * self.coverage_per_new_cell

        # Collision penalty
        if collision:
            reward += self.gamma * self.collision_penalty

        # Action jerk penalty (discourage rapid changes)
        if self.previous_action is not None and action != self.previous_action:
            reward += self.delta * self.jerk_penalty_factor

        # Episode completion bonus
        if action == self.ACTION_DONE and not collision:
            reward += self.episode_completion_bonus

        # Anomaly detection bonus (placeholder - will be filled by CNN)
        # This is set externally when CNN is available
        # reward += self.beta * cnn_anomaly_score

        return reward

    def _get_observation(self) -> np.ndarray:
        """Get current observation (rendered frame).

        Returns:
            RGB image array
        """
        image, _ = self.renderer.render(return_depth=False)

        # Add camera noise if configured
        noise_std = self.camera_noise_config.get("gaussian_std", 0.0)
        if noise_std > 0:
            image = self.renderer.add_camera_noise(image, noise_std)

        return image

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary.

        Returns:
            Info dict with episode statistics
        """
        coverage = len(self.visited_cells) / (self.coverage_grid_size ** 3)

        info = {
            "step": self.current_step,
            "camera_position": self.camera_position.tolist(),
            "camera_rotation": self.camera_rotation.tolist(),
            "n_lesions": len(self.lesions),
            "coverage": coverage,
            "total_reward": self.total_reward,
        }

        # Add lesion bounding boxes if available
        if len(self.lesions) > 0:
            # This would require projection (simplified here)
            info["lesion_centers"] = [l.center.tolist() for l in self.lesions]

        return info

    def render(self) -> Optional[np.ndarray]:
        """Render the environment.

        Returns:
            RGB image if render_mode is 'rgb_array', else None
        """
        if self.render_mode == "rgb_array":
            return self._get_observation()
        return None

    def close(self):
        """Cleanup environment resources."""
        if self.renderer:
            self.renderer.cleanup()


# Register environment with gymnasium
gym.register(
    id="EndoscopyEnv-v0",
    entry_point="backend.sim.env:EndoscopyEnv",
)

