"""Enhanced Gymnasium environment with multi-disease detection capabilities."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
from dataclasses import dataclass

from backend.sim.renderer import EndoscopyRenderer
from backend.sim.lesion_synth import LesionSynthesizer
from backend.utils.config import load_yaml_config


@dataclass
class DiseaseCondition:
    """Represents a disease condition in the GI tract."""
    
    type: str  # "h_pylori_gastritis", "peptic_ulcer", "tumor", "inflammation", "normal"
    center: np.ndarray  # 3D position
    radius: float  # Affected area radius
    severity: float  # 0.0-1.0
    detected: bool = False  # Whether correctly flagged by agent
    confidence: float = 0.0  # CNN confidence when detected


class EndoscopyEnvEnhanced(gym.Env):
    """Enhanced Gymnasium environment for multi-disease detection and clinical decision support.
    
    This environment simulates a virtual endoscopy with multiple pathological conditions:
    - H. pylori gastritis
    - Peptic ulcers
    - Early-stage tumors
    - Inflammation
    - Normal tissue
    
    The agent must:
    1. Navigate the GI tract efficiently
    2. Detect and flag abnormal regions
    3. Maximize coverage while minimizing unnecessary actions
    4. Achieve high sensitivity and specificity
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    # Navigation actions (0-7)
    ACTION_YAW_POS = 0
    ACTION_YAW_NEG = 1
    ACTION_PITCH_POS = 2
    ACTION_PITCH_NEG = 3
    ACTION_FORWARD = 4
    ACTION_BACKWARD = 5
    ACTION_ZOOM_IN = 6
    ACTION_ZOOM_OUT = 7
    
    # Diagnostic actions (8-11)
    ACTION_FLAG_REGION = 8      # Flag current view as abnormal
    ACTION_TAKE_BIOPSY = 9      # Take detailed snapshot (high-res)
    ACTION_REQUEST_AI = 10       # Request CNN inference
    ACTION_DONE = 11             # Finish examination
    
    # Disease types
    DISEASE_TYPES = [
        "normal",
        "h_pylori_gastritis",
        "peptic_ulcer", 
        "gastric_tumor",
        "inflammation",
    ]

    def __init__(
        self,
        config_path: Optional[str] = None,
        gltf_path: Optional[str] = None,
        render_mode: Optional[str] = "rgb_array",
        curriculum_stage: str = "easy",
        use_cnn: bool = False,
        cnn_model = None,
    ):
        """Initialize enhanced endoscopy environment.

        Args:
            config_path: Path to YAML configuration file
            gltf_path: Path or URL to GLTF model
            render_mode: Rendering mode
            curriculum_stage: Curriculum difficulty stage
            use_cnn: Whether to use CNN for anomaly detection
            cnn_model: Pre-trained CNN model for inference
        """
        super().__init__()

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

        # Disease generation settings
        self.diseases_min = 1
        self.diseases_max = 5
        self.disease_detection_radius = 0.3  # How close camera must be to detect

        # Domain randomization
        self.lighting_config = env_config.get("lighting", {})
        self.camera_noise_config = env_config.get("camera_noise", {})

        # Collision settings
        self.collision_threshold = env_config.get("collision_threshold", 0.02)

        # Coverage tracking
        self.coverage_grid_size = env_config.get("coverage_grid_size", 32)

        # Enhanced reward coefficients
        reward_config = self.config.get("reward", {})
        self.reward_coverage = reward_config.get("alpha", 0.3)  # Coverage importance reduced
        self.reward_detection = reward_config.get("beta", 0.6)  # Detection is now primary
        self.reward_collision = reward_config.get("gamma", -0.3)  # Collision penalty
        self.reward_efficiency = reward_config.get("delta", 0.1)  # Efficiency bonus
        
        # Detailed reward components
        self.true_positive_reward = 2.0   # Correctly flag disease
        self.false_positive_penalty = -0.5  # Incorrectly flag normal tissue
        self.false_negative_penalty = -1.0  # Miss disease
        self.coverage_per_cell = 0.05
        self.collision_penalty = -1.0
        self.efficiency_bonus = 0.01  # Per unused step
        self.completion_bonus = 5.0

        # Curriculum learning
        self.curriculum_stage = curriculum_stage
        self._apply_curriculum()

        # Gymnasium spaces
        self.action_space = spaces.Discrete(12)  # 8 navigation + 4 diagnostic
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.render_height, self.render_width, 3),
            dtype=np.uint8,
        )

        # CNN model for inference
        self.use_cnn = use_cnn
        self.cnn_model = cnn_model

        # Initialize renderer and lesion synthesizer
        self.renderer = EndoscopyRenderer(
            width=self.render_width,
            height=self.render_height,
            fov=self.fov,
            near_clip=self.near_clip,
            far_clip=self.far_clip,
        )
        self.lesion_synth = LesionSynthesizer()

        # Load GLTF model if provided
        self.gltf_path = gltf_path
        if gltf_path:
            self.load_model(gltf_path)

        # Episode state
        self.current_step = 0
        self.camera_position = self.initial_position.copy()
        self.camera_rotation = self.initial_rotation.copy()
        self.diseases: List[DiseaseCondition] = []
        self.flagged_positions: List[Tuple[np.ndarray, str, float]] = []  # (position, disease_type, confidence)
        self.visited_cells = set()
        self.camera_trajectory = []
        self.previous_action = None
        self.total_reward = 0.0
        self.zoom_level = 1.0
        
        # Diagnostic metrics
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.cnn_inferences = 0

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
                if "diseases_per_episode" in difficulty:
                    self.diseases_max = difficulty["diseases_per_episode"]
                if "detection_difficulty" in difficulty:
                    # Make detection harder (smaller radius)
                    self.disease_detection_radius *= difficulty["detection_difficulty"]
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
        self.zoom_level = 1.0
        self.flagged_positions = []
        
        # Reset diagnostic metrics
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.cnn_inferences = 0

        # Generate disease conditions
        if self.renderer.vertices is not None:
            self.diseases = self._generate_diseases()
        else:
            self.diseases = []

        # Render initial observation
        observation = self._get_observation()

        # Build info dict
        info = self._get_info()

        return observation, info

    def _generate_diseases(self) -> List[DiseaseCondition]:
        """Generate random disease conditions on the mesh.

        Returns:
            List of DiseaseCondition objects
        """
        diseases = []
        
        n_diseases = self.np_random.integers(self.diseases_min, self.diseases_max + 1)
        
        # Generate diseases of different types
        for _ in range(n_diseases):
            # Random disease type
            disease_type = self.np_random.choice(self.DISEASE_TYPES[1:])  # Exclude "normal"
            
            # Random location on mesh
            vertex_idx = self.np_random.integers(0, len(self.renderer.vertices))
            center = self.renderer.vertices[vertex_idx]
            
            # Random severity
            severity = self.np_random.uniform(0.3, 1.0)
            
            # Radius based on disease type
            radius_map = {
                "h_pylori_gastritis": 0.15,
                "peptic_ulcer": 0.08,
                "gastric_tumor": 0.10,
                "inflammation": 0.20,
            }
            radius = radius_map.get(disease_type, 0.1) * self.np_random.uniform(0.8, 1.2)
            
            disease = DiseaseCondition(
                type=disease_type,
                center=center,
                radius=radius,
                severity=severity,
                detected=False,
                confidence=0.0,
            )
            diseases.append(disease)
        
        return diseases

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

        # Apply action and compute reward
        reward, terminated = self._apply_action(action)

        # Update camera trajectory
        self.camera_trajectory.append(self.camera_position.copy())

        # Check collision
        collision = self.renderer.check_collision(
            self.camera_position,
            threshold=self.collision_threshold,
        )
        
        if collision:
            reward += self.collision_penalty
            terminated = True

        # Update coverage
        new_coverage = self._update_coverage()
        if new_coverage > 0:
            reward += self.reward_coverage * self.coverage_per_cell

        self.total_reward += reward

        # Check truncation
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

    def _apply_action(self, action: int) -> Tuple[float, bool]:
        """Apply action to update camera pose or perform diagnostic action.

        Args:
            action: Action index

        Returns:
            Tuple of (reward, terminated)
        """
        reward = 0.0
        terminated = False

        # Navigation actions
        if action == self.ACTION_YAW_POS:
            self.camera_rotation[2] += self.yaw_delta
        elif action == self.ACTION_YAW_NEG:
            self.camera_rotation[2] -= self.yaw_delta
        elif action == self.ACTION_PITCH_POS:
            self.camera_rotation[1] += self.pitch_delta
        elif action == self.ACTION_PITCH_NEG:
            self.camera_rotation[1] -= self.pitch_delta
        elif action == self.ACTION_FORWARD:
            forward = self._get_forward_vector()
            self.camera_position += forward * self.forward_delta
        elif action == self.ACTION_BACKWARD:
            forward = self._get_forward_vector()
            self.camera_position -= forward * self.forward_delta
        elif action == self.ACTION_ZOOM_IN:
            self.zoom_level = min(3.0, self.zoom_level + self.zoom_delta)
        elif action == self.ACTION_ZOOM_OUT:
            self.zoom_level = max(0.5, self.zoom_level - self.zoom_delta)
        
        # Diagnostic actions
        elif action == self.ACTION_FLAG_REGION:
            reward += self._flag_region()
        elif action == self.ACTION_TAKE_BIOPSY:
            reward += self._take_biopsy()
        elif action == self.ACTION_REQUEST_AI:
            reward += self._request_ai_inference()
        elif action == self.ACTION_DONE:
            # Calculate final reward based on detection performance
            reward += self._compute_final_reward()
            terminated = True

        # Clamp rotation angles
        self.camera_rotation[1] = np.clip(self.camera_rotation[1], -np.pi/2, np.pi/2)

        # Update renderer
        self.renderer.set_camera_pose(self.camera_position, self.camera_rotation)

        return reward, terminated

    def _flag_region(self) -> float:
        """Flag the current view as containing an abnormality.

        Returns:
            Reward for flagging
        """
        reward = 0.0
        
        # Check if camera is near any disease
        disease_detected = None
        min_distance = float('inf')
        
        for disease in self.diseases:
            distance = np.linalg.norm(self.camera_position - disease.center)
            if distance < disease.radius and distance < min_distance:
                disease_detected = disease
                min_distance = distance
        
        # Get CNN prediction if available
        cnn_confidence = 0.5  # Default
        predicted_type = "normal"
        
        if self.use_cnn and self.cnn_model is not None:
            cnn_confidence, predicted_type = self._run_cnn_inference()
        
        # Record flagging
        self.flagged_positions.append((
            self.camera_position.copy(),
            predicted_type,
            cnn_confidence
        ))
        
        # Compute reward
        if disease_detected is not None:
            # True positive
            if not disease_detected.detected:
                reward = self.true_positive_reward * disease_detected.severity
                disease_detected.detected = True
                disease_detected.confidence = cnn_confidence
                self.true_positives += 1
            else:
                # Already detected, small penalty for redundancy
                reward = -0.1
        else:
            # False positive
            reward = self.false_positive_penalty
            self.false_positives += 1
        
        return reward

    def _take_biopsy(self) -> float:
        """Take a high-resolution snapshot (simulated biopsy).

        Returns:
            Reward for biopsy action
        """
        # Similar to flagging, but with higher reward if near disease
        reward = 0.0
        
        for disease in self.diseases:
            distance = np.linalg.norm(self.camera_position - disease.center)
            if distance < disease.radius * 0.5:  # Must be very close
                reward = 0.5 * disease.severity
                break
        
        return reward

    def _request_ai_inference(self) -> float:
        """Request CNN inference on current frame.

        Returns:
            Reward for AI request (slight penalty to encourage efficiency)
        """
        self.cnn_inferences += 1
        
        if self.use_cnn and self.cnn_model is not None:
            self._run_cnn_inference()
        
        # Small penalty to encourage efficiency
        return -0.05

    def _run_cnn_inference(self) -> Tuple[float, str]:
        """Run CNN inference on current observation.

        Returns:
            Tuple of (confidence, predicted_disease_type)
        """
        if not self.use_cnn or self.cnn_model is None:
            return 0.5, "normal"
        
        # This would call the actual CNN model
        # For now, return placeholder
        # In real implementation:
        # obs = self._get_observation()
        # prediction = self.cnn_model.predict(obs)
        # return prediction
        
        return 0.5, "normal"

    def _compute_final_reward(self) -> float:
        """Compute final reward when episode ends.

        Returns:
            Final reward bonus/penalty
        """
        reward = 0.0
        
        # Count undetected diseases (false negatives)
        for disease in self.diseases:
            if not disease.detected:
                reward += self.false_negative_penalty * disease.severity
                self.false_negatives += 1
        
        # Efficiency bonus for finishing early
        steps_saved = max(0, self.max_steps - self.current_step)
        reward += steps_saved * self.efficiency_bonus
        
        # Completion bonus if good performance
        if self.true_positives > 0 and self.false_positives < self.true_positives:
            reward += self.completion_bonus
        
        return reward

    def _get_forward_vector(self) -> np.ndarray:
        """Get camera's forward direction vector.

        Returns:
            Forward vector (unit length)
        """
        roll, pitch, yaw = self.camera_rotation

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
        voxel_idx = tuple(
            ((self.camera_position + 1) * self.coverage_grid_size / 2).astype(int)
        )

        if voxel_idx not in self.visited_cells:
            self.visited_cells.add(voxel_idx)
            return 1.0 / (self.coverage_grid_size ** 3)
        else:
            return 0.0

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
        """Get info dictionary with enhanced metrics.

        Returns:
            Info dict with episode statistics and diagnostic metrics
        """
        coverage = len(self.visited_cells) / (self.coverage_grid_size ** 3)
        
        # Calculate detection metrics
        sensitivity = self.true_positives / max(1, len(self.diseases))
        precision = self.true_positives / max(1, self.true_positives + self.false_positives)
        f1_score = 2 * (precision * sensitivity) / max(0.001, precision + sensitivity)

        info = {
            "step": self.current_step,
            "camera_position": self.camera_position.tolist(),
            "camera_rotation": self.camera_rotation.tolist(),
            "zoom_level": self.zoom_level,
            "n_diseases": len(self.diseases),
            "coverage": coverage,
            "total_reward": self.total_reward,
            
            # Diagnostic metrics
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "sensitivity": sensitivity,
            "precision": precision,
            "f1_score": f1_score,
            "cnn_inferences": self.cnn_inferences,
            
            # Disease information
            "diseases": [
                {
                    "type": d.type,
                    "severity": d.severity,
                    "detected": d.detected,
                    "confidence": d.confidence,
                }
                for d in self.diseases
            ],
        }

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


# Register enhanced environment with gymnasium
gym.register(
    id="EndoscopyEnhanced-v0",
    entry_point="backend.sim.env_enhanced:EndoscopyEnvEnhanced",
)

