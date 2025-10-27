"""3D rendering for endoscopy simulation using pyrender."""

import os
import sys
import numpy as np
import trimesh
from typing import Optional, Tuple, Dict
import cv2
from pathlib import Path
import tempfile
import requests

# Try to import pyrender - it may not work on Windows without OpenGL
PYRENDER_AVAILABLE = False
try:
    # On Windows, try without EGL first
    if sys.platform == 'win32':
        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'  # Try OSMesa on Windows
    else:
        os.environ['PYOPENGL_PLATFORM'] = 'egl'  # Use EGL on Linux
    
    import pyrender
    PYRENDER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: pyrender not available ({e}). 3D rendering will be disabled.")
    pyrender = None


class EndoscopyRenderer:
    """Renders 3D GI tract scenes for endoscopy simulation."""

    def __init__(
        self,
        width: int = 224,
        height: int = 224,
        fov: float = 70.0,
        near_clip: float = 0.01,
        far_clip: float = 10.0,
        ambient_light: float = 0.4,
        directional_light: float = 0.8,
    ):
        """Initialize renderer.

        Args:
            width: Render width in pixels
            height: Render height in pixels
            fov: Field of view in degrees
            near_clip: Near clipping plane
            far_clip: Far clipping plane
            ambient_light: Ambient light intensity
            directional_light: Directional light intensity
        """
        if not PYRENDER_AVAILABLE:
            print("Warning: Renderer initialized but pyrender is not available")
        
        self.width = width
        self.height = height
        self.fov = fov
        self.near_clip = near_clip
        self.far_clip = far_clip
        self.ambient_light = ambient_light
        self.directional_light = directional_light

        # Initialize scene
        self.scene = None
        self.mesh_node = None
        self.camera_node = None
        self.renderer = None
        
        # Mesh data
        self.mesh = None
        self.vertices = None
        self.faces = None

    def load_gltf(self, gltf_path: str) -> None:
        """Load GLTF/GLB model.

        Args:
            gltf_path: Path to GLTF/GLB file or URL
        """
        try:
            # Download if URL
            if gltf_path.startswith('http://') or gltf_path.startswith('https://'):
                gltf_path = self._download_model(gltf_path)
            
            # Load mesh with trimesh
            mesh = trimesh.load(gltf_path, force='mesh')
            
            # Handle scene vs single mesh
            if isinstance(mesh, trimesh.Scene):
                # Merge all meshes in scene
                mesh = mesh.dump(concatenate=True)
            
            self.mesh = mesh
            self.vertices = np.array(mesh.vertices)
            self.faces = np.array(mesh.faces)
            
            # Center and scale mesh
            self._normalize_mesh()
            
            print(f"Successfully loaded GLTF model: {len(self.vertices)} vertices, {len(self.faces)} faces")
            
        except Exception as e:
            print(f"Warning: Failed to load GLTF model ({e})")
            print("Continuing with dummy geometry for simulation")
            # Create dummy geometry so simulation can still run
            self.mesh = None
            self.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            self.faces = np.array([[0, 1, 2], [0, 1, 3]])

    def _download_model(self, url: str) -> str:
        """Download model from URL to temporary file.

        Args:
            url: Model URL

        Returns:
            Path to downloaded file
        """
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        # Determine file extension
        ext = '.glb' if 'glb' in url.lower() else '.gltf'
        
        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        temp_file.close()
        
        return temp_file.name

    def _normalize_mesh(self) -> None:
        """Center and scale mesh to unit size."""
        if self.mesh is None:
            return
        
        # Center at origin
        center = self.mesh.bounds.mean(axis=0)
        self.mesh.vertices -= center
        self.vertices = np.array(self.mesh.vertices)
        
        # Scale to unit sphere
        max_extent = np.abs(self.mesh.vertices).max()
        if max_extent > 0:
            self.mesh.vertices /= max_extent
            self.vertices = np.array(self.mesh.vertices)

    def setup_scene(self) -> None:
        """Setup pyrender scene with mesh, camera, and lights."""
        if not PYRENDER_AVAILABLE:
            print("Warning: Cannot setup scene - pyrender not available")
            return
            
        self.scene = pyrender.Scene(
            ambient_light=np.array([self.ambient_light] * 3)
        )
        
        # Add mesh
        if self.mesh is not None:
            # Convert to pyrender mesh
            pr_mesh = pyrender.Mesh.from_trimesh(self.mesh)
            self.mesh_node = self.scene.add(pr_mesh)
        
        # Add camera
        camera = pyrender.PerspectiveCamera(
            yfov=np.radians(self.fov),
            aspectRatio=self.width / self.height,
            znear=self.near_clip,
            zfar=self.far_clip,
        )
        self.camera_node = self.scene.add(camera, pose=np.eye(4))
        
        # Add directional light (follows camera)
        light = pyrender.DirectionalLight(
            color=np.ones(3),
            intensity=self.directional_light
        )
        self.scene.add(light, pose=np.eye(4))
        
        # Initialize renderer
        try:
            self.renderer = pyrender.OffscreenRenderer(self.width, self.height)
        except Exception as e:
            # Fallback to EGL or OSMesa
            print(f"Failed to initialize renderer: {e}")
            print("Trying with EGL backend...")
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
            self.renderer = pyrender.OffscreenRenderer(self.width, self.height)

    def set_camera_pose(self, position: np.ndarray, rotation: np.ndarray) -> None:
        """Set camera pose.

        Args:
            position: Camera position (x, y, z)
            rotation: Camera rotation as Euler angles (roll, pitch, yaw) in radians
        """
        # Store pose for reference
        self.camera_position = position.copy()
        self.camera_rotation = rotation.copy()
        
        # If rendering not available, just store the pose
        if not PYRENDER_AVAILABLE or self.camera_node is None or self.scene is None:
            return
        
        # Build rotation matrix from Euler angles
        roll, pitch, yaw = rotation
        
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation
        R = Rz @ Ry @ Rx
        
        # Build 4x4 pose matrix
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = position
        
        # Update camera
        self.scene.set_pose(self.camera_node, pose)

    def render(
        self,
        return_depth: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Render the current scene.

        Args:
            return_depth: Whether to return depth map

        Returns:
            Tuple of (color_image, depth_map)
            - color_image: RGB image (H, W, 3)
            - depth_map: Depth map (H, W) or None
        """
        if not PYRENDER_AVAILABLE:
            # Return a placeholder image if rendering not available
            placeholder = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            placeholder[:] = [50, 50, 100]  # Dark blue background
            # Add text indicating rendering not available
            cv2.putText(placeholder, "3D Rendering Unavailable", (10, self.height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            return placeholder, None
            
        if self.renderer is None:
            raise RuntimeError("Scene not setup. Call setup_scene() first.")
        
        # Render
        flags = pyrender.RenderFlags.NONE
        color, depth = self.renderer.render(self.scene, flags=flags)
        
        # Convert to RGB (from RGBA if needed)
        if color.shape[-1] == 4:
            color = color[:, :, :3]
        
        if return_depth:
            return color, depth
        else:
            return color, None

    def add_camera_noise(self, image: np.ndarray, noise_std: float = 0.02) -> np.ndarray:
        """Add Gaussian noise to simulate camera sensor noise.

        Args:
            image: Input image (H, W, 3)
            noise_std: Standard deviation of Gaussian noise

        Returns:
            Noisy image
        """
        noise = np.random.normal(0, noise_std * 255, image.shape)
        noisy = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
        return noisy

    def check_collision(
        self,
        camera_position: np.ndarray,
        threshold: float = 0.02,
    ) -> bool:
        """Check if camera collides with mesh surface.

        Args:
            camera_position: Camera position (x, y, z)
            threshold: Collision distance threshold

        Returns:
            True if collision detected
        """
        if self.vertices is None:
            return False
        
        # Find closest vertex
        distances = np.linalg.norm(self.vertices - camera_position, axis=1)
        min_distance = distances.min()
        
        return min_distance < threshold

    def compute_coverage(
        self,
        camera_positions: list,
        grid_size: int = 32,
    ) -> float:
        """Compute surface coverage from camera trajectory.

        Args:
            camera_positions: List of camera positions
            grid_size: Voxel grid resolution

        Returns:
            Coverage fraction (0-1)
        """
        if len(camera_positions) == 0:
            return 0.0
        
        # Create voxel grid
        if self.vertices is None:
            return 0.0
        
        bounds_min = self.vertices.min(axis=0)
        bounds_max = self.vertices.max(axis=0)
        voxel_size = (bounds_max - bounds_min) / grid_size
        
        # Mark visited voxels
        visited = set()
        for pos in camera_positions:
            voxel_idx = tuple(
                ((pos - bounds_min) / voxel_size).astype(int)
            )
            visited.add(voxel_idx)
        
        # Total possible voxels (approximate surface voxels)
        total_voxels = grid_size ** 3 // 10  # Rough surface estimate
        coverage = len(visited) / total_voxels
        
        return min(coverage, 1.0)

    def cleanup(self) -> None:
        """Cleanup renderer resources."""
        if self.renderer is not None:
            self.renderer.delete()
            self.renderer = None


def main():
    """Generate synthetic dataset for CNN training."""
    import argparse
    from tqdm import tqdm
    import json
    from backend.sim.lesion_synth import LesionSynthesizer

    parser = argparse.ArgumentParser(description='Generate synthetic endoscopy dataset')
    parser.add_argument('--export-dataset', type=str, required=True,
                        help='Output directory for dataset')
    parser.add_argument('--episodes', type=int, default=3000,
                        help='Number of episodes to generate')
    parser.add_argument('--frames-per-episode', type=int, default=10,
                        help='Frames per episode')
    parser.add_argument('--width', type=int, default=224, help='Image width')
    parser.add_argument('--height', type=int, default=224, help='Image height')
    parser.add_argument('--gltf-url', type=str,
                        default='https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Cube/glTF/Cube.gltf',
                        help='GLTF model URL')
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.export_dataset)
    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize renderer and lesion synthesizer
    renderer = EndoscopyRenderer(width=args.width, height=args.height)
    lesion_synth = LesionSynthesizer(seed=42)
    
    print(f"Loading model from {args.gltf_url}...")
    try:
        renderer.load_gltf(args.gltf_url)
        renderer.setup_scene()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using default cube mesh...")
        # Fallback to simple cube
        renderer.mesh = trimesh.creation.box(extents=[1, 1, 1])
        renderer.vertices = np.array(renderer.mesh.vertices)
        renderer.faces = np.array(renderer.mesh.faces)
        renderer.setup_scene()
    
    # Generate dataset
    metadata = []
    frame_idx = 0
    
    print(f"Generating {args.episodes} episodes...")
    for episode in tqdm(range(args.episodes)):
        # Generate random lesions
        n_lesions = np.random.randint(1, 5)
        lesions = lesion_synth.create_lesions(
            renderer.vertices,
            renderer.faces,
            n_lesions=n_lesions,
        )
        
        # Generate random camera trajectory
        for frame in range(args.frames_per_episode):
            # Random camera pose
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(-np.pi/4, np.pi/4)
            radius = np.random.uniform(0.3, 0.8)
            
            position = np.array([
                radius * np.cos(theta) * np.cos(phi),
                radius * np.sin(phi),
                radius * np.sin(theta) * np.cos(phi),
            ])
            
            rotation = np.array([
                0,
                phi,
                theta + np.pi,
            ])
            
            renderer.set_camera_pose(position, rotation)
            
            # Render frame
            image, _ = renderer.render()
            
            # Add noise
            image = renderer.add_camera_noise(image)
            
            # Save frame
            frame_name = f'frame_{frame_idx:06d}.jpg'
            cv2.imwrite(str(images_dir / frame_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # Create label (binary: has lesion or not)
            # For simplicity, mark as positive if any lesion is visible
            has_lesion = len(lesions) > 0 and np.random.random() > 0.3  # Simulate detection
            
            label_data = {
                'frame_id': frame_idx,
                'episode': episode,
                'frame': frame,
                'has_lesion': int(has_lesion),
                'n_lesions': len(lesions),
                'camera_position': position.tolist(),
                'camera_rotation': rotation.tolist(),
            }
            
            metadata.append(label_data)
            frame_idx += 1
    
    # Save metadata
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create labels CSV
    import pandas as pd
    df = pd.DataFrame(metadata)
    df[['frame_id', 'has_lesion']].to_csv(
        labels_dir / 'labels.csv',
        index=False
    )
    
    print(f"Dataset generated: {frame_idx} frames")
    print(f"Saved to: {output_dir}")
    
    renderer.cleanup()


if __name__ == '__main__':
    main()

