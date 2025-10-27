"""Synthetic lesion generation for GI tract surfaces."""

import numpy as np
from typing import List, Tuple, Optional, Dict
import cv2
from dataclasses import dataclass

# Simple Perlin noise alternative (no C++ compilation needed)
def simple_perlin(x: float, y: float, seed: int = 0) -> float:
    """Simple noise function as Perlin alternative."""
    np.random.seed(int((x * 12.9898 + y * 78.233 + seed) * 43758.5453) % 2**31)
    return np.random.random() * 2 - 1


@dataclass
class Lesion:
    """Represents a synthetic lesion."""
    center: np.ndarray  # 3D position
    radius: float
    color: np.ndarray  # RGB
    mask: Optional[np.ndarray] = None  # 2D mask
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)


class LesionSynthesizer:
    """Generates synthetic ulcer-like lesions on 3D surfaces."""

    def __init__(
        self,
        base_color: Tuple[int, int, int] = (180, 50, 50),
        color_variation: int = 30,
        perlin_scale: float = 10.0,
        perlin_octaves: int = 4,
        seed: Optional[int] = None,
    ):
        """Initialize lesion synthesizer.

        Args:
            base_color: Base RGB color for lesions (reddish-brown)
            color_variation: Random variation in color
            perlin_scale: Scale for Perlin noise (controls irregularity)
            perlin_octaves: Octaves for Perlin noise
            seed: Random seed for reproducibility
        """
        self.base_color = np.array(base_color)
        self.color_variation = color_variation
        self.perlin_scale = perlin_scale
        self.perlin_octaves = perlin_octaves
        
        if seed is not None:
            np.random.seed(seed)
        
        self.rng = np.random.default_rng(seed)

    def generate_lesion_mask(
        self,
        size: int = 128,
        irregularity: float = 0.5,
    ) -> np.ndarray:
        """Generate an irregular lesion mask using Perlin noise.

        Args:
            size: Size of the mask (square)
            irregularity: Irregularity factor (0-1), higher = more irregular

        Returns:
            Binary mask (0 or 1) of shape (size, size)
        """
        # Create coordinate grid
        x = np.linspace(0, self.perlin_scale, size)
        y = np.linspace(0, self.perlin_scale, size)
        
        # Generate Perlin noise
        noise_map = np.zeros((size, size))
        offset_x = self.rng.uniform(0, 1000)
        offset_y = self.rng.uniform(0, 1000)
        
        for i in range(size):
            for j in range(size):
                # Use simple noise instead of Perlin (no C++ compiler needed)
                noise_map[i, j] = simple_perlin(
                    x[j] + offset_x,
                    y[i] + offset_y,
                    seed=int(offset_x * 1000)
                )
        
        # Normalize to 0-1
        noise_map = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
        
        # Create circular base
        center = size // 2
        Y, X = np.ogrid[:size, :size]
        dist_from_center = np.sqrt((X - center)**2 + (Y - center)**2)
        max_radius = size // 2
        circular_mask = (dist_from_center <= max_radius).astype(float)
        
        # Combine circle with noise for irregularity
        combined = circular_mask * (1 - irregularity) + noise_map * irregularity
        
        # Threshold to binary mask
        threshold = 0.4 + self.rng.uniform(-0.1, 0.1)
        mask = (combined > threshold).astype(np.uint8)
        
        # Morphological operations to smooth edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask

    def generate_lesion_texture(
        self,
        mask: np.ndarray,
        background_color: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate lesion texture with color variation.

        Args:
            mask: Binary mask defining lesion shape
            background_color: Background mucosal color (optional)

        Returns:
            RGB texture of shape (H, W, 3)
        """
        h, w = mask.shape
        texture = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Default mucosal background color (pinkish)
        if background_color is None:
            background_color = np.array([220, 180, 180])
        
        # Fill background
        texture[:, :] = background_color
        
        # Generate varied lesion color
        lesion_pixels = mask > 0
        n_lesion_pixels = lesion_pixels.sum()
        
        if n_lesion_pixels > 0:
            # Add spatial color variation
            for c in range(3):
                base = self.base_color[c]
                variation = self.rng.normal(0, self.color_variation, (h, w))
                color_channel = np.clip(base + variation, 0, 255).astype(np.uint8)
                texture[:, :, c][lesion_pixels] = color_channel[lesion_pixels]
            
            # Add some darker spots for realism (depth cues)
            dark_spots = (self.rng.random((h, w)) > 0.8) & lesion_pixels
            texture[dark_spots] = (texture[dark_spots] * 0.6).astype(np.uint8)
        
        return texture

    def sample_surface_points(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        n_samples: int = 1,
        min_distance: float = 0.1,
    ) -> List[np.ndarray]:
        """Sample random points on mesh surface for lesion placement.

        Args:
            vertices: Mesh vertices (N, 3)
            faces: Mesh faces (M, 3)
            n_samples: Number of points to sample
            min_distance: Minimum distance between sampled points

        Returns:
            List of 3D points
        """
        sampled_points = []
        max_attempts = n_samples * 100
        attempts = 0
        
        while len(sampled_points) < n_samples and attempts < max_attempts:
            attempts += 1
            
            # Randomly select a face
            face_idx = self.rng.integers(0, len(faces))
            face = faces[face_idx]
            
            # Get triangle vertices
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            
            # Random barycentric coordinates
            r1, r2 = self.rng.random(2)
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            
            # Interpolate point on triangle
            point = v0 + r1 * (v1 - v0) + r2 * (v2 - v0)
            
            # Check minimum distance
            if len(sampled_points) == 0:
                sampled_points.append(point)
            else:
                distances = [np.linalg.norm(point - p) for p in sampled_points]
                if min(distances) >= min_distance:
                    sampled_points.append(point)
        
        return sampled_points

    def create_lesions(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        n_lesions: int = 3,
        size_range: Tuple[float, float] = (0.01, 0.05),
        irregularity: float = 0.5,
    ) -> List[Lesion]:
        """Create multiple lesions on a mesh surface.

        Args:
            vertices: Mesh vertices (N, 3)
            faces: Mesh faces (M, 3)
            n_lesions: Number of lesions to create
            size_range: (min_radius, max_radius) in meters
            irregularity: Lesion shape irregularity (0-1)

        Returns:
            List of Lesion objects
        """
        # Sample lesion centers
        min_distance = size_range[1] * 2  # Prevent overlap
        centers = self.sample_surface_points(
            vertices, faces, n_lesions, min_distance
        )
        
        lesions = []
        for center in centers:
            # Random radius
            radius = self.rng.uniform(size_range[0], size_range[1])
            
            # Generate mask
            mask_size = 128
            mask = self.generate_lesion_mask(mask_size, irregularity)
            
            # Generate color with variation
            color = self.base_color + self.rng.integers(
                -self.color_variation,
                self.color_variation,
                size=3
            )
            color = np.clip(color, 0, 255)
            
            lesion = Lesion(
                center=center,
                radius=radius,
                color=color,
                mask=mask,
            )
            lesions.append(lesion)
        
        return lesions

    def project_lesions_to_image(
        self,
        lesions: List[Lesion],
        camera_matrix: np.ndarray,
        image_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Project 3D lesions to 2D image space.

        Args:
            lesions: List of 3D lesions
            camera_matrix: 4x4 camera projection matrix
            image_size: (width, height) of output image

        Returns:
            Tuple of (label_mask, bounding_boxes)
            - label_mask: Binary mask (H, W) indicating lesion pixels
            - bounding_boxes: List of dicts with keys 'x', 'y', 'w', 'h', 'lesion_id'
        """
        width, height = image_size
        label_mask = np.zeros((height, width), dtype=np.uint8)
        bounding_boxes = []
        
        for lesion_id, lesion in enumerate(lesions):
            # Project center to 2D
            center_3d = np.append(lesion.center, 1)  # Homogeneous coords
            center_2d_h = camera_matrix @ center_3d
            
            # Skip if behind camera
            if center_2d_h[2] <= 0:
                continue
            
            # Perspective divide
            center_2d = center_2d_h[:2] / center_2d_h[2]
            
            # Convert to pixel coordinates (NDC to screen space)
            cx = int((center_2d[0] + 1) * width / 2)
            cy = int((1 - center_2d[1]) * height / 2)
            
            # Estimate projected radius (simple approximation)
            distance = np.linalg.norm(lesion.center - camera_matrix[:3, 3])
            pixel_radius = int(lesion.radius / distance * min(width, height))
            
            if pixel_radius < 2:
                continue
            
            # Resize lesion mask to projected size
            mask_size = pixel_radius * 2
            if mask_size > 0:
                resized_mask = cv2.resize(
                    lesion.mask,
                    (mask_size, mask_size),
                    interpolation=cv2.INTER_LINEAR
                )
                
                # Compute bounding box
                x1 = max(0, cx - pixel_radius)
                y1 = max(0, cy - pixel_radius)
                x2 = min(width, cx + pixel_radius)
                y2 = min(height, cy + pixel_radius)
                
                # Paste mask onto label
                mask_x1 = max(0, pixel_radius - cx)
                mask_y1 = max(0, pixel_radius - cy)
                mask_x2 = mask_x1 + (x2 - x1)
                mask_y2 = mask_y1 + (y2 - y1)
                
                if mask_x2 > mask_x1 and mask_y2 > mask_y1:
                    label_mask[y1:y2, x1:x2] = np.maximum(
                        label_mask[y1:y2, x1:x2],
                        resized_mask[mask_y1:mask_y2, mask_x1:mask_x2]
                    )
                    
                    # Add bounding box
                    bounding_boxes.append({
                        'lesion_id': lesion_id,
                        'x': x1,
                        'y': y1,
                        'w': x2 - x1,
                        'h': y2 - y1,
                        'center_3d': lesion.center.tolist(),
                    })
        
        return label_mask, bounding_boxes

