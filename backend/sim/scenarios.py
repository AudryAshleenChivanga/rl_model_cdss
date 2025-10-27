"""Clinical disease scenarios for endoscopy simulation."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Lesion:
    """Represents a single lesion/abnormality."""
    type: str  # "ulcer", "erosion", "polyp", "tumor", "inflammation", "h_pylori"
    position: np.ndarray  # 3D position on mesh
    size: float  # Lesion diameter
    severity: str  # "mild", "moderate", "severe"
    color: Tuple[int, int, int]  # RGB color


@dataclass
class ClinicalScenario:
    """Represents a clinical disease scenario."""
    id: str
    name: str
    description: str
    lesion_types: List[str]
    lesion_count_range: Tuple[int, int]
    severity_distribution: dict
    background_inflammation: bool
    difficulty: str  # "easy", "medium", "hard"


# Define clinical scenarios
SCENARIOS = {
    "healthy": ClinicalScenario(
        id="healthy",
        name="Healthy Mucosa",
        description="Normal gastric mucosa with no pathological findings",
        lesion_types=[],
        lesion_count_range=(0, 0),
        severity_distribution={},
        background_inflammation=False,
        difficulty="easy"
    ),
    
    "peptic_ulcer": ClinicalScenario(
        id="peptic_ulcer",
        name="Peptic Ulcer Disease",
        description="Single or multiple gastric/duodenal ulcers with inflamed borders",
        lesion_types=["ulcer", "erosion"],
        lesion_count_range=(1, 3),
        severity_distribution={"mild": 0.2, "moderate": 0.5, "severe": 0.3},
        background_inflammation=True,
        difficulty="medium"
    ),
    
    "h_pylori_gastritis": ClinicalScenario(
        id="h_pylori_gastritis",
        name="H. pylori Gastritis",
        description="Chronic gastritis caused by Helicobacter pylori infection",
        lesion_types=["h_pylori", "erosion", "inflammation"],
        lesion_count_range=(3, 8),
        severity_distribution={"mild": 0.3, "moderate": 0.6, "severe": 0.1},
        background_inflammation=True,
        difficulty="medium"
    ),
    
    "gastric_cancer": ClinicalScenario(
        id="gastric_cancer",
        name="Gastric Adenocarcinoma",
        description="Malignant tumor with irregular borders and ulceration",
        lesion_types=["tumor", "ulcer"],
        lesion_count_range=(1, 2),
        severity_distribution={"severe": 1.0},
        background_inflammation=True,
        difficulty="hard"
    ),
    
    "erosive_gastritis": ClinicalScenario(
        id="erosive_gastritis",
        name="Erosive Gastritis",
        description="Multiple superficial erosions with surrounding erythema",
        lesion_types=["erosion", "inflammation"],
        lesion_count_range=(5, 15),
        severity_distribution={"mild": 0.4, "moderate": 0.5, "severe": 0.1},
        background_inflammation=True,
        difficulty="medium"
    ),
    
    "polyps": ClinicalScenario(
        id="polyps",
        name="Gastric Polyps",
        description="Benign polyps (hyperplastic or fundic gland polyps)",
        lesion_types=["polyp"],
        lesion_count_range=(1, 5),
        severity_distribution={"mild": 0.8, "moderate": 0.2},
        background_inflammation=False,
        difficulty="easy"
    ),
    
    "mixed_pathology": ClinicalScenario(
        id="mixed_pathology",
        name="Mixed Pathology",
        description="Multiple concurrent conditions (ulcer + gastritis + erosions)",
        lesion_types=["ulcer", "erosion", "inflammation", "h_pylori"],
        lesion_count_range=(4, 10),
        severity_distribution={"mild": 0.3, "moderate": 0.4, "severe": 0.3},
        background_inflammation=True,
        difficulty="hard"
    ),
}


class ScenarioGenerator:
    """Generates lesions based on clinical scenarios."""
    
    # Color definitions for different lesion types
    LESION_COLORS = {
        "ulcer": ([220, 180, 180], 20),  # Pale with crater
        "erosion": ([200, 120, 120], 15),  # Reddish-brown
        "polyp": ([240, 200, 190], 10),  # Pale pink/flesh-colored
        "tumor": ([180, 140, 140], 25),  # Irregular, darker
        "inflammation": ([210, 90, 90], 20),  # Bright red/inflamed
        "h_pylori": ([190, 100, 100], 15),  # Reddish with spots
    }
    
    # Size ranges for different lesion types (in model units)
    LESION_SIZES = {
        "ulcer": (0.05, 0.15),
        "erosion": (0.02, 0.06),
        "polyp": (0.03, 0.10),
        "tumor": (0.08, 0.25),
        "inflammation": (0.03, 0.08),
        "h_pylori": (0.02, 0.05),
    }
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize scenario generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
    
    def generate_scenario(
        self,
        scenario_id: str,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> List[Lesion]:
        """Generate lesions for a specific scenario.
        
        Args:
            scenario_id: Scenario identifier
            vertices: Mesh vertices
            faces: Mesh faces
            
        Returns:
            List of lesions
        """
        if scenario_id not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_id}")
        
        scenario = SCENARIOS[scenario_id]
        
        # Determine number of lesions
        n_lesions = self.rng.integers(*scenario.lesion_count_range, endpoint=True)
        
        if n_lesions == 0:
            return []
        
        # Generate lesions
        lesions = []
        for _ in range(n_lesions):
            # Choose lesion type
            lesion_type = self.rng.choice(scenario.lesion_types)
            
            # Choose severity
            severity = self._sample_severity(scenario.severity_distribution)
            
            # Choose random position on mesh
            face_idx = self.rng.integers(0, len(faces))
            face = faces[face_idx]
            # Barycentric coordinates for point on triangle
            u, v = self.rng.random(2)
            if u + v > 1:
                u, v = 1 - u, 1 - v
            w = 1 - u - v
            
            position = (
                u * vertices[face[0]] +
                v * vertices[face[1]] +
                w * vertices[face[2]]
            )
            
            # Determine size based on type and severity
            size_min, size_max = self.LESION_SIZES[lesion_type]
            severity_multiplier = {"mild": 0.7, "moderate": 1.0, "severe": 1.5}[severity]
            size = self.rng.uniform(size_min, size_max) * severity_multiplier
            
            # Determine color
            base_color, variation = self.LESION_COLORS[lesion_type]
            color = tuple(
                int(np.clip(c + self.rng.integers(-variation, variation), 0, 255))
                for c in base_color
            )
            
            lesion = Lesion(
                type=lesion_type,
                position=position,
                size=size,
                severity=severity,
                color=color
            )
            lesions.append(lesion)
        
        return lesions
    
    def _sample_severity(self, severity_dist: dict) -> str:
        """Sample severity level from distribution.
        
        Args:
            severity_dist: Severity distribution dict
            
        Returns:
            Severity level string
        """
        if not severity_dist:
            return "mild"
        
        severities = list(severity_dist.keys())
        probs = list(severity_dist.values())
        return self.rng.choice(severities, p=probs)


def get_scenario_list() -> List[dict]:
    """Get list of available scenarios for UI.
    
    Returns:
        List of scenario info dicts
    """
    return [
        {
            "id": s.id,
            "name": s.name,
            "description": s.description,
            "difficulty": s.difficulty,
        }
        for s in SCENARIOS.values()
    ]

