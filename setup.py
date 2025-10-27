"""Setup script for H. pylori RL Simulator."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "backend" / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = requirements_file.read_text(encoding="utf-8").splitlines()
    # Filter out comments and empty lines
    requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]

setup(
    name="hpylori-rl-sim",
    version="1.0.0",
    author="H. pylori RL Simulator Contributors",
    description="Research prototype for H. pylori CDSS 3D Endoscopy RL Simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hpylori-rl-sim",
    packages=find_packages(include=["backend", "backend.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.10.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hpylori-api=backend.api.main:main",
            "hpylori-train-cnn=backend.models.cnn.train_cnn:main",
            "hpylori-train-rl=backend.models.rl.train_rl:main",
            "hpylori-render=backend.sim.renderer:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    keywords=[
        "reinforcement-learning",
        "medical-simulation",
        "endoscopy",
        "computer-vision",
        "research",
        "h-pylori",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/hpylori-rl-sim/issues",
        "Source": "https://github.com/yourusername/hpylori-rl-sim",
    },
)

